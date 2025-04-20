import argparse
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score # Import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import pandas as pd
from typing import Union, List, Tuple, Dict, Optional
from flwr.common import (
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
import warnings # To suppress KMeans warnings for small k

# Suppress ConvergenceWarning from KMeans when n_init=1
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.cluster._kmeans')
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide") # Silhouette score can warn

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def add_label_noise(dataset, symmetric_noise_ratio=0.0, asymmetric_noise_ratio=0.0):
    targets = np.array(dataset.targets)
    num_classes = 10
    num_samples = len(targets)
    indices = np.arange(num_samples)

    if symmetric_noise_ratio > 0:
        num_noisy = int(symmetric_noise_ratio * num_samples)
        noisy_indices = np.random.choice(indices, num_noisy, replace=False)
        for i in noisy_indices:
            new_label = np.random.choice([x for x in range(num_classes) if x != targets[i]])
            targets[i] = new_label

    if asymmetric_noise_ratio > 0:
        for c in range(num_classes):
            class_indices = np.where(targets == c)[0]
            num_noisy = int(asymmetric_noise_ratio * len(class_indices))
            noisy_indices = np.random.choice(class_indices, num_noisy, replace=False)
            for i in noisy_indices:
                new_label = (targets[i] + 1) % num_classes
                targets[i] = new_label

    dataset.targets = targets.tolist()
    return dataset

def get_client_dataloader(cid, num_clients, batch_size=32, train=True, noise_level=0.0):
    if train:
        client_dataset = CIFAR10(root="./data", train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
        # Apply noise *before* splitting data to ensure consistency if needed across splits
        # Note: In this setup, each client gets distinct data, so noise is applied per client partition effectively.
        num_samples_total = len(client_dataset)
        samples_per_client = num_samples_total // num_clients
        indices = list(range(cid * samples_per_client, (cid + 1) * samples_per_client))
        subset_for_noise = Subset(client_dataset, indices)

        # Create a temporary dataset to apply noise specific to this client's indices
        temp_targets = [client_dataset.targets[i] for i in indices]
        temp_dataset = type(client_dataset)(root="./data", train=True, download=False, transform=client_dataset.transform) # Create instance without downloading
        temp_dataset.data = [client_dataset.data[i] for i in indices]
        temp_dataset.targets = temp_targets

        # Apply noise to the temporary subset
        temp_dataset = add_label_noise(temp_dataset, symmetric_noise_ratio=noise_level, asymmetric_noise_ratio=0)

        # Create final subset with potentially noisy labels
        subset = Subset(temp_dataset, list(range(len(indices)))) # Use indices 0..N-1 for the subset

    else:
        # Test data usually remains clean
        client_dataset = CIFAR10(root="./data", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
        num_samples = len(client_dataset) // num_clients
        indices = list(range(cid * num_samples, (cid + 1) * num_samples))
        subset = Subset(client_dataset, indices)

    return DataLoader(subset, batch_size=batch_size, shuffle=True if train else False)
class RobustLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, num_classes=10):
        """
        Implements the Active-Passive Loss.
        """
        super(RobustLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.epsilon = 1e-6

    def forward(self, outputs, targets):
        # Active Loss: Normalized Cross Entropy
        log_probs = F.log_softmax(outputs, dim=1)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).float().to(outputs.device)
        nce = -1 * (torch.sum(log_probs * one_hot, dim=1)) / (-log_probs.sum(dim=1) + self.epsilon)
        nce = nce.mean()

        # Passive Loss: Normalized Reverse Cross Entropy
        pred = F.softmax(outputs, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(targets, self.num_classes).float().to(outputs.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        rce = rce.mean()

        total_loss = self.alpha * nce + self.beta * rce
        return total_loss



class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, num_clients, noise_level=0.0):
        self.model = Net().to(device)
        self.cid = cid
        self.num_clients = num_clients
        self.noise_level = noise_level
        # Pass noise level when getting train dataloader
        self.trainloader = get_client_dataloader(cid, num_clients, train=True, noise_level=noise_level)
        self.testloader = get_client_dataloader(cid, num_clients, train=False) # Test loader remains clean
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict({k: torch.tensor(v).to(device) for k, v in state_dict.items()})

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        total_loss = 0
        num_batches = 0 # Keep track of batches for accurate averaging
        for epoch in range(1): # Keeping epochs=1 as in original code
            for images, labels in self.trainloader:
                if len(images) == 0: continue # Skip empty batches if dataset size isn't perfectly divisible
                images, labels = images.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return self.get_parameters(config={}), len(self.trainloader.dataset), {
            "loss": avg_loss,
            "noise_level": self.noise_level # Report the assigned noise level
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        with torch.no_grad():
            for images, labels in self.testloader:
                if len(images) == 0: continue
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                num_batches +=1
        accuracy = correct / total if total > 0 else 0
        avg_loss = test_loss / num_batches if num_batches > 0 else 0
        return avg_loss, len(self.testloader.dataset), {
            "accuracy": accuracy,
            "noise_level": self.noise_level # Include noise level for evaluation context if needed
        }

class FedCluster(FedAvg):
    def __init__(self, min_clusters_search=2, max_clusters_search=5, contamination=0.1, anomaly_detector="isolation_forest", *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Define the range for K-Means cluster search
        self.min_clusters_search = min_clusters_search
        self.max_clusters_search = max_clusters_search
        self.contamination = contamination
        self.anomaly_detector = anomaly_detector

        # Initialize anomaly detector based on choice
        if anomaly_detector == "isolation_forest":
            self.detector = IsolationForest(contamination=contamination, random_state=42)
        elif anomaly_detector == "one_class_svm":
            # OneClassSVM might require more tuning, nu corresponds roughly to contamination
            self.detector = OneClassSVM(kernel='rbf', nu=contamination, gamma='auto')
        else:
            raise ValueError(f"Unsupported anomaly detector: {anomaly_detector}")

        self.scaler = StandardScaler()
        self.client_metrics = []
        self.round_metrics = []

    def detect_anomalies(self, scaled_metrics):
        """Detects anomalies using the configured detector."""
        if scaled_metrics.shape[0] == 0: # No clients, no anomalies
             return [], np.array([])
        if self.anomaly_detector == "isolation_forest":
            scores = self.detector.fit_predict(scaled_metrics) # Returns 1 for inliers, -1 for outliers
            normal_indices = np.where(scores == 1)[0]
        elif self.anomaly_detector == "one_class_svm":
            scores = self.detector.fit_predict(scaled_metrics) # Returns 1 for inliers, -1 for outliers
            normal_indices = np.where(scores == 1)[0]
        else:
             normal_indices = np.arange(len(scaled_metrics)) # Default to all normal if detector unknown (shouldn't happen)
             scores = np.ones(len(scaled_metrics))

        return normal_indices.tolist(), scores # Return list of indices and the scores array

    def find_optimal_k(self, data: np.ndarray) -> int:
        """Finds the optimal number of clusters using Silhouette Score."""
        best_k = self.min_clusters_search
        best_score = -1 # Silhouette score ranges from -1 to 1

        # Determine the actual maximum k possible based on data size
        # Need at least k samples for k clusters, and silhouette needs at least 2 clusters.
        max_k = min(self.max_clusters_search, data.shape[0] - 1)

        if data.shape[0] < self.min_clusters_search or max_k < self.min_clusters_search :
             print(f"Warning: Not enough data points ({data.shape[0]}) to search for clusters in range [{self.min_clusters_search}, {max_k}]. Defaulting to k=1 (no clustering).")
             return 1 # Cannot perform clustering

        for k in range(self.min_clusters_search, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # Use n_init>1 for stability
                labels = kmeans.fit_predict(data)

                # Check if clustering resulted in only one cluster (can happen with specific data)
                if len(np.unique(labels)) < 2:
                    #print(f"Warning: KMeans resulted in less than 2 clusters for k={k}. Skipping.")
                    continue # Cannot compute silhouette score

                score = silhouette_score(data, labels)
                #print(f" K={k}, Silhouette Score: {score:.4f}") # Optional: Debugging output

                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception as e:
                print(f"Error during KMeans or Silhouette for k={k}: {e}")
                continue # Skip this k if error occurs

        print(f"Optimal k found: {best_k} with Silhouette Score: {best_score:.4f}")
        return best_k


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        # 1. Extract metrics for anomaly detection and clustering
        metrics = []
        client_cids = [] # Keep track of original client indices
        valid_results = [] # Filter out potential failures reflected in results
        for idx, (client_proxy, fit_res) in enumerate(results):
             # Basic check if fit_res looks valid
             if hasattr(fit_res, 'metrics') and "loss" in fit_res.metrics and "noise_level" in fit_res.metrics:
                 metrics.append([
                     fit_res.metrics["loss"],
                     fit_res.metrics["noise_level"]
                 ])
                 client_cids.append(idx) # Store original index
                 valid_results.append((client_proxy, fit_res)) # Keep the valid result tuple
             else:
                 print(f"Warning: Skipping result from client {client_proxy.cid} due to missing metrics.")

        if not metrics:
             print("Warning: No valid client metrics received for aggregation.")
             return super().aggregate_fit(server_round, valid_results, failures) # Fallback if no metrics


        metrics_array = np.array(metrics)
        # Ensure metrics are finite and not NaN before scaling
        if not np.all(np.isfinite(metrics_array)):
            print("Warning: Non-finite values detected in client metrics. Replacing with 0.")
            metrics_array = np.nan_to_num(metrics_array, nan=0.0, posinf=0.0, neginf=0.0)


        # 2. Scale the metrics
        scaled_metrics = self.scaler.fit_transform(metrics_array)

        # 3. Perform anomaly detection
        normal_client_indices_relative, anomaly_scores = self.detect_anomalies(scaled_metrics) # Indices relative to the 'metrics' list
        num_total_clients = len(metrics)
        num_normal_clients = len(normal_client_indices_relative)
        num_anomalous_clients = num_total_clients - num_normal_clients

        print(f"Round {server_round}: Detected {num_anomalous_clients} anomalous clients out of {num_total_clients}.")

        # Map relative indices back to original indices from 'results'
        normal_client_original_indices = [client_cids[i] for i in normal_client_indices_relative]
        scaled_metrics_normal = scaled_metrics[normal_client_indices_relative] if num_normal_clients > 0 else np.array([])


        # Store anomaly scores mapped to original client indices
        anomaly_scores_mapped = {client_cids[i]: score for i, score in enumerate(anomaly_scores)}

        # --- Clustering Section ---
        optimal_k = 1
        cluster_labels = [-1] * num_total_clients # Default: -1 (not clustered or anomalous)
        cluster_weights = {}

        # Only proceed with clustering if we have enough normal clients for the search range
        if num_normal_clients >= self.min_clusters_search:
            # 4. Find optimal k for normal clients
            optimal_k = self.find_optimal_k(scaled_metrics_normal)

            if optimal_k > 1: # Proceed only if clustering is meaningful (k>1)
                # 5. Cluster the normal clients using the optimal k
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                cluster_labels_normal = kmeans.fit_predict(scaled_metrics_normal)

                # Map cluster labels back to original client indices
                for i, relative_idx in enumerate(normal_client_indices_relative):
                    original_idx = client_cids[relative_idx]
                    cluster_labels[original_idx] = cluster_labels_normal[i] # Assign cluster label

                # Calculate cluster weights (based on number of clients per cluster among normal ones)
                for label in range(optimal_k):
                    # Count clients in this cluster (among normal clients)
                    count_in_cluster = np.sum(cluster_labels_normal == label)
                    if num_normal_clients > 0:
                         cluster_weights[label] = count_in_cluster / num_normal_clients
                    else:
                         cluster_weights[label] = 0
            else:
                print(f"Round {server_round}: Optimal k={optimal_k}, treating all normal clients as one group.")
                # If optimal k=1, treat all normal clients as a single group (label 0)
                for i, relative_idx in enumerate(normal_client_indices_relative):
                    original_idx = client_cids[relative_idx]
                    cluster_labels[original_idx] = 0 # Assign all normal clients to cluster 0
                if num_normal_clients > 0:
                    cluster_weights[0] = 1.0 # Single cluster holds all weight
                else:
                    cluster_weights = {} # No clusters if no normal clients


        # 6. Aggregate parameters with cluster-based weighting (or just average normal clients if k=1)
        parameters_aggregated = None
        aggregate_metrics = {
            "num_clusters_found": optimal_k,
            "num_normal_clients": num_normal_clients,
            "num_anomalous_clients": num_anomalous_clients,
            "cluster_weights": str(cluster_weights), # Convert dict to str for logging
            "anomaly_detector": self.anomaly_detector
        }

        if num_normal_clients > 0:
            weighted_parameters = []
            total_weight_sum = 0 # For normalization if needed

            if optimal_k > 1:
                # Aggregate using cluster weights
                for label in range(optimal_k):
                    # Get original indices of normal clients belonging to this cluster
                    cluster_original_indices = [
                        client_cids[relative_idx]
                        for i, relative_idx in enumerate(normal_client_indices_relative)
                        if cluster_labels_normal[i] == label
                    ]

                    if cluster_original_indices:
                        cluster_weight = cluster_weights.get(label, 0)
                        # Aggregate parameters within the cluster first (simple average)
                        params_to_aggregate = [
                            (parameters_to_ndarrays(results[i][1].parameters), results[i][1].num_examples)
                            for i in cluster_original_indices
                        ]
                        if params_to_aggregate:
                             aggregated_cluster_params = aggregate(params_to_aggregate)
                             # Weight this cluster's aggregated params by the cluster weight
                             weight = sum(p[1] for p in params_to_aggregate) * cluster_weight # Weight by num_examples * cluster_proportion
                             weighted_parameters.append((aggregated_cluster_params, weight))
                             total_weight_sum += weight

            else: # optimal_k is 1, simple average of all normal clients
                 params_to_aggregate = [
                     (parameters_to_ndarrays(results[i][1].parameters), results[i][1].num_examples)
                     for i in normal_client_original_indices
                 ]
                 if params_to_aggregate:
                     # Aggregate all normal clients together
                     aggregated_normal_params = aggregate(params_to_aggregate)
                     weight = sum(p[1] for p in params_to_aggregate) # Total examples from normal clients
                     weighted_parameters.append((aggregated_normal_params, weight))
                     total_weight_sum += weight


            # Final aggregation across weighted clusters (or just the single normal group)
            if weighted_parameters:
                 # Re-normalize weights before final aggregation
                 final_params_to_aggregate = [
                     (params, w / total_weight_sum if total_weight_sum > 0 else 0)
                     for params, w in weighted_parameters
                 ]
                 # Use Flower's aggregate function, now it expects weights summing ~1
                 aggregated_ndarrays = aggregate(final_params_to_aggregate)
                 parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

            else:
                 print(f"Warning: Round {server_round}: No parameters to aggregate after filtering/clustering.")
                 # Fallback? Or return None? Let's fallback to FedAvg on normal clients if aggregation failed.
                 if num_normal_clients > 0:
                     print("Falling back to simple FedAvg on normal clients.")
                     params_to_aggregate = [
                        (parameters_to_ndarrays(results[i][1].parameters), results[i][1].num_examples)
                        for i in normal_client_original_indices
                     ]
                     aggregated_ndarrays = aggregate(params_to_aggregate)
                     parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
                 else: # No normal clients at all
                     return None, aggregate_metrics # Cannot aggregate

        else: # No normal clients detected
            print(f"Warning: Round {server_round}: No normal clients detected. Skipping aggregation.")
            # Decide fallback behavior: return None, or last good model? Returning None for now.
            return None, aggregate_metrics

        # Store metrics for analysis (map results using original indices)
        client_metrics_this_round = []
        for i in range(len(results)): # Iterate through original results list
             fit_res = results[i][1]
             is_normal = i in normal_client_original_indices
             client_metrics_this_round.append({
                 'client_original_index': i,
                 'loss': fit_res.metrics.get('loss', float('nan')),
                 'noise_level': fit_res.metrics.get('noise_level', float('nan')),
                 'anomaly_score': anomaly_scores_mapped.get(i, 1), # Default score 1 (normal) if not in map
                 'is_normal': is_normal,
                 'cluster_label': cluster_labels[i] if is_normal else -1 # Assign cluster label or -1 if anomalous
             })

        self.client_metrics.append({
            'round': server_round,
            'metrics_details': client_metrics_this_round, # Store detailed list
            'optimal_k': optimal_k,
            'anomaly_detector': self.anomaly_detector,
            'num_normal_clients': num_normal_clients,
            'num_anomalous_clients': num_anomalous_clients
        })


        return parameters_aggregated, aggregate_metrics


    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # Store evaluation metrics per client
        metrics = []
        for client_proxy, eval_res in results:
            metrics.append({
                'client_cid': client_proxy.cid, # Store client ID for reference
                'accuracy': eval_res.metrics.get('accuracy', float('nan')),
                'noise_level': eval_res.metrics.get('noise_level', float('nan')), # Get noise level reported by client
                'num_examples': eval_res.num_examples
            })

        self.round_metrics.append({
            'round': server_round,
            'metrics': metrics # Store list of dictionaries
        })

        # Calculate weighted average accuracy across all reporting clients
        accuracies = [res.num_examples * res.metrics["accuracy"] for _, res in results if "accuracy" in res.metrics]
        examples = [res.num_examples for _, res in results if "accuracy" in res.metrics]

        if not examples or sum(examples) == 0:
             print(f"Warning: Round {server_round}: No examples reported for evaluation.")
             return 0.0, {"accuracy": 0.0} # Return 0 if no examples


        aggregated_accuracy = np.sum(accuracies) / np.sum(examples)
        print(f"Round {server_round} Evaluation Aggregate Accuracy: {aggregated_accuracy:.4f}")

        return aggregated_accuracy, {"accuracy": aggregated_accuracy}


def save_metrics(client_metrics, round_metrics, output_dir):
    """Saves detailed client fit metrics and round evaluation metrics."""
    os.makedirs(output_dir, exist_ok=True) # Ensure directory exists

    # Process and Save Client Fit Metrics
    client_fit_data = []
    for round_data in client_metrics:
        round_num = round_data['round']
        optimal_k = round_data['optimal_k']
        detector = round_data['anomaly_detector']
        num_normal = round_data['num_normal_clients']
        num_anomalous = round_data['num_anomalous_clients']
        for client_detail in round_data['metrics_details']:
             client_fit_data.append({
                 'round': round_num,
                 'client_original_index': client_detail['client_original_index'],
                 'loss': client_detail['loss'],
                 'noise_level': client_detail['noise_level'],
                 'anomaly_score': client_detail['anomaly_score'],
                 'is_normal': client_detail['is_normal'],
                 'cluster_label': client_detail['cluster_label'],
                 'optimal_k_round': optimal_k,
                 'anomaly_detector': detector,
                 'num_normal_round': num_normal,
                 'num_anomalous_round': num_anomalous
             })

    if client_fit_data:
        client_df = pd.DataFrame(client_fit_data)
        client_df.to_csv(f"{output_dir}/client_fit_metrics_detailed.csv", index=False)
        print(f"Saved detailed client fit metrics to {output_dir}/client_fit_metrics_detailed.csv")
    else:
        print("No client fit metrics were recorded to save.")


    # Process and Save Round Evaluation Metrics (Accuracy Progression)
    round_eval_data = []
    avg_accuracy_per_round = []
    for round_data in round_metrics:
        round_num = round_data['round']
        total_examples = 0
        weighted_accuracy_sum = 0
        for client_eval in round_data['metrics']:
             round_eval_data.append({
                 'round': round_num,
                 'client_cid': client_eval['client_cid'],
                 'accuracy': client_eval['accuracy'],
                 'noise_level': client_eval['noise_level'],
                 'num_examples': client_eval['num_examples']
             })
             if not np.isnan(client_eval['accuracy']):
                 total_examples += client_eval['num_examples']
                 weighted_accuracy_sum += client_eval['accuracy'] * client_eval['num_examples']

        avg_acc = weighted_accuracy_sum / total_examples if total_examples > 0 else 0
        avg_accuracy_per_round.append({'round': round_num, 'average_accuracy': avg_acc})

    if round_eval_data:
        round_df = pd.DataFrame(round_eval_data)
        round_df.to_csv(f"{output_dir}/round_evaluation_metrics_detailed.csv", index=False)
        print(f"Saved detailed round evaluation metrics to {output_dir}/round_evaluation_metrics_detailed.csv")
    else:
        print("No round evaluation metrics were recorded to save.")

    if avg_accuracy_per_round:
        round_avg_df = pd.DataFrame(avg_accuracy_per_round)
        round_avg_df.to_csv(f"{output_dir}/average_accuracy_progression.csv", index=False)
        print(f"Saved average accuracy progression to {output_dir}/average_accuracy_progression.csv")
        # Print final accuracy
        final_accuracy = round_avg_df['average_accuracy'].iloc[-1]
        print(f"\nFinal Average Accuracy across clients: {final_accuracy:.4f}")
    else:
        print("No average accuracy data to save or report.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning with Client Clustering and Optimal K Search")
    parser.add_argument("--num_rounds", type=int, default=50, help="Number of rounds to simulate")
    parser.add_argument("--num_clients", type=int, default=10, help="Number of clients in the simulation")
    # Replace n_clusters with min/max for search range
    parser.add_argument("--min_clusters_search", type=int, default=2, help="Minimum number of clusters to test for K-Means")
    parser.add_argument("--max_clusters_search", type=int, default=5, help="Maximum number of clusters to test for K-Means")
    parser.add_argument("--contamination", type=float, default=0.1, help="Expected proportion of outliers for anomaly detection")
    parser.add_argument("--anomaly_detector", type=str, default="isolation_forest",
                      choices=["isolation_forest", "one_class_svm"],
                      help="Anomaly detection method to use")
    args = parser.parse_args()

    # Define noise levels for different clients (ensure length matches or exceeds num_clients if using modulo)
    # Example: 3 normal (0.0), 4 slightly noisy (0.1), 3 highly noisy (0.99) -> Matches num_clients=10
    symmetric_noise = [0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.1, 0.99, 0.99, 0.99]
    if args.num_clients > len(symmetric_noise):
         print(f"Warning: num_clients ({args.num_clients}) > len(symmetric_noise) ({len(symmetric_noise)}). Noise levels will repeat.")


    # Create output directory
    import os
    output_dir = f"outputs_optimal_k_{args.anomaly_detector}_contam{args.contamination}" # More descriptive name
    os.makedirs(output_dir, exist_ok=True)

    # Function to compute aggregate metrics (average accuracy) - standard approach
    def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
        """Computes weighted average of metrics."""
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics if "accuracy" in m]
        examples = [num_examples for num_examples, m in metrics if "accuracy" in m]
        if not examples:
            return {"accuracy": 0.0}
        return {"accuracy": sum(accuracies) / sum(examples)}

    # Initialize strategy with search range instead of fixed n_clusters
    strategy = FedCluster(
        min_clusters_search=args.min_clusters_search,
        max_clusters_search=args.max_clusters_search,
        contamination=args.contamination,
        anomaly_detector=args.anomaly_detector,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, # Use standard weighted avg for final eval metric
        # Pass other FedAvg parameters if needed, e.g., fraction_fit, min_fit_clients
        fraction_fit=1.0, # Example: train on all clients
        min_fit_clients=max(2, int(args.num_clients * 0.8)), # Example: require 80% or at least 2 clients
        min_available_clients=args.num_clients, # Wait for all clients
    )

    # Define client function carefully handling noise assignment
    def client_fn(cid: str) -> fl.client.Client:
        client_id = int(cid)
        # Assign noise level based on client ID modulo length of noise list
        noise_idx = client_id % len(symmetric_noise)
        client_noise_level = symmetric_noise[noise_idx]
        print(f"Creating client {cid} with noise level: {client_noise_level:.2f}")
        return FlowerClient(
            cid=client_id,
            num_clients=args.num_clients,
            noise_level=client_noise_level
        ).to_client()


    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        # Ensure client resources are sufficient if running complex ops, otherwise keep minimal
        client_resources={'num_cpus': 0.1, 'num_gpus': 0.1}, # Minimal resources for simulation
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    # Save metrics after simulation
    save_metrics(strategy.client_metrics, strategy.round_metrics, output_dir)

    print("\n--- Simulation History (Losses Centralized) ---")
    print(history.losses_centralized)
    print("\n--- Simulation History (Metrics Centralized - Accuracy) ---")
    print(history.metrics_centralized)
