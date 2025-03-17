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
        client_dataset = add_label_noise(client_dataset, symmetric_noise_ratio=noise_level, asymmetric_noise_ratio=0)
    else:
        client_dataset = CIFAR10(root="./data", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))

    num_samples = len(client_dataset) // num_clients
    indices = list(range(cid * num_samples, (cid + 1) * num_samples))
    subset = Subset(client_dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, num_clients, noise_level=0.0):
        self.model = Net().to(device)
        self.cid = cid
        self.num_clients = num_clients
        self.noise_level = noise_level
        self.trainloader = get_client_dataloader(cid, num_clients, train=True, noise_level=noise_level)
        self.testloader = get_client_dataloader(cid, num_clients, train=False)
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
        for epoch in range(1):
            for images, labels in self.trainloader:
                images, labels = images.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / len(self.trainloader)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {
            "loss": avg_loss,
            "noise_level": self.noise_level
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        accuracy = correct / total
        return test_loss / len(self.testloader), len(self.testloader.dataset), {
            "accuracy": accuracy,
            "noise_level": self.noise_level
        }

class FedCluster(FedAvg):
    def __init__(self, n_clusters=3, contamination=0.1, anomaly_detector="isolation_forest", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.anomaly_detector = anomaly_detector
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
        # Initialize anomaly detector based on choice
        if anomaly_detector == "isolation_forest":
            self.detector = IsolationForest(contamination=contamination, random_state=42)
        elif anomaly_detector == "one_class_svm":
            self.detector = OneClassSVM(kernel='rbf', nu=contamination)
        else:
            raise ValueError(f"Unsupported anomaly detector: {anomaly_detector}")
            
        self.scaler = StandardScaler()
        self.client_metrics = []
        self.round_metrics = []

    def detect_anomalies(self, scaled_metrics):
        if self.anomaly_detector == "isolation_forest":
            # Isolation Forest returns -1 for outliers, 1 for normal
            scores = self.detector.fit_predict(scaled_metrics)
            normal_clients = [i for i, score in enumerate(scores) if score == 1]
        else:  # One-Class SVM
            # One-Class SVM returns -1 for outliers, 1 for normal
            scores = self.detector.fit_predict(scaled_metrics)
            normal_clients = [i for i, score in enumerate(scores) if score == 1]
        
        return normal_clients, scores

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Extract metrics for clustering
        metrics = []
        for _, fit_res in results:
            metrics.append([
                fit_res.metrics["loss"],
                fit_res.metrics["noise_level"]
            ])
        
        metrics_array = np.array(metrics)
        
        # Scale the metrics
        scaled_metrics = self.scaler.fit_transform(metrics_array)
        
        # Perform anomaly detection using the selected method
        normal_clients, anomaly_scores = self.detect_anomalies(scaled_metrics)
        
        # Cluster the normal clients
        if len(normal_clients) >= self.n_clusters:
            cluster_labels = self.kmeans.fit_predict(scaled_metrics[normal_clients])
            
            # Calculate cluster weights
            cluster_weights = {}
            for label in range(self.n_clusters):
                cluster_clients = [normal_clients[i] for i in range(len(normal_clients)) 
                                 if cluster_labels[i] == label]
                cluster_weights[label] = len(cluster_clients) / len(normal_clients)
            
            # Aggregate parameters with cluster-based weighting
            weighted_parameters = []
            for label in range(self.n_clusters):
                cluster_clients = [normal_clients[i] for i in range(len(normal_clients)) 
                                 if cluster_labels[i] == label]
                if cluster_clients:
                    cluster_params = [
                        (parameters_to_ndarrays(results[i][1].parameters), 
                         results[i][1].num_examples * cluster_weights[label])
                        for i in cluster_clients
                    ]
                    weighted_parameters.extend(cluster_params)
            
            # Store metrics for analysis
            self.client_metrics.append({
                'round': server_round,
                'metrics': metrics,
                'anomaly_scores': anomaly_scores,
                'cluster_labels': cluster_labels if len(normal_clients) >= self.n_clusters else None,
                'anomaly_detector': self.anomaly_detector
            })
            
            parameters_aggregated = ndarrays_to_parameters(aggregate(weighted_parameters))
            return parameters_aggregated, {
                "num_clusters": self.n_clusters,
                "num_normal_clients": len(normal_clients),
                "cluster_weights": cluster_weights,
                "anomaly_detector": self.anomaly_detector
            }
        
        # Fallback to standard FedAvg if not enough normal clients
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        # Store evaluation metrics
        metrics = []
        for _, eval_res in results:
            metrics.append({
                'accuracy': eval_res.metrics['accuracy'],
                'noise_level': eval_res.metrics['noise_level']
            })
        
        self.round_metrics.append({
            'round': server_round,
            'metrics': metrics
        })
        
        # Calculate weighted average accuracy
        accuracies = [eval_res.num_examples * eval_res.metrics["accuracy"] for _, eval_res in results]
        examples = [eval_res.num_examples for _, eval_res in results]
        
        return np.sum(accuracies) / np.sum(examples), {"accuracy": np.sum(accuracies) / np.sum(examples)}

def save_metrics(client_metrics, round_metrics, output_dir):
    # Save client metrics
    client_data = []
    for m in client_metrics:
        for i in range(len(m['metrics'])):
            client_data.append({
                'round': m['round'],
                'client_id': i,
                'loss': m['metrics'][i][0],
                'noise_level': m['metrics'][i][1],
                'anomaly_score': m['anomaly_scores'][i],
                'cluster': m['cluster_labels'][i] if m['cluster_labels'] is not None and i < len(m['cluster_labels']) else -1,
                'anomaly_detector': m['anomaly_detector']
            })
    
    client_df = pd.DataFrame(client_data)
    client_df.to_csv(f"{output_dir}/client_metrics.csv", index=False)
    
    # Save round metrics with accuracy progression
    round_data = []
    for m in round_metrics:
        for i in range(len(m['metrics'])):
            round_data.append({
                'round': m['round'],
                'client_id': i,
                'accuracy': m['metrics'][i]['accuracy'],
                'noise_level': m['metrics'][i]['noise_level']
            })
    
    round_df = pd.DataFrame(round_data)
    
    # Calculate average accuracy per round
    round_avg = round_df.groupby('round')['accuracy'].mean().reset_index()
    round_avg.to_csv(f"{output_dir}/accuracy_progression.csv", index=False)
    
    # Save detailed round metrics
    round_df.to_csv(f"{output_dir}/round_metrics.csv", index=False)
    
    # Print final accuracy
    final_accuracy = round_avg['accuracy'].iloc[-1]
    print(f"\nFinal Average Accuracy: {final_accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning with Client Clustering")
    parser.add_argument("--num_rounds", type=int, default=50, help="Number of rounds to simulate")
    parser.add_argument("--num_clients", type=int, default=10, help="Number of clients in the simulation")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters for client grouping")
    parser.add_argument("--contamination", type=float, default=0.1, help="Expected proportion of outliers")
    parser.add_argument("--anomaly_detector", type=str, default="isolation_forest",
                      choices=["isolation_forest", "one_class_svm"],
                      help="Anomaly detection method to use")
    args = parser.parse_args()

    # Define noise levels for different clients
    symmetric_noise = [0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.1, 0.99, 0.99, 0.99]

    # Create output directory
    import os
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize strategy
    strategy = FedCluster(
        n_clusters=args.n_clusters,
        contamination=args.contamination,
        anomaly_detector=args.anomaly_detector,
        evaluate_metrics_aggregation_fn=lambda metrics: {"accuracy": np.mean([m["accuracy"] for _, m in metrics])}
    )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=lambda cid: FlowerClient(
            int(cid),
            args.num_clients,
            noise_level=symmetric_noise[int(cid) % len(symmetric_noise)]
        ).to_client(),
        num_clients=args.num_clients,
        client_resources={'num_cpus': 0.1, 'num_gpus': 0},
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    # Save metrics after simulation
    save_metrics(strategy.client_metrics, strategy.round_metrics, output_dir)
