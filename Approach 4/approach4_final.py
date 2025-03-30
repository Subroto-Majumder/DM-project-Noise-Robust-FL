import argparse
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
import random
from flwr.common import (
    EvaluateRes, FitRes, Metrics, NDArrays, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate as flwr_aggregate
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import FitIns
from collections import defaultdict
import time

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler # Still useful for scaling loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10


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

class CELoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, num_classes=10):
        super(CELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        return self.ce(outputs, targets.long())

def get_client_dataloader(cid, num_clients, batch_size=32, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    full_dataset = CIFAR10(root="./data", train=train, download=True, transform=transform)
    num_samples = len(full_dataset) // num_clients
    indices = list(range(cid * num_samples, (cid + 1) * num_samples))
    subset = Subset(full_dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=train)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, num_clients):
        self.model = Net().to(device)
        self.cid = cid
        self.num_clients = num_clients
        self.train_loader = get_client_dataloader(cid, num_clients, train=True)
        self.test_loader = get_client_dataloader(cid, num_clients, train=False)
        self.loss_fn = CELoss(num_classes=NUM_CLASSES)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.num_examples = len(self.train_loader.dataset) 

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        epoch_loss = 0
        steps = 0
        for epoch in range(1): 
            for images, labels in self.train_loader:
                images, labels = images.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels) 
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                steps += 1

        avg_loss = epoch_loss / steps if steps > 0 else -1.0
        metrics = {"train_loss": float(avg_loss)}
        return self.get_parameters(config={}), self.num_examples, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        test_loss, correct, total, num_batches = 0, 0, 0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                num_batches += 1
        avg_loss = test_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        num_test_examples = len(self.test_loader.dataset)
        return float(avg_loss), num_test_examples, {"accuracy": float(accuracy)}

# --- Adversarial Client (Simple Label Flipper) ---
class SimpleAdversarialClient(FlowerClient):
    def __init__(self, cid, num_clients, flip_fraction=0.9):
        super().__init__(cid, num_clients)
        self.flip_fraction = flip_fraction 
        print(f"[Adv Client {self.cid}] Initialized (Label Flipper: {self.flip_fraction*100}%)")

    def _flip_labels(self, labels_batch):
        """Flips a fraction of labels in a batch randomly."""
        labels_np = labels_batch.cpu().numpy()
        flipped_labels = labels_np.copy()
        batch_size = len(labels_np)
        num_to_flip = int(self.flip_fraction * batch_size)

        if num_to_flip == 0:
            return labels_batch

        flip_indices = np.random.choice(batch_size, num_to_flip, replace=False)
        for i in flip_indices:
            original_label = labels_np[i]
            other_labels = [l for l in range(NUM_CLASSES) if l != original_label]
            if other_labels: 
                flipped_labels[i] = np.random.choice(other_labels)

        return torch.tensor(flipped_labels, dtype=torch.long).to(labels_batch.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        epoch_loss = 0
        steps = 0
        for epoch in range(1): 
            for images, labels in self.train_loader:
                images = images.to(device)
                flipped_labels = self._flip_labels(labels).to(device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, flipped_labels) 
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                steps += 1

        avg_loss = epoch_loss / steps if steps > 0 else -1.0 
        metrics = {"train_loss": float(avg_loss)}
        return self.get_parameters(config={}), self.num_examples, metrics
    

# --- Loss-Based Reputation Strategy ---
class LossReputationStrategy(fl.server.strategy.Strategy):
    def __init__(
        self,
        evaluate_fn: Optional[callable], 
        fraction_fit: float = 0.5,
        min_fit_clients: int = 3,      
        min_available_clients: int = 5,
        initial_parameters: Optional[Parameters] = None,
        reputation_alpha: float = 0.5,   
        freshness_decay: float = 0.9,   
        reputation_threshold: float = 0.2, 
        initial_reputation: float = 0.5,
    ):
        super().__init__()
        if evaluate_fn is None:
             raise ValueError("LossReputationStrategy requires an evaluate_fn.")
        self.evaluate_fn = evaluate_fn
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters

        # Reputation state
        self.reputation_alpha = reputation_alpha
        self.freshness_decay = freshness_decay
        self.reputation_threshold = reputation_threshold
        self.initial_reputation = initial_reputation

        self.client_reputations: Dict[str, float] = defaultdict(lambda: self.initial_reputation)
        self.client_interaction_history: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    def initialize_parameters(self, client_manager):
        if self.initial_parameters: return self.initial_parameters
        return ndarrays_to_parameters([val.cpu().numpy() for _, val in Net().state_dict().items()])

    def configure_fit(self, server_round, parameters, client_manager):
        config = {}
        fit_ins = FitIns(parameters, config)
        sample_size = int(client_manager.num_available() * self.fraction_fit)
        num_clients = max(sample_size, self.min_fit_clients)
        clients = client_manager.sample(num_clients=num_clients, min_num_clients=self.min_fit_clients)
        print(f"\n[Round {server_round}] Sampling {len(clients)} clients for training.")
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        if not results: return None, {}

        # --- 1. Extract Loss Metric ---
        client_losses = {} # cid -> train_loss
        valid_results = []
        for client, fit_res in results:
            metrics = fit_res.metrics
            if metrics and "train_loss" in metrics and isinstance(metrics["train_loss"], (int, float)) and metrics["train_loss"] >= 0:
                 client_losses[client.cid] = metrics["train_loss"]
                 valid_results.append((client, fit_res))
            else:
                 print(f"Warning: Invalid/missing loss from client {client.cid}. Skipping.")

        if len(valid_results) < self.min_fit_clients:
             print("Warning: Not enough valid results for reputation update/aggregation.")
             return None, {} 

        # --- 2. K-Means Clustering on Loss ---
        is_malicious = {} # cid -> bool
        loss_values = np.array([[client_losses[c.cid]] for c, _ in valid_results]) # Reshape for K-Means
        cids_in_order = [c.cid for c, _ in valid_results]


        if len(valid_results) >= 2 and np.std(loss_values) > 1e-6:
            try:
                scaler = StandardScaler()
                scaled_losses = scaler.fit_transform(loss_values)

                kmeans = KMeans(n_clusters=2, random_state=server_round, n_init=10)
                labels = kmeans.fit_predict(scaled_losses)
                # Identify "bad" cluster (heuristic: higher average loss)
                # Handle cases where clusters might be empty or means are identical
                mean_losses = []
                for i in range(2):
                    cluster_data = scaled_losses[labels == i]
                    mean_losses.append(np.mean(cluster_data) if len(cluster_data) > 0 else -np.inf)

                if len(set(labels)) > 1 and mean_losses[0] != mean_losses[1]:
                    bad_cluster_label = np.argmax(mean_losses)
                    print(f"  K-Means: Bad cluster={bad_cluster_label} (Scaled Mean Losses: {mean_losses})")
                    for i, cid in enumerate(cids_in_order):
                        is_malicious[cid] = (labels[i] == bad_cluster_label)
                else:
                    print("  K-Means: Could not distinguish clusters based on loss. Treating all as positive.")
                    for cid in cids_in_order: is_malicious[cid] = False 

            except Exception as e:
                print(f"  K-Means failed: {e}. Treating clients as positive.")
                for cid in cids_in_order: is_malicious[cid] = False 
        else:
            print(f"  Skipping K-Means (results={len(valid_results)}, std_dev={np.std(loss_values):.4f}). Treating clients as positive.")
            for cid in cids_in_order: is_malicious[cid] = False 


        # --- 3. Update Reputation ---
        current_reputations = {}
        print("  Updating Reputations:")
        for client, fit_res in valid_results:
            cid = client.cid
            # Determine interaction type based on K-Means result for this client
            interaction_type = 'malicious' if is_malicious.get(cid, False) else 'positive'
            self.client_interaction_history[cid].append((server_round, interaction_type))

            # Calculate reputation score
            i_p_eff, i_m_eff = 0.0, 0.0
            for r, type in self.client_interaction_history[cid]:
                 decay = self.freshness_decay ** (server_round - r)
                 if type == 'positive': i_p_eff += 1.0 * decay
                 else: i_m_eff += 1.0 * decay
            # Add base contribution from initial reputation
            base_contrib = self.initial_reputation * (self.freshness_decay ** server_round)
            i_p_eff += base_contrib

            num = self.reputation_alpha * i_p_eff
            den = num + (1.0 - self.reputation_alpha) * i_m_eff
            rep = num / den if den > 1e-9 else self.initial_reputation
            self.client_reputations[cid] = rep
            current_reputations[cid] = rep
            print(f"    Client {cid}: Loss={client_losses[cid]:.4f}, Type={interaction_type}, Rep={rep:.3f}")

        # --- 4. Aggregate Parameters (Weighted by Reputation * num_examples) ---
        total_weight = 0.0
        weighted_param_updates = [] 
        trusted_clients_count = 0

        for client, fit_res in valid_results:
            cid = client.cid
            reputation = self.client_reputations.get(cid, self.initial_reputation)

            if reputation >= self.reputation_threshold:
                 weight = reputation * fit_res.num_examples
                 if weight <= 0: continue 
                 # Skip clients with zero or negative weight

                 ndarrays = parameters_to_ndarrays(fit_res.parameters)
                 if isinstance(ndarrays, list) and all(isinstance(arr, np.ndarray) for arr in ndarrays):
                     weighted_param_updates.append((ndarrays, float(weight))) # Ensure weight is float
                     total_weight += float(weight)
                     trusted_clients_count += 1
                 else:
                     print(f"Warning: Client {cid} parameters were not in expected List[np.ndarray] format. Skipping.")
            else:
                 print(f"    Excluding Client {cid} (Rep: {reputation:.3f} < Threshold: {self.reputation_threshold})")


        aggregated_parameters: Optional[Parameters] = None
        if total_weight > 0 and weighted_param_updates:
            print(f"  Aggregating {trusted_clients_count}/{len(valid_results)} clients (Total weight: {total_weight:.2f})")
            num_layers = len(weighted_param_updates[0][0])
            aggregated_ndarrays = []

            for layer_idx in range(num_layers):
                # Accumulate the weighted sum for this specific layer across all clients
                layer_sum = np.zeros_like(weighted_param_updates[0][0][layer_idx])

                for client_params, client_weight in weighted_param_updates:
                    # Access the specific layer (numpy array) and multiply by weight
                    current_layer = client_params[layer_idx]
                    layer_sum += current_layer * client_weight 

                # Calculate the weighted average for this layer
                layer_average = layer_sum / total_weight
                aggregated_ndarrays.append(layer_average)

            aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        else:
            print("  No clients met reputation threshold or provided valid updates. Model not updated.")

        # --- 5. Aggregate Metrics for Logging ---
        metrics_aggregated = {}
        if valid_results:
             avg_rep_all = np.mean([self.client_reputations.get(c.cid, self.initial_reputation) for c, _ in valid_results])
             num_malicious_detected = sum(1 for cid in is_malicious if is_malicious.get(cid, False))
             metrics_aggregated["avg_reputation_all_valid"] = float(avg_rep_all)
             metrics_aggregated["num_malicious_detected"] = int(num_malicious_detected)
             metrics_aggregated["num_clients_aggregated"] = int(trusted_clients_count)
             if client_losses:
                  avg_loss = np.mean([client_losses[c.cid] for c, _ in valid_results if c.cid in client_losses])
                  metrics_aggregated["avg_train_loss_all_valid"] = float(avg_loss)

        return aggregated_parameters, metrics_aggregated

    # --- Evaluation Methods  ---
    def configure_evaluate(self, server_round, parameters, client_manager): return []
    def aggregate_evaluate(self, server_round, results, failures): return None, {}
    def evaluate(self, server_round, parameters):
        if not self.evaluate_fn: return None
        eval_res = self.evaluate_fn(server_round, parameters_to_ndarrays(parameters), {})
        if eval_res is None: return None
        loss, metrics = eval_res
        print(f"[Round {server_round}] Server Eval - Loss: {loss:.4f}, Acc: {metrics.get('accuracy', -1):.4f}")
        return float(loss), {k: float(v) for k, v in metrics.items()} 


# --- Server-Side Evaluation Function ---
def get_server_evaluate_fn(model_class, test_dataset):
    """Returns a function for server-side evaluation."""
    test_loader = DataLoader(test_dataset, batch_size=128)

    def evaluate(server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]):
        model = model_class().to(device) 
        params_dict = zip(model.state_dict().keys(), parameters_ndarrays)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"Error loading state dict during server evaluation: {e}")
            return None 

        loss_fn = CELoss().to(device) # Instantiate loss
        model.eval()
        test_loss, correct, total, num_batches = 0.0, 0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                try:
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    num_batches += 1
                except Exception as e:
                    print(f"Error during batch evaluation: {e}")
                    continue 

        if num_batches == 0: return None 

        avg_loss = test_loss / num_batches
        accuracy = correct / total if total > 0 else 0.0
        return float(avg_loss), {"accuracy": float(accuracy)}

    return evaluate


# --- Main Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Loss-Based Reputation FL")
    parser.add_argument("--rounds", type=int, default=20, help="Number of rounds")
    parser.add_argument("--num_clients", type=int, default=10, help="Total clients")
    parser.add_argument("--malicious_ratio", type=float, default=0.6, help="Fraction of malicious clients")
    parser.add_argument("--flip_fraction", type=float, default=0.9, help="Label flip rate for adversaries")
    parser.add_argument("--fraction_fit", type=float, default=0.5, help="Fraction of clients per round")
    parser.add_argument("--min_fit_clients", type=int, default=5, help="Min clients for training")
    parser.add_argument("--reputation_threshold", type=float, default=0.2, help="Min reputation for aggregation (used only if strategy is 'reputation')")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["reputation", "fedavg"], 
        default="reputation",            
        help="Aggregation strategy to use ('reputation' or 'fedavg')"
    )


    args = parser.parse_args()

    # Load clean test data for server 
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)

    # Create server evaluation function
    server_evaluate_fn = get_server_evaluate_fn(Net, test_dataset) # Pass the Net class

    # --- Select Strategy based on Argument ---
    selected_strategy = None
    print(f"\n*** Using Strategy: {args.strategy.upper()} ***") 

    if args.strategy == "reputation":
        selected_strategy = LossReputationStrategy(
            evaluate_fn=server_evaluate_fn,
            fraction_fit=args.fraction_fit,
            min_fit_clients=args.min_fit_clients,
            min_available_clients=args.num_clients // 2, 
            reputation_threshold=args.reputation_threshold,

        )
    elif args.strategy == "fedavg":
        from flwr.server.strategy import FedAvg
        selected_strategy = FedAvg(
            evaluate_fn=server_evaluate_fn,
            fraction_fit=args.fraction_fit,
            min_fit_clients=args.min_fit_clients,
            min_available_clients=args.num_clients // 2, 
        )
    else:
        raise ValueError(f"Unknown strategy specified: {args.strategy}")


    # Identify malicious clients
    num_malicious = int(args.num_clients * args.malicious_ratio)
    malicious_client_ids = random.sample(range(args.num_clients), num_malicious)
    print(f"Malicious client IDs ({num_malicious}): {malicious_client_ids}")

    # Client function 
    def client_fn(cid: str) -> fl.client.Client:
        cid_int = int(cid)
        if cid_int in malicious_client_ids:
            client_obj = SimpleAdversarialClient(
                cid=cid_int,
                num_clients=args.num_clients,
                flip_fraction=args.flip_fraction
            )
        else:
            client_obj = FlowerClient(
                cid=cid_int,
                num_clients=args.num_clients
            )
        return client_obj.to_client()

    print(f"\nStarting Simulation:")
    print(f"  Rounds: {args.rounds}, Clients: {args.num_clients}")
    print(f"  Malicious Ratio: {args.malicious_ratio*100:.1f}% (Flip Fraction: {args.flip_fraction*100:.1f}%)")
    print(f"  Fraction Fit: {args.fraction_fit}, Min Fit Clients: {args.min_fit_clients}")
    if args.strategy == "reputation": 
        print(f"  Reputation Threshold: {args.reputation_threshold}")
    print("") 

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        client_resources={'num_cpus': 1, 'num_gpus': 0.0} if device == torch.device("cpu") else {'num_cpus': 1, 'num_gpus': 0.1},
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=selected_strategy,
    )

    print("\nSimulation Finished.")
