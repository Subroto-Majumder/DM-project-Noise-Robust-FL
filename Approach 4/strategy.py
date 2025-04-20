# --- START OF FILE strategy.py ---

import flwr as fl
from flwr.common import (
    EvaluateRes, FitRes, Metrics, NDArrays, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays
)
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import FitIns
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from model import Net # Need Net for evaluation
from loss import CELoss,RobustLoss # Need loss for evaluation
import torch
from torch.utils.data import DataLoader

# --- Helper Function for Evaluation within Strategy ---
def evaluate_model_on_data(parameters_ndarrays: NDArrays, dataset, batch_size=128):
    """Evaluates a model defined by parameters_ndarrays on the given dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device) # Assuming Net is the model class
    params_dict = zip(model.state_dict().keys(), parameters_ndarrays)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"  [Server Eval Helper] Error loading state dict: {e}")
        return None, {} # Indicate failure

    loss_fn = CELoss().to(device) # Use the same loss as server
    data_loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    test_loss, correct, total, num_batches = 0.0, 0, 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
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
                print(f"  [Server Eval Helper] Error during batch evaluation: {e}")
                continue

    if num_batches == 0 or total == 0:
        print("  [Server Eval Helper] Warning: No batches evaluated or zero total samples.")
        return None, {}

    avg_loss = test_loss / num_batches
    accuracy = correct / total
    return float(avg_loss), {"accuracy": float(accuracy)}


# --- Modified Loss-Based Reputation Strategy (Now uses Server Validation) ---
class LossReputationStrategy(fl.server.strategy.Strategy):
    def __init__(
        self,
        evaluate_fn: Optional[callable],
        server_val_dataset, # New: Pass the clean validation dataset
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
        if server_val_dataset is None:
            raise ValueError("LossReputationStrategy requires a server_val_dataset.")

        self.evaluate_fn = evaluate_fn
        self.server_val_dataset = server_val_dataset # Store the validation dataset
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
        # Ensure model parameters are initialized on the server if not provided
        model = Net()
        ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
        return fl.common.ndarrays_to_parameters(ndarrays)

    def configure_fit(self, server_round, parameters, client_manager):
        config = {}
        fit_ins = FitIns(parameters, config)
        # Sample clients
        available_clients = client_manager.num_available()
        sample_size = int(available_clients * self.fraction_fit)
        num_clients = max(sample_size, self.min_fit_clients)
        # Ensure we don't request more clients than available
        num_clients = min(num_clients, available_clients)
        if num_clients < self.min_fit_clients:
            print(f"Warning: Not enough available clients ({available_clients}) to meet min_fit_clients ({self.min_fit_clients}). Waiting.")
            return [] # Wait for more clients


        clients = client_manager.sample(num_clients=num_clients, min_num_clients=self.min_fit_clients)
        print(f"\n[Round {server_round}] Sampling {len(clients)} clients for training.")
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            print(f"[Round {server_round}] Aggregate fit: No results received.")
            return None, {}
        if failures:
            print(f"[Round {server_round}] Aggregate fit: Received {len(failures)} failures.")

        # --- 1. Evaluate Each Client's Model on Server Validation Set ---
        client_val_losses = {} # cid -> validation_loss
        valid_results_for_clustering = [] # Store (client, fit_res, val_loss) tuples

        print(f"  [Round {server_round}] Evaluating {len(results)} client models on server validation set...")
        for client, fit_res in results:
            if fit_res.status.code != fl.common.Code.OK:
                 print(f"    Skipping client {client.cid} due to FitRes status: {fit_res.status.message}")
                 continue
            try:
                parameters_ndarrays = parameters_to_ndarrays(fit_res.parameters)
                # Use the helper function to evaluate
                val_loss, val_metrics = evaluate_model_on_data(parameters_ndarrays, self.server_val_dataset)

                if val_loss is not None:
                    client_val_losses[client.cid] = val_loss
                    valid_results_for_clustering.append((client, fit_res, val_loss))
                    # Optional: Log accuracy too
                    # print(f"    Client {client.cid}: Server Val Loss={val_loss:.4f}, Acc={val_metrics.get('accuracy', -1):.4f}")
                else:
                    print(f"    Warning: Failed to evaluate model from client {client.cid} on server validation set.")

            except Exception as e:
                print(f"    Error processing result from client {client.cid}: {e}")


        if len(valid_results_for_clustering) < max(2, self.min_fit_clients): # Need at least 2 for clustering, and ideally min_fit
             print(f"  Warning: Not enough valid results ({len(valid_results_for_clustering)}) after server validation for clustering/aggregation. Min required: {max(2, self.min_fit_clients)}. Skipping round aggregation.")
             return None, {} # Can't proceed

        # --- 2. K-Means Clustering on Validation Loss ---
        is_malicious = {} # cid -> bool
        # Use validation losses for clustering
        loss_values = np.array([[client_val_losses[c.cid]] for c, _, _ in valid_results_for_clustering])
        cids_in_order = [c.cid for c, _, _ in valid_results_for_clustering]

        print(f"  Performing K-Means on {len(loss_values)} server validation losses...")
        if len(valid_results_for_clustering) >= 2 and np.std(loss_values) > 1e-6: # Check std dev to avoid clustering identical values
            try:
                # Scaling is still important for K-Means performance
                scaler = StandardScaler()
                scaled_losses = scaler.fit_transform(loss_values)

                kmeans = KMeans(n_clusters=2, random_state=server_round, n_init=10) # Use n_init
                labels = kmeans.fit_predict(scaled_losses)

                # Identify "bad" cluster (heuristic: higher average validation loss)
                mean_losses = []
                for i in range(2): # Assuming n_clusters=2
                    cluster_data = scaled_losses[labels == i]
                    # Calculate mean only if cluster is not empty
                    mean_losses.append(np.mean(cluster_data) if len(cluster_data) > 0 else -np.inf)

                # Check if we got two distinct clusters with different means
                if len(set(labels)) > 1 and mean_losses[0] != mean_losses[1]:
                    bad_cluster_label = np.argmax(mean_losses)
                    print(f"  K-Means: Bad cluster={bad_cluster_label} (Scaled Mean Val Losses: {mean_losses})")
                    for i, cid in enumerate(cids_in_order):
                        is_malicious[cid] = (labels[i] == bad_cluster_label)
                else:
                    # Handle cases: only one cluster found, or means are identical (unlikely with float losses)
                    print("  K-Means: Could not distinguish clusters based on validation loss. Treating all as positive.")
                    for cid in cids_in_order: is_malicious[cid] = False

            except Exception as e:
                print(f"  K-Means on validation loss failed: {e}. Treating all evaluated clients as positive.")
                for cid in cids_in_order: is_malicious[cid] = False
        else:
             # Handle cases: only one result, or all losses are identical
             print(f"  Skipping K-Means (results={len(valid_results_for_clustering)}, std_dev={np.std(loss_values):.4f}). Treating all evaluated clients as positive.")
             for cid in cids_in_order: is_malicious[cid] = False

        # --- 3. Update Reputation ---
        current_reputations = {}
        print("  Updating Reputations based on K-Means(Val Loss):")
        # Use valid_results_for_clustering which contains clients evaluated successfully
        for client, fit_res, val_loss in valid_results_for_clustering:
            cid = client.cid
            # Determine interaction type based on K-Means result
            interaction_type = 'malicious' if is_malicious.get(cid, False) else 'positive'
            self.client_interaction_history[cid].append((server_round, interaction_type))

            # Calculate reputation score (same logic as before)
            i_p_eff, i_m_eff = 0.0, 0.0
            for r, type_hist in self.client_interaction_history[cid]:
                 decay = self.freshness_decay ** (server_round - r)
                 if type_hist == 'positive': i_p_eff += 1.0 * decay
                 else: i_m_eff += 1.0 * decay
            # Add base contribution from initial reputation
            base_contrib = self.initial_reputation * (self.freshness_decay ** server_round)
            i_p_eff += base_contrib

            num = self.reputation_alpha * i_p_eff
            den = num + (1.0 - self.reputation_alpha) * i_m_eff
            rep = num / den if den > 1e-9 else self.initial_reputation # Avoid division by zero
            self.client_reputations[cid] = rep
            current_reputations[cid] = rep
            print(f"    Client {cid}: Val Loss={val_loss:.4f}, Type={interaction_type}, New Rep={rep:.3f}")

        # --- 4. Aggregate Parameters (Weighted by Reputation * num_examples) ---
        total_weight = 0.0
        weighted_param_updates = []
        trusted_clients_count = 0

        # Iterate through the successfully evaluated clients again
        for client, fit_res, _ in valid_results_for_clustering: # val_loss not needed here
            cid = client.cid
            reputation = self.client_reputations.get(cid, self.initial_reputation)

            if reputation >= self.reputation_threshold:
                 # Use num_examples reported by the client for weighting
                 weight = reputation * fit_res.num_examples
                 if weight <= 0:
                      print(f"    Skipping Client {cid} due to zero/negative weight (Rep={reputation:.3f}, Examples={fit_res.num_examples})")
                      continue

                 try:
                    ndarrays = parameters_to_ndarrays(fit_res.parameters)
                    # Basic check for valid format (list of numpy arrays)
                    if isinstance(ndarrays, list) and all(isinstance(arr, np.ndarray) for arr in ndarrays):
                        weighted_param_updates.append((ndarrays, float(weight))) # Ensure weight is float
                        total_weight += float(weight)
                        trusted_clients_count += 1
                    else:
                         print(f"    Warning: Client {cid} parameters were not in expected List[np.ndarray] format after validation. Skipping.")
                 except Exception as e:
                     print(f"    Error converting parameters for client {cid}: {e}. Skipping.")

            else:
                 print(f"    Excluding Client {cid} (Rep: {reputation:.3f} < Threshold: {self.reputation_threshold})")


        aggregated_parameters: Optional[Parameters] = None
        if total_weight > 0 and weighted_param_updates:
            print(f"  Aggregating parameters from {trusted_clients_count}/{len(valid_results_for_clustering)} evaluated clients (Total weight: {total_weight:.2f})")
            # Use FedAvg aggregation logic (same as before)
            num_layers = len(weighted_param_updates[0][0])
            aggregated_ndarrays = []
            for layer_idx in range(num_layers):
                layer_sum = np.zeros_like(weighted_param_updates[0][0][layer_idx], dtype=np.float64) # Use float64 for sum
                for client_params, client_weight in weighted_param_updates:
                    current_layer = client_params[layer_idx]
                    layer_sum += current_layer * client_weight
                layer_average = layer_sum / total_weight
                aggregated_ndarrays.append(layer_average.astype(weighted_param_updates[0][0][layer_idx].dtype)) # Cast back to original dtype

            aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
            print("  Parameter aggregation successful.")

        else:
            print("  No clients met reputation threshold or provided valid updates after validation. Global model not updated this round.")
            # Keep the previous round's parameters if available
            if self.initial_parameters: # Should ideally store the *last aggregated* parameters
                 aggregated_parameters = self.initial_parameters # Fallback, could be improved
            else:
                 aggregated_parameters = None # No update possible

        # --- 5. Aggregate Metrics for Logging ---
        metrics_aggregated = {}
        if valid_results_for_clustering: # Base metrics on clients that were successfully evaluated
             # Average reputation of clients *evaluated* on the server set this round
             avg_rep_evaluated = np.mean([self.client_reputations.get(c.cid, self.initial_reputation) for c, _, _ in valid_results_for_clustering])
             num_malicious_detected = sum(1 for cid in is_malicious if is_malicious.get(cid, False)) # Based on K-Means
             metrics_aggregated["avg_reputation_evaluated"] = float(avg_rep_evaluated)
             metrics_aggregated["num_malicious_detected_val"] = int(num_malicious_detected) # Suffix to clarify method
             metrics_aggregated["num_clients_evaluated"] = len(valid_results_for_clustering)
             metrics_aggregated["num_clients_aggregated"] = int(trusted_clients_count)
             # Average validation loss across evaluated clients
             avg_val_loss = np.mean([val_loss for _, _, val_loss in valid_results_for_clustering])
             metrics_aggregated["avg_server_val_loss_clients"] = float(avg_val_loss)
             # Include average *reported* train loss from clients if available in fit_res.metrics
             train_losses = [fit_res.metrics.get("train_loss", np.nan) for _, fit_res, _ in valid_results_for_clustering if fit_res.metrics]
             train_losses = [l for l in train_losses if not np.isnan(l)] # Filter out NaNs
             if train_losses:
                 metrics_aggregated["avg_reported_train_loss_evaluated"] = float(np.mean(train_losses))


        return aggregated_parameters, metrics_aggregated

    # --- Evaluation Methods (Remain the same) ---
    def configure_evaluate(self, server_round, parameters, client_manager):
        # No client-side evaluation needed in this setup, server handles it
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        # Not receiving evaluate results from clients
        return None, {}

    def evaluate(self, server_round, parameters):
        # This uses the main evaluate_fn passed during init for *global model* evaluation
        if not self.evaluate_fn or parameters is None:
             print(f"[Round {server_round}] Server Eval: Skipping (no evaluate_fn or no parameters).")
             return None

        # Convert parameters if they are NDArrays (can happen if aggregation failed)
        if isinstance(parameters, list):
             ndarrays = parameters
        else:
             ndarrays = parameters_to_ndarrays(parameters)

        eval_res = self.evaluate_fn(server_round, ndarrays, {})
        if eval_res is None:
             print(f"[Round {server_round}] Server Eval: evaluate_fn returned None.")
             return None

        loss, metrics = eval_res
        # Ensure metrics is a dict, even if evaluate_fn returns just loss
        if not isinstance(metrics, dict): metrics = {}

        print(f"[Round {server_round}] Server Eval (Global Model) - Loss: {loss:.4f}, Acc: {metrics.get('accuracy', -1):.4f}")
        # Log metrics from evaluate_fn
        metrics_to_log = {f"server_eval_{k}": float(v) for k, v in metrics.items()}
        metrics_to_log["server_eval_loss"] = float(loss)
        return float(loss), metrics_to_log

# --- END OF FILE strategy.py ---