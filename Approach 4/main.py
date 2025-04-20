# --- START OF FILE main.py ---

import argparse
import flwr as fl
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import random
from torch.utils.data import Subset # Needed for server val set

from model import Net
from loss import CELoss
# Make sure to import the correct strategy name if you changed it
from strategy import LossReputationStrategy # Or ServerValidationReputationStrategy if renamed
from client import FlowerClient, SimpleAdversarialClient
from server import get_server_evaluate_fn
from data import get_server_validation_subset # Import the new data function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10


# --- Main Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server Validation Reputation FL") # Updated description
    parser.add_argument("--rounds", type=int, default=40, help="Number of rounds")
    parser.add_argument("--num_clients", type=int, default=10, help="Total clients")
    parser.add_argument("--malicious_ratio", type=float, default=0.60, help="Fraction of malicious clients")
    parser.add_argument("--flip_fraction", type=float, default=0.9, help="Label flip rate for adversaries")
    parser.add_argument("--fraction_fit", type=float, default=0.8, help="Fraction of clients per round")
    # Ensure min_fit_clients <= num_clients * fraction_fit realistically
    parser.add_argument("--min_fit_clients", type=int, default=5, help="Min clients for training")
    # Reputation threshold might need tuning based on validation loss range
    parser.add_argument("--reputation_threshold", type=float, default=0.2, help="Min reputation for aggregation")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["reputation", "fedavg"], # Keep choices simple, 'reputation' now means server validation
        default="reputation",
        help="Aggregation strategy ('reputation' for server validation, 'fedavg')"
    )
    parser.add_argument("--server_val_set_size", type=int, default=500, help="Size of server validation subset")


    args = parser.parse_args()

    # Load clean test data for server evaluation AND server validation subset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)

    # --- Create the Server Validation Subset ---
    server_val_subset = get_server_validation_subset(test_dataset, subset_size=args.server_val_set_size)
    print(f"Created server validation subset with {len(server_val_subset)} clean samples.")

    # Create server evaluation function (for the *final* global model)
    server_evaluate_fn = get_server_evaluate_fn(Net, test_dataset) # Pass the Net class

    # --- Select Strategy based on Argument ---
    selected_strategy = None
    print(f"\n*** Using Strategy: {args.strategy.upper()} ***")

    if args.strategy == "reputation":
        selected_strategy = LossReputationStrategy( # Using the modified strategy
            evaluate_fn=server_evaluate_fn,
            server_val_dataset=server_val_subset, # Pass the validation subset
            fraction_fit=args.fraction_fit,
            min_fit_clients=args.min_fit_clients,
            # Ensure min_available >= min_fit
            min_available_clients=max(args.min_fit_clients, args.num_clients // 2),
            reputation_threshold=args.reputation_threshold,
            # Optionally pass initial_parameters if needed
            # initial_parameters=fl.common.ndarrays_to_parameters(initial_model_params),
        )
    elif args.strategy == "fedavg":
        from flwr.server.strategy import FedAvg
        selected_strategy = FedAvg(
            evaluate_fn=server_evaluate_fn,
            fraction_fit=args.fraction_fit,
            min_fit_clients=args.min_fit_clients,
            min_available_clients=max(args.min_fit_clients, args.num_clients // 2),
            # Optionally pass initial_parameters if needed
            # initial_parameters=fl.common.ndarrays_to_parameters(initial_model_params),
        )
    else:
        raise ValueError(f"Unknown strategy specified: {args.strategy}")


    # Identify malicious clients
    num_malicious = int(args.num_clients * args.malicious_ratio)
    # Ensure malicious clients don't exceed total clients
    num_malicious = min(num_malicious, args.num_clients)
    malicious_client_ids = random.sample(range(args.num_clients), num_malicious)
    print(f"Malicious client IDs ({num_malicious}): {malicious_client_ids}")

    # Client function
    def client_fn(cid: str) -> fl.client.Client:
        cid_int = int(cid)
        # Determine if the client is malicious BEFORE creating the object
        is_malicious = cid_int in malicious_client_ids

        if is_malicious:
            # print(f"Creating malicious client {cid_int}") # Debug print
            client_obj = SimpleAdversarialClient(
                cid=cid_int,
                num_clients=args.num_clients,
                flip_fraction=args.flip_fraction
            )
        else:
            # print(f"Creating benign client {cid_int}") # Debug print
            client_obj = FlowerClient(
                cid=cid_int,
                num_clients=args.num_clients
            )
        # Convert the FlowerClient/SimpleAdversarialClient to a flwr.client.Client
        return client_obj.to_client()

    print(f"\nStarting Simulation:")
    print(f"  Rounds: {args.rounds}, Total Clients: {args.num_clients}")
    print(f"  Malicious Ratio: {args.malicious_ratio*100:.1f}% ({num_malicious} clients, Flip Fraction: {args.flip_fraction*100:.1f}%)")
    print(f"  Fraction Fit: {args.fraction_fit}, Min Fit Clients: {args.min_fit_clients}")
    if args.strategy == "reputation":
        print(f"  Reputation Strategy: Server Validation (Set Size: {args.server_val_set_size}, Threshold: {args.reputation_threshold})")
    else:
        print(f"  Reputation Strategy: FedAvg (Baseline)")
    print("")

    # Set client resources (adjust GPU if needed)
    client_resources = {'num_cpus': 1, 'num_gpus': 0.0} # Start with CPU
    if device == torch.device("cuda"):
        client_resources = {'num_cpus': 1, 'num_gpus': 0.1} # Small fraction for testing
        print("Attempting to use GPU resources for clients.")


    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=selected_strategy,
        # Optional: Ray init arguments if running distributed simulation
        # ray_init_args={"include_dashboard": False}
    )

    print("\nSimulation Finished.")
    # You can add code here to process/plot the history object
    # print(history.metrics_centralized) # Print centralized metrics (accuracy, loss)
    # print(history.metrics_distributed) # Print distributed metrics (from aggregate_fit)

# --- END OF FILE main.py ---