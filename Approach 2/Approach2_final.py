import argparse
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
import random
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

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

class CELoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, num_classes=10):
        super(CELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        return self.ce(outputs, targets)
    
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


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)

def get_client_dataloader(cid, num_clients, batch_size=32, train=True, noise_level=0.0):

    if train:
        client_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
        client_dataset = add_label_noise(client_dataset, symmetric_noise_ratio=noise_level, asymmetric_noise_ratio=0)
    else:
        client_dataset = test_dataset

    num_samples = len(client_dataset) // num_clients
    indices = list(range(cid * num_samples, (cid + 1) * num_samples))
    subset = Subset(client_dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True)


symmetric_noise = [0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.1, 0.99, 0.99, 0.99]

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, num_clients, noise_level=0.0):
        self.model = Net().to(device)
        self.cid = cid
        self.num_clients = num_clients
        self.noise_level = noise_level
        self.trainloader = get_client_dataloader(cid, num_clients, train=True, noise_level=noise_level)
        self.testloader = get_client_dataloader(cid, num_clients, train=False)
        self.criterion = CELoss(num_classes=10)
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
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"loss": avg_loss}

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
        return test_loss / len(self.testloader), len(self.testloader.dataset), {"accuracy": accuracy}
    

# Custom Aggregation Strategy
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": np.sum(accuracies) / np.sum(examples)}

from typing import Union, List, Tuple, Dict, Optional
from flwr.common import Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate

class FedCustomOutlier(FedAvg):
    def __init__(self, delta=1.0, *args, **kwargs):
        """
        delta: constant offset added to the median loss to set the threshold.
        Clients with loss > median_loss + delta are dropped.
        """
        super().__init__(*args, **kwargs)
        self.delta = delta

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        losses = [fit_res.metrics["loss"] for _, fit_res in results]
        median_loss = np.median(losses)
        threshold = median_loss + self.delta
        print(f"[Round {server_round}] Median Loss: {median_loss:.4f}, Threshold: {threshold:.4f}")
        
        filtered_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results if fit_res.metrics["loss"] <= threshold
        ]
        excluded_clients = len(results) - len(filtered_results)
        print(f"[Round {server_round}] Excluded Clients: {excluded_clients}")

        if filtered_results:
            parameters_aggregated = ndarrays_to_parameters(aggregate(filtered_results))
            return parameters_aggregated, {"threshold": threshold}
        else:
            print(f"[Round {server_round}] No client passed the threshold, using all clients.")
            return super().aggregate_fit(server_round, results, failures)


sstrategy = FedCustomOutlier(delta=0.20, evaluate_metrics_aggregation_fn=weighted_average)
basic_strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Simulation CLI")
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=50,
        help="Number of rounds to simulate (default: 50)"
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=10,
        help="Number of clients in the simulation (default: 10)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="basic_strategy",
        choices=["basic_strategy", "sstrategy"],
        help="Strategy to use: 'basic_strategy' for FedAvg or 'sstrategy' for FedCustomOutlier (default: basic_strategy)"
    )
    args = parser.parse_args()

    num_clients = args.num_clients
    num_rounds = args.num_rounds

    strategy = sstrategy if args.strategy == "sstrategy" else basic_strategy


    fl.simulation.start_simulation(
        client_fn=lambda cid: FlowerClient(
            int(cid),
            num_clients,
            noise_level=symmetric_noise[int(cid) % len(symmetric_noise)]
        ),
        num_clients=num_clients,
        client_resources={'num_cpus': 1, 'num_gpus': 1},
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

