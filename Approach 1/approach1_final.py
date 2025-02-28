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
import random
from flwr.common import (
    EvaluateRes,
    FitRes,
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


def get_client_dataloader(cid, num_clients, batch_size=32, train=True):
    """Splits dataset for different clients."""
    dataset_to_use = dataset if train else test_dataset
    num_samples = len(dataset_to_use) // num_clients
    indices = list(range(cid * num_samples, (cid + 1) * num_samples))
    subset = Subset(dataset_to_use, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, num_clients, loss_type="robust"):
        self.model = Net().to(device)
        self.cid = cid
        self.num_clients = num_clients
        self.trainloader = get_client_dataloader(cid, num_clients, train=True)
        self.testloader = get_client_dataloader(cid, num_clients, train=False)
        # Select loss function based on loss_type
        if loss_type == "robust":
            self.criterion = RobustLoss(num_classes=10, alpha=1, beta=1)
        else:
            self.criterion = CELoss(num_classes=10, alpha=1, beta=1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict({k: torch.tensor(v).to(device) for k, v in state_dict.items()})

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1): 
            for images, labels in self.trainloader:
                images, labels = images.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

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


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": np.sum(accuracies) / np.sum(examples)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Noise Robust Federated Learning")
    parser.add_argument("--loss", choices=["robust", "ce"], default="robust",
                        help="Select loss function type: robust (Active-Passive) or ce (Cross Entropy)")
    parser.add_argument("--symmetric_noise", type=float, default=0.0,
                        help="Symmetric noise ratio for CIFAR10 labels")
    parser.add_argument("--asymmetric_noise", type=float, default=0.0,
                        help="Asymmetric noise ratio for CIFAR10 labels")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Number of rounds for simulation")
    parser.add_argument("--num_clients", type=int, default=10,
                        help="Number of clients for simulation")
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataset = add_label_noise(dataset,
                              symmetric_noise_ratio=args.symmetric_noise,
                              asymmetric_noise_ratio=args.asymmetric_noise)
    test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)

    # Define strategy 
    strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)


    fl.simulation.start_simulation(
        client_fn=lambda cid: FlowerClient(int(cid), args.num_clients, loss_type=args.loss),
        num_clients=args.num_clients,
        client_resources={'num_cpus': 1, 'num_gpus': 1},
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
