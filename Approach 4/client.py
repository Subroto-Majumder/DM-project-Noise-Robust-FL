import flwr as fl
from data import get_client_dataloader
from model import Net
from loss import CELoss, RobustLoss
import torch
import torch.optim as optim
import numpy as np

NUM_CLASSES=10


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, num_clients):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net().to(self.device)
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
        self.model.to(self.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        epoch_loss = 0
        steps = 0
        for epoch in range(1): 
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
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
                images, labels = images.to(self.device), labels.to(self.device)
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
                images = images.to(self.device)
                flipped_labels = self._flip_labels(labels).to(self.device)

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
    