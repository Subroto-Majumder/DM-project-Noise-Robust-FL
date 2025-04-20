
import torch
from torch.utils.data import DataLoader, Subset

from flwr.common import (
    EvaluateRes, FitRes, Metrics, NDArrays, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays
)

from typing import List, Tuple, Dict, Optional, Union
from loss import CELoss,RobustLoss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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