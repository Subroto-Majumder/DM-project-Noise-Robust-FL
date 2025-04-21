# 🧠 Federated Learning with Robust Client Selection

This approach implements a **custom federated learning strategy** using [Flower](https://flower.ai/) to improve model performance in the presence of **noisy clients**. The method selectively **excludes clients with high training loss** from the aggregation process using a **median-based thresholding** mechanism.

---

## 🚀 Key Features

- **Custom Federated Strategy (`FedCustomOutlier`)**:  
  Extends `FedAvg` to drop outlier clients whose training loss is significantly higher than others.
  
- **Label Noise Simulation**:  
  Clients are initialized with varying levels of symmetric label noise to simulate heterogeneous data quality.

- **Thresholding Strategy**:
  - Computes median client loss.
  - Sets a threshold:  
    `threshold = median_loss + delta`
  - Clients with loss above this threshold are excluded from aggregation.

- **CIFAR-10 Dataset**:  
  Standard 10-class image classification dataset used for training and evaluation.

---

## 🧪 Experiment Setup

In a **custom test case**, the system simulates:

- **7 clients with low noise**: noise levels of `0.0` or `0.1`
- **3 clients with high noise**: noise levels of `0.99`

```python
symmetric_noise = [0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.1, 0.99, 0.99, 0.99]
```

These noise levels are applied during client creation.

---

## 📊 Observations

- The approach slightly improves accuracy in noisy environments.
- As more sophisticated **client metrics** (e.g., gradient norms, update norms, learning curves) are integrated, the improvement is expected to increase.
- The strategy is **modular** and can be applied **on top of existing aggregation methods**.

---

## 🛠 How to Run (Colab or Local)

1. **Install Dependencies**:
   ```bash
   pip install -U "flwr[simulation]"
   ```

   If using Google Colab, ensure that `ray`, `protobuf`, and `cryptography` versions are compatible if you hit errors.

2. **Run the Simulation**:
   ```bash
   python your_script.py --num_rounds 50 --num_clients 10 --strategy sstrategy
   ```

   You can switch strategies:
   - `basic_strategy`: Standard FedAvg
   - `sstrategy`: Custom FedCustomOutlier

---

## 🔧 Core Components

### `FedCustomOutlier` (Custom Strategy)
```python
threshold = median_loss + delta
```

- Filters out clients with loss > threshold.
- Prevents highly noisy updates from degrading global model.

### `FlowerClient`
- Loads data with client-specific label noise.
- Implements standard training and evaluation routines.


