# Approach 3: Client Clustering and Adaptive Weighting

This approach implements a robust federated learning system that uses client clustering and adaptive weighting to handle noisy and adversarial clients. The implementation focuses on grouping clients with similar characteristics and using anomaly detection to identify and handle outliers.

## Key Features

1. **Client Clustering**
   - Uses K-means clustering to group clients based on their performance metrics
   - Clusters are formed based on loss and noise level characteristics
   - Helps in identifying groups of clients with similar behavior

2. **Anomaly Detection**
   - Supports two anomaly detection methods:
     - Isolation Forest: Better for high-dimensional data, robust to outliers
     - One-Class SVM: Better at finding complex boundaries
   - Identifies potentially adversarial or noisy clients

3. **Adaptive Weighting**
   - Assigns weights to clusters instead of individual clients
   - Reduces the impact of adversarial clients
   - Balances contribution from different client groups

4. **Comprehensive Metrics Tracking**
   - Tracks client-specific metrics
   - Monitors accuracy progression
   - Records noise levels and cluster assignments

## Implementation Details

### Model Architecture
- CNN-based model for CIFAR10 classification
- Two convolutional layers with ReLU activation
- Max pooling and fully connected layers

### Client Setup
- Each client has a unique noise level
- Noise distribution:
  - 3 clients: No noise (0.0)
  - 3 clients: Low noise (0.1)
  - 3 clients: High noise (0.99)

### Clustering Process
1. Scale client metrics (loss and noise level)
2. Perform anomaly detection to identify outliers
3. Cluster remaining clients using K-means
4. Apply cluster-based weighting during aggregation

### Anomaly Detection Methods
1. **Isolation Forest** (Default)
   - Works by isolating observations by randomly selecting a feature and then randomly selecting a split value
   - Faster for high-dimensional data 
   - Less sensitive to parameter choices
   - Better at detecting outliers in sparse datasets
   - Assigns scores of -1 for outliers and 1 for normal data points
   
2. **One-Class SVM**
   - Based on Support Vector Machines
   - Creates a boundary that separates normal from abnormal data
   - Works better for finding complex decision boundaries
   - More computationally intensive
   - Better when normal data is well-structured

The default method is Isolation Forest because:
- It performs well with the metrics we track (loss and noise level)
- It's faster and requires less tuning
- It's more robust to the varying distributions of client metrics

## Usage

### Running the Simulation
```bash
# Using Isolation Forest (default)
python approach_3.py --num_rounds 50 --num_clients 10 --n_clusters 3 --contamination 0.1 --anomaly_detector isolation_forest

# Using One-Class SVM
python approach_3.py --num_rounds 50 --num_clients 10 --n_clusters 3 --contamination 0.1 --anomaly_detector one_class_svm
```

### Parameters
- `--num_rounds`: Number of training rounds (default: 50)
- `--num_clients`: Number of clients in simulation (default: 10)
- `--n_clusters`: Number of clusters for client grouping (default: 3)
- `--contamination`: Expected proportion of outliers (default: 0.1)
- `--anomaly_detector`: Choice of anomaly detection method (default: isolation_forest)
  - Options: `isolation_forest` or `one_class_svm`

### Visualization
```bash
python plot_results.py
```

## Output Files

1. `client_metrics.csv`
   - Round number
   - Client ID
   - Loss
   - Noise level
   - Anomaly score
   - Cluster assignment
   - Anomaly detector used

2. `round_metrics.csv`
   - Round number
   - Client ID
   - Accuracy
   - Noise level

3. `accuracy_progression.csv`
   - Round number
   - Average accuracy

4. `accuracy_and_noise_progression.png`
   - Top plot: Accuracy progression over rounds
   - Bottom plot: Noise levels for each client

## Dependencies
- PyTorch
- Flower (flwr)
- scikit-learn
- pandas
- matplotlib
- seaborn

## Results Analysis
The implementation provides:
- Visualization of learning progression
- Analysis of client behavior
- Effectiveness of outlier detection
- Impact of clustering on model performance

## Advantages
1. Robust to label noise
2. Handles adversarial clients
3. Adapts to heterogeneous client performance
4. Provides detailed metrics for analysis
5. Supports multiple anomaly detection methods