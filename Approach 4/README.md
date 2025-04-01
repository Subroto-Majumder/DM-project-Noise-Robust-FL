# Approach 4: Loss-Based Reputation System for Adversarial FL

This approach implements a robust federated learning system designed to mitigate the impact of simple adversarial clients (specifically label flippers) by using a reputation mechanism based on client training loss. The system identifies clients exhibiting abnormally high training loss, reduces their influence on the global model through a reputation score, and compares its performance against standard Federated Averaging (FedAvg).

## Key Features

1.  **Reputation System**
    *   Assigns a dynamic reputation score to each client based on their historical behavior.
    *   Employs freshness decay to give more weight to recent interactions.
    *   Uses a tunable weighting factor (`alpha`) to balance positive and malicious evidence.

2.  **Loss-Based Anomaly Detection**
    *   Uses K-Means clustering on reported client `train_loss` values in each round.
    *   Assumes 2 clusters: "normal loss" (presumed benign) and "high loss" (presumed malicious).
    *   Identifies the "bad" cluster using the heuristic that label-flipping adversaries often exhibit higher training loss.

3.  **Freshness Decay**
    *   Older interactions (positive or malicious) have less impact on the current reputation score than recent ones, controlled by a decay factor.

4.  **Reputation-Weighted Aggregation**
    *   Filters out clients whose reputation falls below a defined threshold.
    *   Weights the model updates from remaining clients by `reputation * num_examples`, reducing the influence of low-reputation (suspected malicious) clients.

## Implementation Details

### Model Architecture
- Standard CNN model for CIFAR-10 classification (as defined in `Net` class).

### Client Setup
- Simulates a mixed environment of benign and adversarial clients.
- **Benign Clients:** Use `FlowerClient`, training normally on their local data partition.
- **Adversarial Clients:** Use `SimpleAdversarialClient`, which inherits from `FlowerClient` but flips a specified fraction (`flip_fraction`) of labels randomly during its `fit` method's training loop.
- The proportion of adversarial clients is controlled by `malicious_ratio`.

### Reputation Process
1.  **Collect Metrics:** The server receives model updates and a metrics dictionary containing `{"train_loss": ...}` from participating clients after `fit`.
2.  **Cluster Losses:** Performs K-Means clustering (k=2) on the `train_loss` values from clients that returned valid metrics. *Note: Scaling loss values before clustering can optionally be done using StandardScaler.*
3.  **Identify Bad Cluster:** Determines which cluster centroid has the higher average loss; clients in this cluster are tentatively marked as 'malicious' for this round. Handles cases where clustering fails or clusters aren't distinct.
4.  **Update History:** Appends the interaction (round, type='positive'/'malicious') to each client's history.
5.  **Calculate Reputation:** Computes the reputation score $R_i^{(t)}$ for client $i$ at round $t$ using the following formula, incorporating freshness decay ($\gamma$) and weighting factor ($\alpha$):
$$
    \begin{align*}
    I_{p, i}^{(t)} &= \left( \sum_{\substack{(r, \text{'positive'}) \in H_i^{(t)} \\ r \le t}} \gamma^{(t-r)} \right) + R_0 \cdot \gamma^t \\
    I_{m, i}^{(t)} &= \sum_{\substack{(r, \text{'malicious'}) \in H_i^{(t)} \\ r \le t}} \gamma^{(t-r)} \\
    D_i^{(t)} &= \alpha \cdot I_{p, i}^{(t)} + (1 - \alpha) \cdot I_{m, i}^{(t)} \\
    R_i^{(t)} &=
      \begin{cases}
        \frac{\alpha \cdot I_{p, i}^{(t)}}{D_i^{(t)}} & \quad \text{if } D_i^{(t)} > \epsilon \\
        R_0 & \quad \text{if } D_i^{(t)} \le \epsilon
      \end{cases}
    \end{align*}
$$

where $H_i^{(t)}$ is the interaction history, $R_0$ is the initial reputation, and $\epsilon$ is a small value (e.g., 1e-9).

6.  **Weighted Aggregation:** Aggregates model parameters from clients with $R_i^{(t)} \ge \text{reputation threshold}$, weighting each client's contribution by $R_i^{(t)} \times \text{num examples}$.

## Usage

### Running the Simulation (via `run.sh`)
The `run.sh` script facilitates running experiments and collecting results. It executes the main Python script (`approach4_final.py`) and manages output files.
WARNING: running ```run.sh``` will overwrite the previous files, if they had the same training settings.



```bash
# Example: Run Reputation System with 60% malicious clients
./run.sh --folder outputs --strategy reputation --malicious_ratio 0.6 --rounds 50 # Add other python args...

# Example: Run standard FedAvg with 60% malicious clients for comparison
./run.sh --folder outputs --strategy fedavg --malicious_ratio 0.6 --rounds 50 # Add other python args...

```

*Note: It's recommended to run experiments into the same `--folder` and then process results once using the helper scripts (see Visualization).*

### Key Parameters (`run.sh` passes these to Python)
- `--folder`: Directory to store output `.txt` log files.
- `--strategy`: Aggregation strategy (`reputation` or `fedavg`). Default: `reputation`.
- `--rounds`: Number of federated learning rounds.
- `--num_clients`: Total number of clients in the simulation.
- `--malicious_ratio`: Fraction of clients designated as adversarial (0.0 to 1.0).
- `--flip_fraction`: Fraction of labels flipped per batch by adversarial clients (0.0 to 1.0).
- `--fraction_fit`: Fraction of available clients selected for training each round.
- `--min_fit_clients`: Minimum number of clients required for training.
- `--reputation_threshold`: Minimum reputation score required for a client's update to be included in aggregation (only used when `strategy=reputation`).

### Visualization
After running all desired experiments into the *same* output folder:
1.  Navigate into the output folder: `cd <your_output_folder>`
2.  Run the CSV creator: `python3 ../csv_creator.py` (adjust path if needed)
3.  Run the plotter: `python3 ../plotter.py` (adjust path if needed)

This generates `accuracy.csv`, `loss.csv` (containing centralized evaluation results), and `.png` plots comparing strategies under the same conditions (e.g., `accuracy_mal60_comparison.png`).

## Output Files (in specified `--folder`)

1.  **`[strategy]_[condition].txt`**: Raw log output from the Flower simulation for each specific run (e.g., `reputation_mal60.txt`, `fedavg_mal60.txt`).
2.  **`accuracy.csv`**: Centralized accuracy per round for all runs, generated by `csv_creator.py`.
3.  **`loss.csv`**: Centralized loss per round for all runs, generated by `csv_creator.py`.
4.  **`accuracy_[condition]_comparison.png`**: Plots comparing the accuracy of different strategies under the same condition (e.g., same malicious ratio), generated by `plotter.py`.

## Dependencies
- Python 3.x
- PyTorch
- Flower (`flwr`)
- NumPy
- scikit-learn (for KMeans, StandardScaler)
- pandas (for plotter)
- matplotlib (for plotter)

## Results Analysis
The generated outputs allow for:
- Direct comparison of the `reputation` strategy vs. standard `fedavg` under adversarial pressure.
- Visualization of learning progression and stability for each strategy.
- Analysis of how different `malicious_ratio` values impact performance.
- Observation of the reputation system's effectiveness (and potential failure points like temporary dips).
