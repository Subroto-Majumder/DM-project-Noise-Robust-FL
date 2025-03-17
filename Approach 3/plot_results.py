import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the accuracy progression data
accuracy_df = pd.read_csv('outputs/accuracy_progression.csv')

# Read the round metrics for noise levels
round_metrics_df = pd.read_csv('outputs/round_metrics.csv')

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot accuracy progression
ax1.plot(accuracy_df['round'], accuracy_df['accuracy'], 'b-', linewidth=2, label='Average Accuracy')
ax1.set_xlabel('Round')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy Progression Over Rounds')
ax1.grid(True)
ax1.legend()

# Plot noise levels for each client
for client_id in round_metrics_df['client_id'].unique():
    client_data = round_metrics_df[round_metrics_df['client_id'] == client_id]
    ax2.plot(client_data['round'], client_data['noise_level'], 
             label=f'Client {client_id}', alpha=0.7)

ax2.set_xlabel('Round')
ax2.set_ylabel('Noise Level')
ax2.set_title('Noise Levels for Each Client')
ax2.grid(True)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot
plt.savefig('outputs/accuracy_and_noise_progression.png', bbox_inches='tight', dpi=300)
plt.close()

# Print final statistics
print("\nFinal Statistics:")
print(f"Final Average Accuracy: {accuracy_df['accuracy'].iloc[-1]:.4f}")
print("\nNoise Levels per Client:")
print(round_metrics_df.groupby('client_id')['noise_level'].mean()) 