import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# Define containers
containers = [
    {"multiplier": 90, "inhabitants": 10}, {"multiplier": 89, "inhabitants": 8},
    {"multiplier": 80, "inhabitants": 6}, {"multiplier": 73, "inhabitants": 4},
    {"multiplier": 50, "inhabitants": 4}, {"multiplier": 37, "inhabitants": 3},
    {"multiplier": 31, "inhabitants": 2}, {"multiplier": 20, "inhabitants": 2},
    {"multiplier": 17, "inhabitants": 1}, {"multiplier": 10, "inhabitants": 1}
]

# Parameters
RANDOM_LEVELS = [0.1, 0.2, 0.3,0.4, 0.5]
CONVERGENCE_ITERATIONS_LEVELS = [5,10,15, 25]
NUM_TRIALS = 500
TOTAL_X = 100
container_indices = range(len(containers))
container_pairs = list(combinations(container_indices, 2))

# Initialization methods
def init_uniform():
    return np.ones(len(containers)) * (TOTAL_X/len(containers))

def init_exponential():
    weights = np.random.exponential(scale=1.0, size=len(containers))
    return (weights / weights.sum()) * TOTAL_X

def init_dirichlet():
    return np.random.dirichlet(np.ones(len(containers))) * TOTAL_X

def init_powerlaw():
    weights = np.random.power(a=0.5, size=len(containers))
    return (weights / weights.sum()) * TOTAL_X

init_methods = [init_uniform, init_exponential, init_dirichlet, init_powerlaw]

# Calculate grid size
n_random = len(RANDOM_LEVELS)
n_convergence = len(CONVERGENCE_ITERATIONS_LEVELS)
n_rows = n_random
n_cols = n_convergence

plt.figure(figsize=(n_cols * 6, n_rows * 4))  # Dynamic figure size

for random_idx, RANDOM in enumerate(RANDOM_LEVELS):
    for conv_idx, CONVERGENCE_ITERATIONS in enumerate(CONVERGENCE_ITERATIONS_LEVELS):
        print(f"\nProcessing RANDOM={RANDOM}, ITERATIONS={CONVERGENCE_ITERATIONS}")
        
        pair_wins = {pair: 0 for pair in container_pairs}
        pair_profits_all = {pair: [] for pair in container_pairs}  # All trials
        pair_profits_win = {pair: [] for pair in container_pairs}  # Only winning trials
        
        for trial in range(NUM_TRIALS):
            if (trial+1) % 200 == 0:  # Print progress every 200 trials
                print(f"  Trial {trial+1}/{NUM_TRIALS} completed")
                
            # Random initialization for this trial
            method_idx = np.random.randint(0, 4)
            current_X_weights = init_methods[method_idx]()
            # current_X_weights = init_methods[0]()
            # Track container wins across iterations
            container_win_history = np.zeros(len(containers))
            
            for iteration in range(CONVERGENCE_ITERATIONS):
                # Inner simulation to determine winners
                container_wins = np.zeros(len(containers))
                
                for _ in range(100):  # Simulations per iteration
                    # Blend current weights with randomness
                    perturbed_weights = (1 - RANDOM) * current_X_weights + RANDOM * (TOTAL_X/len(containers)) * np.ones(len(containers))
                    perturbed_weights = np.maximum(perturbed_weights, 0)
                    perturbed_weights = (perturbed_weights / perturbed_weights.sum()) * TOTAL_X
                    
                    # Generate X values
                    X = np.random.multinomial(TOTAL_X, perturbed_weights/perturbed_weights.sum())
                    
                    # Calculate profits
                    profits = [
                        (10000 * c["multiplier"]) / (c["inhabitants"] + X[i]) 
                        for i, c in enumerate(containers)
                    ]
                    
                    # Find winning pair
                    max_pair = max(container_pairs, key=lambda pair: profits[pair[0]] + profits[pair[1]] - 50000)
                    container_wins[max_pair[0]] += 1
                    container_wins[max_pair[1]] += 1
                
                # Update container win history (cumulative across iterations)
                container_win_history += container_wins
                
                # Update X weights for next iteration based on win history

                win_ratios = container_win_history / container_win_history.sum()
                current_X_weights = (win_ratios / win_ratios.sum()) * TOTAL_X
            
            # After all iterations, record results for ALL pairs
            X = np.random.multinomial(TOTAL_X, current_X_weights/current_X_weights.sum())
            profits = [
                (10000 * c["multiplier"]) / (c["inhabitants"] + X[i]) 
                for i, c in enumerate(containers)
            ]
            
            # Record profits for ALL pairs in this trial
            for pair in container_pairs:
                pair_profit = profits[pair[0]] + profits[pair[1]] - 50000
                pair_profits_all[pair].append(pair_profit)
            
            # Record winning pair and its profit
            max_pair = max(container_pairs, key=lambda pair: profits[pair[0]] + profits[pair[1]] - 50000)
            pair_wins[max_pair] += 1
            pair_profits_win[max_pair].append(profits[max_pair[0]] + profits[max_pair[1]] - 50000)
        
        # Plotting - keep wins as bar height but show both profit metrics
        ax = plt.subplot(n_rows, n_cols, random_idx * n_cols + conv_idx + 1)
        
        # Sort pairs by win count (as before)
        sorted_pairs = sorted(pair_wins.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Prepare data for plotting
        pair_labels = []
        win_counts = []
        avg_profits_all = []
        avg_profits_win = []
        for pair, wins in sorted_pairs:
            c1, c2 = containers[pair[0]], containers[pair[1]]
            pair_labels.append(f"{c1['multiplier']} & {c2['multiplier']}")
            win_counts.append(wins)
            avg_profits_all.append(np.mean(pair_profits_all[pair]))
            avg_profits_win.append(np.mean(pair_profits_win[pair]) if pair_profits_win[pair] else 0)
        
        # Plot with win counts as bar height
        bars = ax.bar(range(len(sorted_pairs)), win_counts, 
                     color=plt.cm.tab10(random_idx), edgecolor='black')
        
        ax.set_xticks(range(len(sorted_pairs)))
        ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=10)
        ax.set_xlabel("Container Pair", fontsize=10)
        ax.set_ylabel("Wins", fontsize=10)
        ax.set_title(
            f"Randomness: {RANDOM}\nConvergence Iters: {CONVERGENCE_ITERATIONS}",
            fontsize=11
        )
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels with all metrics
        for bar, wins, avg_all, avg_win in zip(bars, win_counts, avg_profits_all, avg_profits_win):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'Avg All: {avg_all:.0f}',
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout(pad=3.0)
plt.show()