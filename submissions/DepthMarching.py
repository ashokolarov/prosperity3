import numpy as np
import matplotlib.pyplot as plt

# Container data (sorted by multiplier for easier verification)
containers = [
    {"multiplier": 90, "inhabitants": 10}, {"multiplier": 89, "inhabitants": 8},
    {"multiplier": 80, "inhabitants": 6}, {"multiplier": 73, "inhabitants": 4},
    {"multiplier": 50, "inhabitants": 4}, {"multiplier": 37, "inhabitants": 3},
    {"multiplier": 31, "inhabitants": 2}, {"multiplier": 20, "inhabitants": 2},
    {"multiplier": 17, "inhabitants": 1}, {"multiplier": 10, "inhabitants": 1}
]

# Constants
BASE_TREASURE = 10000
PENALTY_SECOND = 50000
TOTAL_TEAMS = 10000
INITIAL_DEPTH_TEAMS = TOTAL_TEAMS

def format_profit(profit):
    """Format profit as string with k for thousands"""
    if profit >= 1000:
        return f"{profit/1000:.1f}k"
    return f"{profit:.0f}"

def calculate_profit(container, total_picks, total_teams, is_second=False):
    """Calculate profit for a container considering ALL previous picks"""
    X = 100 * total_picks / (2 * total_teams)  # Percentage of total possible picks
    base_profit = (BASE_TREASURE * container["multiplier"]) / (container["inhabitants"] + X)
    return base_profit - (PENALTY_SECOND if is_second else 0)

def get_best_pair(containers, total_picks, total_teams):
    """Find the optimal container pair considering current distribution of picks"""
    # First container selection
    first_profits = [calculate_profit(c, total_picks[i], total_teams, False) 
                    for i, c in enumerate(containers)]
    best_first = np.argmax(first_profits)
    
    # Second container selection
    second_profits = []
    for i, c in enumerate(containers):
        if i == best_first:
            second_profits.append(-np.inf)
        else:
            profit = calculate_profit(c, total_picks[i], total_teams, True)
            second_profits.append(profit if profit > 0 else -np.inf)
    
    best_second = np.argmax(second_profits)
    if second_profits[best_second] > 0:
        return (best_first, best_second), (first_profits[best_first], second_profits[best_second])
    else:
        return (best_first, None), (first_profits[best_first], 0)

def simulate_strategy(containers, max_depth, linspace_start=1, linspace_end=0):
    """Main simulation with recursive depth-based strategy and linear P decay"""
    # Exponential probability decay that favors early depths
    prob_dist = np.exp(np.linspace(np.log(linspace_start), np.log(linspace_end+1e-10), max_depth))
    
    # Renormalize to ensure start/end exactly match requested values
    if max_depth > 1:
        prob_dist = (prob_dist - prob_dist[-1]) / (prob_dist[0] - prob_dist[-1]) * (linspace_start - linspace_end) + linspace_end
    
    # Tracking variables
    total_picks = np.zeros(len(containers))
    strategy_history = []
    team_allocations = {}
    profit_history = []
    
    # Initialize depth-based team tracking
    depth_teams = {0: INITIAL_DEPTH_TEAMS}
    
    for depth in range(max_depth):
        if not depth_teams.get(depth, 0):
            break  # No teams left at this depth
        
        # Get current probability for this depth
        p = prob_dist[depth]
        
        # Get best pair for current knowledge state
        pair, profits = get_best_pair(containers, total_picks, TOTAL_TEAMS)
        c1, c2 = pair
        profit_per_team = sum(profits)
        
        # Calculate teams sticking vs moving
        teams_at_depth = depth_teams[depth]
        teams_sticking = int(teams_at_depth * (1 - p))
        teams_moving = teams_at_depth - teams_sticking
        
        # Last depth forces all teams to stick
        if depth == max_depth - 1:
            teams_sticking = teams_at_depth
            teams_moving = 0

        # ALL TEAMS MAKE THEIR PICKS (both sticking and moving)
        if c2 is not None:
            total_picks[c1] += teams_at_depth
            total_picks[c2] += teams_at_depth
            pair_str = f"{containers[c1]['multiplier']}-{containers[c2]['multiplier']}"
        else:
            total_picks[c1] += teams_at_depth
            pair_str = f"{containers[c1]['multiplier']} (only)"

        # Track allocations and profits
        strategy_history.append(pair_str)
        profit_history.append(profit_per_team)
        
        # Prepare for next depth if not last
        if teams_moving > 0 and depth < max_depth - 1:
            depth_teams[depth+1] = depth_teams.get(depth+1, 0) + teams_moving
        
        # Record allocation details
        team_allocations[depth] = {
            "depth": depth,
            "p_value": p,
            "pair": pair_str,
            "teams_processed": teams_at_depth,
            "teams_sticking": teams_sticking,
            "teams_moving": teams_moving,
            "profit": profit_per_team,
            "total_picks": total_picks.copy()
        }
    
    return strategy_history, profit_history, team_allocations

def visualize_results(history, profits, allocations):
    """Enhanced visualization with all requested changes"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Profit graph (top)
    ax1.plot(profits, 'r*-', markersize=10, linewidth=2)
    ax1.set_title("Profit per Depth Level")
    ax1.set_ylabel("Profit")
    ax1.grid(True, alpha=0.3)
    
    # Add profit annotations
    for idx, profit in enumerate(profits):
        ax1.annotate(format_profit(profit),
                    xy=(idx, profit),  # Added xy parameter
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    
    # Strategy progression (bottom)
    # First convert pairs to numerical y-values for positioning
    unique_pairs = list(dict.fromkeys(history))  # Maintain order
    y_values = [unique_pairs.index(pair) for pair in history]
    ax2.plot(y_values, 'bo-', markersize=10, linewidth=2)
    
    # Set y-ticks to show container pairs
    ax2.set_yticks(range(len(unique_pairs)))
    ax2.set_yticklabels(unique_pairs)
    ax2.set_title("Optimal Container Pairs by Thinking Depth")
    ax2.set_ylabel("Container Pair")
    ax2.grid(True, alpha=0.3)
    
    # Add team allocation annotations at correct heights with P value as percentage
    for idx, depth in enumerate(allocations):
        alloc = allocations[depth]
        pair_str = alloc['pair']
        y_pos = unique_pairs.index(pair_str)
        
        ax2.annotate(f"P: {alloc['p_value']*100:.0f}%\nStick: {alloc['teams_sticking']}\nMove: {alloc['teams_moving']}",
                    xy=(idx, y_pos),  # Added xy parameter
                    xytext=(10, 0),
                    textcoords='offset points',
                    ha='left',
                    va='center',
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="w", alpha=0.7))
    
    # Common x-axis settings
    for ax in [ax1, ax2]:
        ax.set_xlabel("Depth Level")
        ax.set_xticks(range(len(history)))
        ax.set_xticklabels([str(i+1) for i in range(len(history))])
    
    plt.tight_layout()
    plt.show()

# Run simulation
max_depth = 10
P_start = 1
P_end = 0
strategy_hist, profit_hist, alloc_data = simulate_strategy(containers, max_depth, P_start, P_end)

# Show detailed allocation table
print("Depth | P     | Teams  | Sticking | Moving | Profit    | Container Pair")
print("----------------------------------------------------------------------")
for depth in alloc_data:
    alloc = alloc_data[depth]
    print(f"{alloc['depth']+1:4} | {alloc['p_value']:.2f} | {alloc['teams_processed']:6} | {alloc['teams_sticking']:8} | {alloc['teams_moving']:6} | "
          f"${format_profit(alloc['profit']):>7} | {alloc['pair']}")

visualize_results(strategy_hist, profit_hist, alloc_data)