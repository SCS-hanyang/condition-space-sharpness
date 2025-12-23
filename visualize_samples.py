
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np

# Load Data
RESULTS_DIR = "init_noise_diffusion_memorization/results/cond_sharpness_dynamics_in_attraction_basin"
CSV_PATH = f"{RESULTS_DIR}/jacobian_analysis.csv"

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"File not found: {CSV_PATH}")
    exit(1)

# Ensure output directory exists
import os
os.makedirs(f"{RESULTS_DIR}/sample_plots", exist_ok=True)

# Select first 10 unique prompts
# The CSV contains multiple rows per prompt (one for each step).
# Select first 10 unique prompts for Memorandum and Unmemorized each
unique_prompts_mem = df[df['Group'] == 'Memorized']['Prompt'].unique()[:10]
unique_prompts_unmem = df[df['Group'] == 'Unmemorized']['Prompt'].unique()[:10]
unique_prompts = np.concatenate([unique_prompts_mem, unique_prompts_unmem])

print(f"Generating plots for {len(unique_prompts)} prompts...")

for i, prompt in enumerate(unique_prompts):
    # Filter data for this prompt
    prompt_data = df[df["Prompt"] == prompt].sort_values("Step_Index")
    
    if prompt_data.empty:
        continue
        
    # Get Metadata
    is_memorized = prompt_data["Group"].iloc[0] == "Memorized"
    drop_index = prompt_data["Drop_Index"].iloc[0]
    
    # Get Curve Data (Stored as string representation of list in 'Diff_Curve')
    # All rows for same prompt should have same curve, pick first
    curve_str = prompt_data["Diff_Curve"].iloc[0]
    try:
        diff_curve = ast.literal_eval(curve_str)
    except:
        print(f"Error parsing curve for prompt {i}")
        continue
        
    timesteps = list(range(len(diff_curve)))
    
    # Create Dual-Axis Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot 1: Attraction Basin Dynamics (Left Axis)
    # L2 Norm Diff (Cond - Uncond)
    color = 'tab:blue'
    ax1.set_xlabel('Timestep (t)')
    ax1.set_ylabel('||Cond - Uncond|| (Blue)', color=color)
    ax1.plot(timesteps, diff_curve, color=color, linestyle='-', alpha=0.6, label='Attraction Basin Dynamics')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Highlight Basin Exit
    ax1.axvline(x=drop_index, color='gray', linestyle='--', label=f'Basin Exit (t={drop_index})')

    # Create Twin Axis for Jacobian
    ax2 = ax1.twinx()
    
    # Plot 2: Jacobian Norms (Right Axis)
    # We have data only for specific steps (stored in prompt_data)
    
    steps = prompt_data["Step_Index"].values
    j_te = prompt_data["J_Norm_TE"].values
    j_ie = prompt_data["J_Norm_IE"].values
    
    color_te = 'tab:red'
    color_ie = 'tab:green'
    
    ax2.set_ylabel('Jacobian Norm (Dots)', color='black')
    
    # Plot IE Jacobian
    # Use j_ie data
    ax2.plot(steps, j_ie, '^-', color=color_ie, label='J_IE (Input Embed)', markersize=6, alpha=0.8)
    
    ax2.tick_params(axis='y', labelcolor=color_ie)
    
    # Title & Labels with LaTeX
    group_label = "Memorized" if is_memorized else "Unmemorized"
    
    # Main Title
    plt.suptitle(f"Dynamics of Memorization (Input Embedding Sensitivity) - {group_label} {i+1}", fontsize=14, y=0.98)
    
    # Subtitle / Legend Explanation in Math
    # J_IE definition: Gradient w.r.t input word embeddings x_emb
    ax1.set_title(r"Blue: $\| \epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset) \|_2$  |  Green: $\| \nabla_{e_{in}} \epsilon_\theta(x_t, c) \|_F$  |  Dashed: Basin Exit", 
                  fontsize=10, pad=10)
    
    ax1.set_xlabel('Diffusion Timestep $t$ (0=Start, 50=End)', fontsize=12)
    ax1.set_ylabel(r'$\| \epsilon_c - \epsilon_{unc} \|_2$', color=color, fontsize=12)
    ax2.set_ylabel(r'$\| J_{IE} \|_F$', color=color_ie, fontsize=12)

    # Legends
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], color=color, lw=2, label=r'$\| \epsilon_c - \epsilon_{unc} \|_2$'),
        Line2D([0], [0], color=color_ie, marker='^', markersize=6, label=r'$\| J_{IE} \|_F$'),
        Line2D([0], [0], color='gray', linestyle='--', label='Basin Exit')
    ]
    
    ax1.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90) 
    # Use a different filename for IE plots
    plt.savefig(f"{RESULTS_DIR}/sample_plots/sample_{i+1}_{group_label}_IE.png", dpi=300)
    plt.close()

print("All sample plots generated in:", f"{RESULTS_DIR}/sample_plots")
