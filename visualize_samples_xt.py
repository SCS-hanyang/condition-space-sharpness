
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np
import os

# Load Data
RESULTS_DIR = "init_noise_diffusion_memorization/results/cond_sharpness_xt_dynamics_in_attraction_basin"
CSV_PATH = f"{RESULTS_DIR}/jacobian_xt_analysis.csv"

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"File not found: {CSV_PATH}")
    exit(1)

# Ensure output directory exists
os.makedirs(f"{RESULTS_DIR}/sample_plots", exist_ok=True)

# Select first 10 unique prompts for Memorandum and Unmemorized each
try:
    unique_prompts_mem = df[df['Group'] == 'Memorized']['Prompt'].unique()[:10]
    unique_prompts_unmem = df[df['Group'] == 'Unmemorized']['Prompt'].unique()[:10]
    unique_prompts = np.concatenate([unique_prompts_mem, unique_prompts_unmem])
except Exception as e:
    print(f"Error selecting prompts: {e}")
    unique_prompts = df['Prompt'].unique()[:20]

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
    color = 'tab:blue'
    ax1.plot(timesteps, diff_curve, color=color, linestyle='-', alpha=0.6, label='Attraction Basin Dynamics')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Highlight Basin Exit
    ax1.axvline(x=drop_index, color='gray', linestyle='--', label=f'Basin Exit (t={drop_index})')

    # Create Twin Axis for Jacobian
    ax2 = ax1.twinx()
    
    # Plot 2: J_xt Norms (Right Axis)
    steps = prompt_data["Step_Index"].values
    j_xt = prompt_data["J_Norm_xt"].values
    
    color_xt = 'tab:purple'
    
    # Plot Xt Jacobian
    ax2.plot(steps, j_xt, 's-', color=color_xt, label='J_xt (Latent)', markersize=6, alpha=0.8)
    
    ax2.tick_params(axis='y', labelcolor=color_xt)
    
    # Title & Labels with LaTeX
    group_label = "Memorized" if is_memorized else "Unmemorized"
    
    # Main Title
    plt.suptitle(f"Dynamics of Memorization (Spatial Sharpness) - {group_label} {i+1}", fontsize=14, y=0.98)
    
    # Subtitle / Legend Explanation in Math
    # J_xt definition: Gradient w.r.t latent image x_t
    ax1.set_title(r"Blue: $\| \epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset) \|_2$  |  Purple: $\| \nabla_{x_t} \epsilon_\theta(x_t, c) \|_F$  |  Dashed: Basin Exit", 
                  fontsize=10, pad=10)
    
    ax1.set_xlabel('Diffusion Timestep $t$ (0=Start, 50=End)', fontsize=12)
    ax1.set_ylabel(r'$\| \epsilon_c - \epsilon_{unc} \|_2$', color=color, fontsize=12)
    ax2.set_ylabel(r'$\| J_{x_t} \|_F$', color=color_xt, fontsize=12)

    # Legends
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], color=color, lw=2, label=r'$\| \epsilon_c - \epsilon_{unc} \|_2$'),
        Line2D([0], [0], color=color_xt, marker='s', markersize=6, label=r'$\| J_{x_t} \|_F$'),
        Line2D([0], [0], color='gray', linestyle='--', label='Basin Exit')
    ]
    
    ax1.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90) 
    # Use distinct filename
    plt.savefig(f"{RESULTS_DIR}/sample_plots/sample_{i+1}_{group_label}_xt.png", dpi=300)
    plt.close()

print("All sample plots generated in:", f"{RESULTS_DIR}/sample_plots")
