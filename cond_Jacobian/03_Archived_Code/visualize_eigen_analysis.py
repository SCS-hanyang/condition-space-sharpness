
import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
import os

# Configuration
RESULTS_DIR = "results/cond_Eigen_dynamics_in_attraction_basin"
CSV_PATH = f"{RESULTS_DIR}/eigen_analysis.csv"
OUTPUT_DIR = f"{RESULTS_DIR}/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_list(s):
    try:
        if isinstance(s, str):
            # Check for NaN or empty
            if s == '[]' or s == '':
                return []
            return ast.literal_eval(s)
        return s # Already a list?
    except Exception as e:
        print(f"Error parsing list: {s} - {e}")
        return []



def plot_prompt_analysis(df, prompt, group_name, eigen_col, output_dir, exclude_top1=False):
    # Filter for the specific prompt
    df_prompt = df[df['Prompt'] == prompt].sort_values('Step_Index')
    
    if df_prompt.empty:
        print(f"No data found for prompt: {prompt}")
        return

    # Extract overall dynamics (Diff_Curve is repeated for all rows, take first)
    diff_curve = parse_list(df_prompt.iloc[0]['Diff_Curve'])
    
    # Extract Eigenvalues
    steps = df_prompt['Step_Index'].values
    
    eigenvalues_list = []
    # Check if column exists
    if eigen_col not in df_prompt.columns:
        print(f"Column {eigen_col} not found.")
        return

    for x in df_prompt[eigen_col].values:
        parsed = parse_list(x)
        if len(parsed) == 0:
            parsed = [0]*10 
        eigenvalues_list.append(parsed)
        
    if len(eigenvalues_list) > 0:
        eigenvalues_T = np.array(eigenvalues_list).T
    else:
        eigenvalues_T = None

    drop_index = df_prompt.iloc[0]['Drop_Index']
    
    # Create plot with dual y-axis
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # 1. Plot Diff Curve
    x_all = np.arange(len(diff_curve))
    ax1.plot(x_all, diff_curve, color='navy', label=r'$\|\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset)\|_2$', linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Sampling Time Step ($t$)', fontsize=14)
    ax1.set_ylabel(r'$\|\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset)\|_2$', color='navy', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='navy', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Basin End Line
    ax1.axvline(x=drop_index, color='black', linestyle='--', linewidth=2, label=f'Attraction Basin End ($t={drop_index}$)')
    
    # 2. Plot Eigenvalues on secondary axis
    ax2 = ax1.twinx()
    
    if eigenvalues_T is not None:
        base_cmap = plt.cm.Reds
        # Indices 0 to 9.
        color_intensities = np.linspace(0.9, 0.2, 10)
        
        start_idx = 1 if exclude_top1 else 0
        
        for i in range(start_idx, 10):
            intensity = color_intensities[i]
            color = base_cmap(intensity)
            
            # Label logic
            is_first_entry = (i == start_idx)
            is_last_entry = (i == 9)
            
            label = f'Top-{i+1} $\lambda$' if is_first_entry or is_last_entry else None
            
            # Make the first plotted line thicker and more opaque
            lw = 2.5 if is_first_entry else 1.5
            alpha = 0.9 if is_first_entry else 0.5
            
            ax2.plot(steps, eigenvalues_T[i], color=color, marker='o', markersize=4, 
                     linewidth=lw, alpha=alpha, label=label)
            
        ax2.set_ylabel(r'Eigenvalues $\lambda_k$', color='darkred', fontsize=16)
        ax2.tick_params(axis='y', labelcolor='darkred', labelsize=12)
        
    title_suffix = "Memorization" if group_name == "Memorized" else "Unmemorization"
    if exclude_top1:
        title_suffix += " (Excluding Top-1)"
        
    plt.title(f"Eigenvalue Dynamics for {title_suffix}", fontsize=18, pad=20)
    
    # Custom Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    
    # Create representative lines for legend
    from matplotlib.lines import Line2D
    
    if exclude_top1:
         l_first = Line2D([0], [0], color=base_cmap(color_intensities[1]), linewidth=2.5, marker='o', label=r'Top-2 $\lambda$')
    else:
         l_first = Line2D([0], [0], color=base_cmap(color_intensities[0]), linewidth=2.5, marker='o', label=r'Top-1 $\lambda$')

    l_last = Line2D([0], [0], color=base_cmap(color_intensities[9]), linewidth=1.5, marker='o', alpha=0.5, label=r'Top-10 $\lambda$')
    
    lines2 = [l_first, l_last]
    labels2 = [l.get_label() for l in lines2]
    
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=12)
    
    plt.tight_layout()
    
    # Save
    sanitized_prompt = "".join([c if c.isalnum() else "_" for c in prompt])[:50]
    filename_suffix = "_no_top1" if exclude_top1 else ""
    filename = f"{group_name}_{sanitized_prompt}_eigen_dynamics{filename_suffix}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    if not os.path.exists(CSV_PATH):
        print(f"CSV file not found at {CSV_PATH}")
        return

    print(f"Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Setup directories
    output_dir_te = f"{RESULTS_DIR}/plots"
    output_dir_ie = f"{RESULTS_DIR}/plots2"
    os.makedirs(output_dir_te, exist_ok=True)
    os.makedirs(output_dir_ie, exist_ok=True)
    
    mem_prompts = df[df['Group'] == 'Memorized']['Prompt'].unique()
    unmem_prompts = df[df['Group'] == 'Unmemorized']['Prompt'].unique()
    
    print(f"Processing. TE -> {output_dir_te}, IE -> {output_dir_ie}")
    
    for prompts, group in [(mem_prompts, "Memorized"), (unmem_prompts, "Unmemorized")]:
        print(f"Processing {len(prompts)} {group} prompts...")
        for p in prompts:
            # TE Plots
            plot_prompt_analysis(df, p, group, 'Top10_Eigenvalues_TE', output_dir_te, exclude_top1=False)
            plot_prompt_analysis(df, p, group, 'Top10_Eigenvalues_TE', output_dir_te, exclude_top1=True)
            
            # IE Plots
            plot_prompt_analysis(df, p, group, 'Top10_Eigenvalues_IE', output_dir_ie, exclude_top1=False)
            plot_prompt_analysis(df, p, group, 'Top10_Eigenvalues_IE', output_dir_ie, exclude_top1=True)
    
    print("Done.")

if __name__ == "__main__":
    main()
