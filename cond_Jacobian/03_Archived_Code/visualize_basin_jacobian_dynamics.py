
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def visualize_jacobian_dynamics():
    # Configuration
    RESULTS_DIR = "results/cond_sharpness_dynamics_in_attraction_basin"
    CSV_PATH = f"{RESULTS_DIR}/jacobian_analysis.csv"
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    # Load Data
    df = pd.read_csv(CSV_PATH)
    
    # Prepare data for plotting
    # We want to compare Memorized vs Unmemorized
    # And compare the transition: Inside -> Post1 -> Post2
    
    plot_data_te = [] # Text Encoder (CLIP Output)
    plot_data_ie = [] # Input Embeddings
    
    for _, row in df.iterrows():
        group = row["Group"]
        
        # CLIP Output (TE)
        plot_data_te.append({"Group": group, "State": "Inside Basin", "Step": 0, "Jacobian Norm": row["J_Prev"]})
        plot_data_te.append({"Group": group, "State": "After Basin 1", "Step": 1, "Jacobian Norm": row["J_Post1"]})
        plot_data_te.append({"Group": group, "State": "After Basin 2", "Step": 2, "Jacobian Norm": row["J_Post2"]})
        
        # Input Embedding (IE)
        plot_data_ie.append({"Group": group, "State": "Inside Basin", "Step": 0, "Jacobian Norm": row["J_Prev_IE"]})
        plot_data_ie.append({"Group": group, "State": "After Basin 1", "Step": 1, "Jacobian Norm": row["J_Post1_IE"]})
        plot_data_ie.append({"Group": group, "State": "After Basin 2", "Step": 2, "Jacobian Norm": row["J_Post2_IE"]})
        
    df_te = pd.DataFrame(plot_data_te)
    df_ie = pd.DataFrame(plot_data_ie)
    
    # Create Visualization
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: CLIP Output Jacobian Dynamics
    sns.pointplot(
        data=df_te, 
        x="State", 
        y="Jacobian Norm", 
        hue="Group", 
        ax=axes[0], 
        capsize=.1, 
        errorbar="ci", # Show confidence interval
        dodge=True,
        markers=["o", "s"],
        linestyles=["-", "--"]
    )
    axes[0].set_title("CLIP Output Jacobian Dynamics (TE)")
    axes[0].set_ylabel("Jacobian Frobenius Norm")
    axes[0].set_xlabel("State relative to Attraction Basin Exit")
    
    # Add strip plot for detailed distribution visibility (optional, checking density)
    # sns.stripplot(data=df_te, x="State", y="Jacobian Norm", hue="Group", ax=axes[0], alpha=0.3, dodge=True, legend=False)

    # Plot 2: Input Embedding Jacobian Dynamics
    sns.pointplot(
        data=df_ie, 
        x="State", 
        y="Jacobian Norm", 
        hue="Group", 
        ax=axes[1], 
        capsize=.1,
        errorbar="ci",
        dodge=True,
        markers=["o", "s"],
        linestyles=["-", "--"]
    )
    axes[1].set_title("Input Embedding Jacobian Dynamics (IE)")
    axes[1].set_ylabel("Jacobian Frobenius Norm")
    axes[1].set_xlabel("State relative to Attraction Basin Exit")
    
    plt.tight_layout()
    plt.suptitle("Jacobian Dynamics: Inside vs After Attraction Basin", y=1.05, fontsize=16)
    
    SAVE_PATH = f"{RESULTS_DIR}/jacobian_dynamics_comparison_lineplot.png"
    plt.savefig(SAVE_PATH, bbox_inches='tight', dpi=300)
    print(f"Saved visualization to {SAVE_PATH}")
    
    # Also save a Log Scale version for IE if the range is huge
    fig_log, axes_log = plt.subplots(1, 2, figsize=(16, 6))
    
    # TE Log
    sns.pointplot(data=df_te, x="State", y="Jacobian Norm", hue="Group", ax=axes_log[0], capsize=.1, errorbar="ci", dodge=True)
    axes_log[0].set_yscale("log")
    axes_log[0].set_title("CLIP Output Jacobian (Log Scale)")
    
    # IE Log
    sns.pointplot(data=df_ie, x="State", y="Jacobian Norm", hue="Group", ax=axes_log[1], capsize=.1, errorbar="ci", dodge=True)
    axes_log[1].set_yscale("log")
    axes_log[1].set_title("Input Embedding Jacobian (Log Scale)")
    
    plt.tight_layout()
    plt.suptitle("Jacobian Dynamics (Log Scale)", y=1.05, fontsize=16)
    
    SAVE_PATH_LOG = f"{RESULTS_DIR}/jacobian_dynamics_comparison_log.png"
    plt.savefig(SAVE_PATH_LOG, bbox_inches='tight', dpi=300)
    print(f"Saved log comparison to {SAVE_PATH_LOG}")

if __name__ == "__main__":
    visualize_jacobian_dynamics()
