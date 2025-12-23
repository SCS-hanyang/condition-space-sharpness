import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def main():
    # Setup paths
    base_dir = "results/cond_sharpness_dynamics_in_attraction_basin"
    csv_path = os.path.join(base_dir, "jacobian_analysis.csv")
    output_dir = base_dir # Save in same directory
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    # Load Data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Prepare Data for TE (Text Embedding)
    te_cols = ['J_Prev', 'J_Post1', 'J_Post2']
    df_te = df.melt(id_vars=['Prompt', 'Group'], value_vars=te_cols, var_name='State', value_name='Jacobian_Norm')
    df_te['State'] = df_te['State'].str.replace('J_', '')
    df_te['Metric_Type'] = 'Text Embedding'
    
    # Prepare Data for IE (Input Embedding)
    ie_cols = ['J_Prev_IE', 'J_Post1_IE', 'J_Post2_IE']
    df_ie = df.melt(id_vars=['Prompt', 'Group'], value_vars=ie_cols, var_name='State', value_name='Jacobian_Norm')
    df_ie['State'] = df_ie['State'].str.replace('J_', '').str.replace('_IE', '')
    df_ie['Metric_Type'] = 'Input Embedding'

    # Set Order
    state_order = ['Prev', 'Post1', 'Post2']
    
    # Plotting Function
    def plot_dynamics(data, metric_name, filename):
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # 1. Spaghetti Lines (Individual prompts) - lighter
        # Using units=Prompt is tricky in plain sns without lineplot hue=Group trickery if we want different colors
        # But we can just use hue='Group' with low alpha. 
        # Note: sns.lineplot aggregates by default unless estimator=None and units is set.
        sns.lineplot(
            data=data, 
            x='State', 
            y='Jacobian_Norm', 
            hue='Group', 
            units='Prompt', 
            estimator=None, 
            alpha=0.15, 
            legend=None,
            palette={'Memorized': 'blue', 'Unmemorized': 'orange'}
        )
        
        # 2. Mean Trend (PointPlot) - Bold with error bars
        sns.pointplot(
            data=data, 
            x='State', 
            y='Jacobian_Norm', 
            hue='Group', 
            markers=['o', 's'], 
            capsize=.1, 
            errwidth=1.5,
            palette={'Memorized': 'darkblue', 'Unmemorized': 'darkorange'},
            dodge=0.05
        )
        
        plt.title(f"Jacobian Dynamics ({metric_name})", fontsize=16)
        plt.ylabel(f"Jacobian Norm ({metric_name})", fontsize=14)
        plt.xlabel("Attraction Basin State", fontsize=14)
        plt.legend(title='Group')
        plt.ylim(bottom=0)
        
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close()

    # Generate Plots
    plot_dynamics(df_te, "Text Embedding", "jacobian_dynamics_lineplot_TE.png")
    plot_dynamics(df_ie, "Input Embedding", "jacobian_dynamics_lineplot_IE.png")
    
    # Additional Plot: Log Scale for IE if needed
    plt.figure(figsize=(10, 6))
    df_ie_log = df_ie.copy()
    df_ie_log['Jacobian_Norm'] = np.log1p(df_ie_log['Jacobian_Norm'])
    sns.pointplot(
        data=df_ie_log, 
        x='State', 
        y='Jacobian_Norm', 
        hue='Group', 
        markers=['o', 's'], 
        capsize=.1, 
        palette={'Memorized': 'darkblue', 'Unmemorized': 'darkorange'}
    )
    plt.title("Jacobian Dynamics (Input Embedding) - Log Scale", fontsize=16)
    plt.ylabel("Log(Jacobian Norm + 1)", fontsize=14)
    plt.savefig(os.path.join(output_dir, "jacobian_dynamics_log_IE.png"))
    print(f"Saved log-scale plot to {os.path.join(output_dir, 'jacobian_dynamics_log_IE.png')}")
    plt.close()

if __name__ == "__main__":
    main()
