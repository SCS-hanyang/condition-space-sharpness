
import os
import pandas as pd
import numpy as np

# Correct Use of Absolute Path to be safe
results_dir = "/home/gpuadmin/cssin/init_noise_diffusion_memorization/results/total_analysis_500"
samples = range(24)

mem_stats = []
non_mem_stats = []
n_steps_last = 0

for i in samples:
    path = os.path.join(results_dir, f"sample_{i:03d}", "metrics.csv")
    if not os.path.exists(path):
        continue
    
    try:
        df = pd.read_csv(path)
        # Check actual length
        n_steps = len(df)
        n_steps_last = n_steps
        
        # Memorization
        basin_mem = df["Memorization_Basin"].values
        t_exit_mem = np.argmax(basin_mem)
        
        if t_exit_mem <= 400: 
            len_pre = t_exit_mem
            len_post = n_steps - 1 - t_exit_mem 
            mem_stats.append((len_pre, len_post))
            
        # Non-Memorization
        basin_non = df["Non-memorization_Basin"].values
        t_exit_non = np.argmax(basin_non)
        
        if t_exit_non <= 400:
            len_pre = t_exit_non
            len_post = n_steps - 1 - t_exit_non
            non_mem_stats.append((len_pre, len_post))
            
    except Exception as e:
        print(f"Error {i}: {e}")
        continue

mem_stats = np.array(mem_stats)
non_mem_stats = np.array(non_mem_stats)

print("--- Result Calculation ---")
print(f"Data Length (Steps found in last sample): {n_steps_last}")

if len(mem_stats) > 0:
    avg_pre_mem = np.mean(mem_stats[:, 0])
    avg_post_mem = np.mean(mem_stats[:, 1])
    print(f"[Memorization]")
    print(f"  Avg 0~t_exit Length: {avg_pre_mem:.2f}")
    print(f"  Avg t_exit~End Length: {avg_post_mem:.2f}")
    print(f"  Proposed X-Axis: [-{int(avg_pre_mem)}, +{int(avg_post_mem)}]")

if len(non_mem_stats) > 0:
    avg_pre_non = np.mean(non_mem_stats[:, 0])
    avg_post_non = np.mean(non_mem_stats[:, 1])
    print(f"[Non-memorization]")
    print(f"  Avg 0~t_exit Length: {avg_pre_non:.2f}")
    print(f"  Avg t_exit~End Length: {avg_post_non:.2f}")
    print(f"  Proposed X-Axis: [-{int(avg_pre_non)}, +{int(avg_post_non)}]")
