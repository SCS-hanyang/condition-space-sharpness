import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import time
from diffusers import DDIMScheduler, StableDiffusionPipeline

import argparse

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
BASE_DIR = "/home/gpuadmin/cssin/init_noise_diffusion_memorization"
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
# OUTPUT_DIR will be defined in main loop

sys.path.append(BASE_DIR)

# Try importing local pipeline if available, otherwise standard
try:
    from local_sd_pipeline import LocalStableDiffusionPipeline
except ImportError:
    print("LocalStableDiffusionPipeline not found, using standard.")
    LocalStableDiffusionPipeline = StableDiffusionPipeline

def compute_jacobian_norm_batched(unet, latents, t, prompt_embeds, num_projections=1):
    """
    Batched Hutchinson estimator for ||J||_F.
    Returns a tensor of shape (batch_size,) containing the Jacobian norm for each sample.
    """
    prompt_embeds = prompt_embeds.detach()
    latents = latents.detach()
    
    batch_size = latents.shape[0]
    
    with torch.enable_grad():
        prompt_embeds.requires_grad_(True)
        # Forward pass
        # Note: We pass t (scalar or tensor) to the model.
        noise_pred = unet(latents, t, encoder_hidden_states=prompt_embeds).sample
        
        sq_norm_sum = torch.zeros(batch_size, device=latents.device)
        
        for k in range(num_projections):
            # Random probe v
            v = torch.randn_like(noise_pred)
            
            v_dot_eps_sum = torch.sum(noise_pred * v)
            
            retain = (k < num_projections - 1)
            grads = torch.autograd.grad(v_dot_eps_sum, prompt_embeds, retain_graph=retain, create_graph=False)[0]
            
            # Squared norm per sample
            # Grads shape: [B, SeqLen, Dim] -> flatten to [B, -1]
            grads_sq_flat = grads.view(batch_size, -1).pow(2)
            sample_sq_norms = grads_sq_flat.sum(dim=1) # [B]
            
            sq_norm_sum += sample_sq_norms.detach()
            
        est_frob_sq = sq_norm_sum / max(1, float(num_projections))
        return est_frob_sq.sqrt() # [B]

def load_prompts(num_prompts=500):
    #mem_path = os.path.join(PROMPTS_DIR, "sd1_mem.txt")
    mem_path = os.path.join(PROMPTS_DIR, "mem.txt")
    unmem_path = os.path.join(PROMPTS_DIR, "sd1_nmem.txt")
    
    try:
        with open(mem_path, 'r') as f:
            mem_prompts = [line.strip() for line in f.readlines() if line.strip()]
            
        with open(unmem_path, 'r') as f:
            unmem_prompts = [line.strip() for line in f.readlines() if line.strip()]
        
        # Ensure we don't exceed available prompts
        mem_prompts = mem_prompts[:min(len(mem_prompts), num_prompts)]
        unmem_prompts = unmem_prompts[:min(len(unmem_prompts), num_prompts)]
        
        print(f"Loaded {len(mem_prompts)} memorized prompts and {len(unmem_prompts)} unmemorized prompts.")
        return mem_prompts, unmem_prompts
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return ["cat"]*5, ["dog"]*5

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_projections", type=int, default=1, help="Hutchinson estimator projections")
    parser.add_argument("--init_noise_path", type=str, default=None, help="Path to specific init_noise.pt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--timestep_indices", type=int, nargs='+', default=[80, 60, 40, 20, 0], help="List of timestep indices to analyze")
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Configuration
    model_id = "CompVis/stable-diffusion-v1-4"
    NUM_INIT_NOISES = 100
    NUM_PROMPTS = 10
    BATCH_SIZE = 10 
    NUM_INFERENCE_STEPS = 500  # User requested 500
    NUM_PROJECTIONS = args.num_projections
    
    # Timestep Indices from arguments
    TIMESTEP_INDICES = args.timestep_indices
    
    # Base Output Directory for this experiment series
    EXPERIMENT_BASE_DIR = os.path.join(BASE_DIR, "results", f"jacobian_classification_diff_timestep_proj_{NUM_PROJECTIONS}_seed_{args.seed}")
    os.makedirs(EXPERIMENT_BASE_DIR, exist_ok=True)
    print(f"Experiment Base Directory: {EXPERIMENT_BASE_DIR}")
    
    print("Loading pipeline...")
    pipeline = LocalStableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)
    
    # Generate Timesteps
    pipeline.scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    
    # 1. Extract Prompts
    mem_prompts, unmem_prompts = load_prompts(NUM_PROMPTS)

    # 2. Generate Init Noises (Once for all timesteps)
    print("Generating/Loading Init Noises...")
    gen = torch.Generator(device=device).manual_seed(args.seed)
    
    # Check for user-specified global init_noise.pt first (for reproducibility)
    # Priority: 1. Command Line Arg -> 2. Default Global (init_noise.pt in BASE) -> 3. Experiment Dir
    
    if args.init_noise_path:
        load_path = args.init_noise_path
    else:
        load_path = os.path.join(BASE_DIR, "init_noise.pt")
    
    # We save init noises in the experiment base dir so they are consistent across timesteps
    init_noises_path = os.path.join(EXPERIMENT_BASE_DIR, "init_noise.pt")
    
    if os.path.exists(load_path):
        print(f"Loading init noises from {load_path}")
        init_noises = torch.load(load_path, map_location=device)
        # Verify shape
        if init_noises.shape[0] < NUM_INIT_NOISES:
             print(f"Warning: Loaded init_noise.pt has {init_noises.shape[0]} samples, but {NUM_INIT_NOISES} requested.")
        init_noises = init_noises[:NUM_INIT_NOISES]
        # Save a copy to the experiment directory for record
        if load_path != init_noises_path:
             torch.save(init_noises, init_noises_path)
    elif os.path.exists(init_noises_path):
        print(f"Loading existing init noises from {init_noises_path}")
        init_noises = torch.load(init_noises_path, map_location=device)
    else:
        # Check standard location
        common_path = os.path.join(BASE_DIR, "results", "jacobian_classification", "init_noises.pt")
        if os.path.exists(common_path):
             print(f"Loading common init noises from {common_path}")
             init_noises = torch.load(common_path, map_location=device)
             torch.save(init_noises, init_noises_path)
        else:
            print("Generating new init noises...")
            init_noises = torch.randn(
                (NUM_INIT_NOISES, pipeline.unet.config.in_channels, 64, 64),
                generator=gen,
                device=device,
                dtype=torch.float32
            )
            torch.save(init_noises, init_noises_path)
            print(f"Saved init noises to {init_noises_path}")
    
    # Ensure we have enough noises
    if len(init_noises) < NUM_INIT_NOISES:
         raise ValueError(f"Not enough init noises. Have {len(init_noises)}, need {NUM_INIT_NOISES}")
    init_noises = init_noises[:NUM_INIT_NOISES]

    # 3. Loop over Timestep Indices
    for t_idx in TIMESTEP_INDICES:
        print(f"\n==========================================")
        print(f"Running Analysis for Timestep Index: {t_idx}")
        
        # Get actual timestep value
        # Ensure index is within bounds
        if t_idx >= len(pipeline.scheduler.timesteps):
            print(f"Warning: Index {t_idx} out of bounds for {NUM_INFERENCE_STEPS} steps. Skipping.")
            continue
            
        t_val = pipeline.scheduler.timesteps[t_idx]
        print(f"Corresponding Timestep Value: {t_val.item()}")
        
        # Create sub-directory
        OUTPUT_DIR = os.path.join(EXPERIMENT_BASE_DIR, f"t_idx_{t_idx}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Output Directory: {OUTPUT_DIR}")
        
        # Compute Jacobian Norms (or Load if exists)
        results_csv_path = os.path.join(OUTPUT_DIR, "jacobian_norms_results.csv")
        
        if os.path.exists(results_csv_path):
            print(f"Loading existing results from {results_csv_path}")
            df_results = pd.read_csv(results_csv_path)
        else:
            results = []
            groups = [("Memorized", mem_prompts), ("Unmemorized", unmem_prompts)]
            
            for group_name, prompts in groups:
                print(f"Processing {group_name} prompts...")
                for p_idx, prompt in enumerate(tqdm(prompts, desc=f"{group_name} (t_idx={t_idx})")):
                    # Encode prompt
                    start_time = time.time()
                    prompt_embeds = pipeline._encode_prompt(prompt, device, 1, False, None)
                    
                    # Loop over noise batches
                    jacob_norms_prompt = []
                    num_batches = (NUM_INIT_NOISES + BATCH_SIZE - 1) // BATCH_SIZE
                    
                    for b in range(num_batches):
                        start = b * BATCH_SIZE
                        end = min(start + BATCH_SIZE, NUM_INIT_NOISES)
                        current_bs = end - start
                        
                        batch_noises = init_noises[start:end]
                        batch_prompt_embeds = prompt_embeds.repeat(current_bs, 1, 1)
                        
                        # Compute Jacobian Norm at t=t_val (USING SAME Init Noise)
                        # NOTE: We scale model input. At start (t=T) usually scale=1, but for other t it might differ.
                        # However, user said "init_noise를 input으로".
                        # Standard diffusion: latent at t is scaled.
                        # Should we use 'init_noise' (Gaussian) as the latent directly?
                        # User: "init_noise를 그대로 input으로 넣되" -> Yes, pass init_noise as latents.
                        # But we must still apply 'scale_model_input' if the scheduler requires it for that timestep?
                        # Usually scale_model_input handles sigma scaling for certain schedulers. 
                        # init_noise is N(0,I).
                        # If we assume we are just checking the local geometry at N(0,I) but conditioned on time t.
                        # Then we should probably just pass it.
                        # However, pipeline uses: latent_model_input = self.scheduler.scale_model_input(latents, t)
                        # If we skip this, we might be feeding wrong scale if we were simulating the process.
                        # But since we are probing "what if we are at N(0,I) but time is t", we should probably treat init_noise as the 'current latent'.
                        # IMPORTANT: `scale_model_input` ensures the input is correct for the U-Net.
                        # So we SHOULD use it on init_noise.
                        
                        latent_input = pipeline.scheduler.scale_model_input(batch_noises, t_val)
                        
                        j_norms = compute_jacobian_norm_batched(pipeline.unet, latent_input, t_val, batch_prompt_embeds, num_projections=NUM_PROJECTIONS)
                        jacob_norms_prompt.extend(j_norms.cpu().tolist())
                    
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    
                    # Store results
                    for s_idx, val in enumerate(jacob_norms_prompt):
                        results.append({
                            "Group": group_name,
                            "Prompt_Idx": p_idx,
                            "Sample_Idx": s_idx,
                            "JacobianNorm": val,
                            "Caption": prompt,
                            "TimeTaken": elapsed_time
                        })
    
            df_results = pd.DataFrame(results)
            df_results.to_csv(results_csv_path, index=False)
            print(f"Saved results to {results_csv_path}")

        # Classification Analysis
        print("Calculating optimal classification threshold...")
        df_sorted = df_results.sort_values(by="JacobianNorm").reset_index(drop=True)
        y_sorted = (df_sorted["Group"] == "Memorized").astype(int).values
        values = df_sorted["JacobianNorm"].values
        
        n_total = len(y_sorted)
        n_mem_total = np.sum(y_sorted)
        
        cum_mem = np.cumsum(y_sorted)
        cum_unmem = np.cumsum(1 - y_sorted)
        
        acc_A = (cum_unmem + (n_mem_total - cum_mem)) / n_total 
        acc_B = (cum_mem + ((n_total - n_mem_total) - cum_unmem)) / n_total 
        
        idx_A = np.argmax(acc_A)
        idx_B = np.argmax(acc_B)
        
        if acc_A[idx_A] >= acc_B[idx_B]:
            best_acc = acc_A[idx_A]
            best_idx = idx_A
            direction = "Memorized > Threshold"
        else:
            best_acc = acc_B[idx_B]
            best_idx = idx_B
            direction = "Memorized < Threshold" 
            
        best_thresh = values[best_idx]
        
        print(f"Optimal Threshold: {best_thresh:.6f} ({direction})")
        print(f"Accuracy: {best_acc*100:.2f}%")
        
        # Save metrics to text file
        with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
            f.write(f"Timestep Index: {t_idx}\n")
            f.write(f"Timestep Value: {t_val.item()}\n")
            f.write(f"Optimal Threshold: {best_thresh}\n")
            f.write(f"Direction: {direction}\n")
            f.write(f"Accuracy: {best_acc}\n")

        # Plotting
        print("Generating Plots...")
        sns.set_theme(style="whitegrid")
        misclassified_rate = (1 - best_acc) * 100
        
        def plot_hist_with_threshold(data, threshold, xlim=None, suffix=""):
            plt.figure(figsize=(10, 6))
            sns.histplot(
                data=data, 
                x="JacobianNorm", 
                hue="Group", 
                kde=False,           
                element="bars",      
                stat="count",        
                common_norm=False,   
                multiple="layer",    
                alpha=0.5          
            )
            sns.rugplot(data=data, x="JacobianNorm", hue="Group", height=0.05)
            
            plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f"Threshold: {threshold:.3f}")
            
            y_lim = plt.ylim()
            text_y = y_lim[1] * 0.9
            
            if xlim:
                plt.xlim(xlim)
                if xlim[0] <= threshold <= xlim[1]:
                     plt.text(threshold, text_y, f" Error: {misclassified_rate:.2f}%", color='red', fontweight='bold', va='center')
            else:
                plt.text(threshold, text_y, f" Error: {misclassified_rate:.2f}%", color='red', fontweight='bold', va='center')
        
            title = f"Raw Distribution at Index={t_idx} (T={t_val.item()}) (Error: {misclassified_rate:.2f}%)"
            if xlim: title += f" [Zoomed {xlim}]"
            plt.title(title)
            plt.xlabel("Jacobian Norm ||J||_F")
            plt.legend()
            
            filename = f"jacobian_norm_hist_t_idx_{t_idx}{suffix}.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            plt.savefig(save_path)
            plt.close()
        
        plot_hist_with_threshold(df_results, best_thresh, xlim=None, suffix="")
        plot_hist_with_threshold(df_results, best_thresh, xlim=(0, 25), suffix="_zoomed_0_25")
        
        # Box Plot
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df_results, x="Group", y="JacobianNorm", palette="Set2")
        plt.title(f"Box Plot (Index={t_idx}, T={t_val.item()})")
        save_path_box = os.path.join(OUTPUT_DIR, f"jacobian_norm_boxplot_t_idx_{t_idx}.png")
        plt.savefig(save_path_box)
        plt.close()

if __name__ == "__main__":
    main()
