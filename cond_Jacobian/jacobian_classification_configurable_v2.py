
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from diffusers import DDIMScheduler, StableDiffusionPipeline

import argparse

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
BASE_DIR = "/home/gpuadmin/cssin/cond_Jacobian"
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
# OUTPUT_DIR will be defined in main based on args

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
        noise_pred = unet(latents, t, encoder_hidden_states=prompt_embeds).sample
        
        sq_norm_sum = torch.zeros(batch_size, device=latents.device)
        
        for k in range(num_projections):
            # Random probe v
            v = torch.randn_like(noise_pred)
            
            v_dot_eps_sum = torch.sum(noise_pred * v)
            
            retain = (k < num_projections - 1)
            grads = torch.autograd.grad(v_dot_eps_sum, prompt_embeds, retain_graph=retain, create_graph=False)[0]
            
            # Squared norm per sample
            grads_sq_flat = grads.view(batch_size, -1).pow(2)
            sample_sq_norms = grads_sq_flat.sum(dim=1) # [B]
            
            sq_norm_sum += sample_sq_norms.detach()
            
        est_frob_sq = sq_norm_sum / max(1, float(num_projections))
        return est_frob_sq.sqrt() # [B]

def load_prompts_txt(mem_path, unmem_path, num_mem=None, num_unmem=None):
    try:
        with open(mem_path, 'r') as f:
            mem_prompts = [line.strip() for line in f.readlines() if line.strip()]
        
        with open(unmem_path, 'r') as f:
            unmem_prompts = [line.strip() for line in f.readlines() if line.strip()]
        
        if num_mem:
            mem_prompts = mem_prompts[:min(len(mem_prompts), num_mem)]
        if num_unmem:
            unmem_prompts = unmem_prompts[:min(len(unmem_prompts), num_unmem)]
        
        print(f"Loaded {len(mem_prompts)} memorized prompts and {len(unmem_prompts)} unmemorized prompts.")
        return mem_prompts, unmem_prompts
    except Exception as e:
        print(f"Error loading prompts: {e}")
        # Fallback for testing/debugging
        return ["A cool cat"]*5, ["A boring dog"]*5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_projections", type=int, default=1, help="Hutchinson estimator projections")
    parser.add_argument("--num_mem_prompts", type=int, default=219, help="Number of memorized prompts to use")
    parser.add_argument("--num_unmem_prompts", type=int, default=219, help="Number of unmemorized prompts to use")
    args = parser.parse_args()

    # Configuration
    model_id = "sd2-community/stable-diffusion-2"
    NUM_INIT_NOISES = 100
    NUM_MEM_PROMPTS = args.num_mem_prompts
    NUM_UNMEM_PROMPTS = args.num_unmem_prompts
    BATCH_SIZE = 10 
    NUM_INFERENCE_STEPS = 1000 
    NUM_PROJECTIONS = args.num_projections
    
    # Prompt Files
    mem_file_path = os.path.join(PROMPTS_DIR, "sd2_mem219.txt")
    unmem_file_path = os.path.join(PROMPTS_DIR, "sd2_nmem219.txt")

    # Update Output Dir to reflect model and prompt counts
    # We will append the actual model used to the dir name later if possible, 
    # but for now let's keep the base name and just accept we might be using a fallback.
    dir_name = f"jacobian_classification_sd2_proj_{NUM_PROJECTIONS}_m{NUM_MEM_PROMPTS}_u{NUM_UNMEM_PROMPTS}"
    OUTPUT_DIR = os.path.join(BASE_DIR, "results", dir_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # Determine dtype
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    print(f"Loading pipeline: {model_id}...")
    pipeline = None
    
    # List of models to try in order
    models_to_try = [
        "sd2-community/stable-diffusion-2",
        "sd2-community/stable-diffusion-2-base",
        "sd2-community/stable-diffusion-2-1",
        "sd2-community/stable-diffusion-2-1-base",
    ]
    
    for model in models_to_try:
        print(f"Attempting to load: {model}")
        try:
            pipeline = LocalStableDiffusionPipeline.from_pretrained(
                model, 
                torch_dtype=dtype,
                local_files_only=False,
                cache_dir="/home/gpuadmin/cssin/hf_cache",
            ).to(device)
            pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
            pipeline.set_progress_bar_config(disable=True)
            print(f"Successfully loaded {model}")
            model_id = model # Update model_id to the one that actually loaded
            break
        except Exception as e:
            print(f"Failed to load {model}: {e}")
            continue
            
    if pipeline is None:
        print("CRITICAL ERROR: Could not load ANY model. Exiting.")
        return

    # 1. Extract Prompts
    mem_prompts, unmem_prompts = load_prompts_txt(mem_file_path, unmem_file_path, NUM_MEM_PROMPTS, NUM_UNMEM_PROMPTS)

    # 2. Generate Init Noises
    print("Generating Init Noises...")
    # Use a fixed generator for reproducibility
    seed = 42
    gen = torch.Generator(device=device).manual_seed(seed)
    
    # Save init noises to disk to ensure we track what was used
    init_noises_path = os.path.join(OUTPUT_DIR, "init_noises.pt")
    
    # Determine latent size from model config
    latent_size = pipeline.unet.config.sample_size
    in_channels = pipeline.unet.config.in_channels
    print(f"Using latent size: {latent_size}x{latent_size}")

    if os.path.exists(init_noises_path):
        print(f"Loading existing init noises from {init_noises_path}")
        init_noises = torch.load(init_noises_path, map_location=device)
    else:
        init_noises = torch.randn(
            (NUM_INIT_NOISES, in_channels, latent_size, latent_size),
            generator=gen,
            device=device,
            dtype=torch.float32
        )
        torch.save(init_noises, init_noises_path)
        print(f"Saved init noises to {init_noises_path}")

    # Define T as the starting step
    pipeline.scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    t_T = pipeline.scheduler.timesteps[0]
    print(f"Using Starting Step T={t_T.item()}")

    # 3. Compute Jacobian Norms (or Load if exists)
    results_csv_path = os.path.join(OUTPUT_DIR, "jacobian_norms_results.csv")
    
    if os.path.exists(results_csv_path):
        print(f"Loading existing results from {results_csv_path}")
        df_results = pd.read_csv(results_csv_path)
    else:
        results = []
        groups = [("Memorized", mem_prompts), ("Unmemorized", unmem_prompts)]
        
        for group_name, prompts in groups:
            print(f"Processing {group_name} prompts...")
            for p_idx, prompt in enumerate(tqdm(prompts, desc=group_name)):
                # Encode prompt once
                # Note: SD2 uses the same signature in LocalStableDiffusionPipeline inherited from StableDiffusionPipeline
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
                    
                    # Compute Jacobian Norm at t=T
                    latent_input = pipeline.scheduler.scale_model_input(batch_noises, t_T)
                    
                    # Ensure dtype match
                    unet_dtype = pipeline.unet.dtype
                    latent_input = latent_input.to(dtype=unet_dtype)
                    batch_prompt_embeds = batch_prompt_embeds.to(dtype=unet_dtype)

                    j_norms = compute_jacobian_norm_batched(pipeline.unet, latent_input, t_T, batch_prompt_embeds, num_projections=NUM_PROJECTIONS)
                    jacob_norms_prompt.extend(j_norms.cpu().tolist())
                
                # Store results
                for s_idx, val in enumerate(jacob_norms_prompt):
                    results.append({
                        "Group": group_name,
                        "Prompt_Idx": p_idx,
                        "Sample_Idx": s_idx,
                        "JacobianNorm": val,
                        "Caption": prompt
                    })

        df_results = pd.DataFrame(results)
        df_results.to_csv(results_csv_path, index=False)
        print(f"Saved results to {results_csv_path}")

    # 4. Classification Analysis
    print("Calculating optimal classification threshold...")
    df_sorted = df_results.sort_values(by="JacobianNorm").reset_index(drop=True)
    y_sorted = (df_sorted["Group"] == "Memorized").astype(int).values
    values = df_sorted["JacobianNorm"].values
    
    n_total = len(y_sorted)
    n_mem_total = np.sum(y_sorted)
    
    if n_total == 0:
        print("No results to analyze.")
        return

    cum_mem = np.cumsum(y_sorted)
    cum_unmem = np.cumsum(1 - y_sorted)
    
    acc_A = (cum_unmem + (n_mem_total - cum_mem)) / n_total # Mem > T
    acc_B = (cum_mem + ((n_total - n_mem_total) - cum_unmem)) / n_total # Mem < T
    
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

    # --- Strict Misclassification Statistics ---
    print("\n--- Strict Prompt-Level Misclassification ---")
    mem_prompt_indices = df_results[df_results["Group"] == "Memorized"]["Prompt_Idx"].unique()
    unmem_prompt_indices = df_results[df_results["Group"] == "Unmemorized"]["Prompt_Idx"].unique()
    
    n_mem_prompts_actual = len(mem_prompt_indices)
    n_unmem_prompts_actual = len(unmem_prompt_indices)
    
    if direction == "Memorized > Threshold":
        # Memorized Prompt is misclassified if ANY of its samples are <= Threshold
        mis_mem_prompt_mask = df_results[df_results["Group"] == "Memorized"].groupby("Prompt_Idx")["JacobianNorm"].min() <= best_thresh
        count_mem_misclassified = mis_mem_prompt_mask.sum()
        
        # Unmemorized Prompt is misclassified if ANY of its samples are > Threshold
        mis_unmem_prompt_mask = df_results[df_results["Group"] == "Unmemorized"].groupby("Prompt_Idx")["JacobianNorm"].max() > best_thresh
        count_unmem_misclassified = mis_unmem_prompt_mask.sum()
        
    else: # Memorized < Threshold
        mis_mem_prompt_mask = df_results[df_results["Group"] == "Memorized"].groupby("Prompt_Idx")["JacobianNorm"].max() >= best_thresh
        count_mem_misclassified = mis_mem_prompt_mask.sum()
        
        mis_unmem_prompt_mask = df_results[df_results["Group"] == "Unmemorized"].groupby("Prompt_Idx")["JacobianNorm"].min() < best_thresh
        count_unmem_misclassified = mis_unmem_prompt_mask.sum()

    print(f"Memorized -> Classified as Unmemorized (Strict): {count_mem_misclassified} / {n_mem_prompts_actual} ({count_mem_misclassified/n_mem_prompts_actual*100:.2f}%)")
    print(f"Unmemorized -> Classified as Memorized (Strict): {count_unmem_misclassified} / {n_unmem_prompts_actual} ({count_unmem_misclassified/n_unmem_prompts_actual*100:.2f}%)")

    # --- Plotting ---
    print("\nGenerating Plots...")
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
    
        title = f"Raw Distribution at T={t_T.item()} (Error: {misclassified_rate:.2f}%)"
        if xlim: title += f" [Zoomed {xlim}]"
        plt.title(title)
        plt.xlabel("Jacobian Norm ||J||_F")
        plt.legend()
        
        filename = f"jacobian_norm_hist_t_T{suffix}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved histogram to {save_path}")

    # 1. Full Range Histogram
    plot_hist_with_threshold(df_results, best_thresh, xlim=None, suffix="")

    # 2. Zoomed Histogram (Optional, adjusted based on data distribution, hard to predict)
    # plot_hist_with_threshold(df_results, best_thresh, xlim=(0, 25), suffix="_zoomed_0_25")

    # 3. Box Plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_results, x="Group", y="JacobianNorm", palette="Set2")
    plt.title("Box Plot (Statistics)")
    save_path_box = os.path.join(OUTPUT_DIR, "jacobian_norm_boxplot_t_T.png")
    plt.savefig(save_path_box)
    plt.close()
    print(f"Saved boxplot to {save_path_box}")

    # 4. ECDF
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(data=df_results, x="JacobianNorm", hue="Group", linewidth=2)
    plt.axvline(x=best_thresh, color='red', linestyle='--', alpha=0.5, label="Optimal Threshold")
    plt.title("ECDF (Cumulative Distribution)")
    plt.xlabel("Jacobian Norm ||J||_F")
    plt.legend()
    save_path_ecdf = os.path.join(OUTPUT_DIR, "jacobian_norm_ecdf_t_T.png")
    plt.savefig(save_path_ecdf)
    plt.close()
    print(f"Saved ECDF to {save_path_ecdf}")

    # 5. Visualizing Misclassified Memorized Prompts
    print("\nAnalyzing Misclassified Memorized Prompts...")
    
    # Filter: Memorized prompts that were classified as Unmemorized
    if direction == "Memorized > Threshold":
        # Misclassified: Mem <= Threshold
        misclassified_mask = (df_results["Group"] == "Memorized") & (df_results["JacobianNorm"] <= best_thresh)
    else:
        # Misclassified: Mem >= Threshold
        misclassified_mask = (df_results["Group"] == "Memorized") & (df_results["JacobianNorm"] >= best_thresh)
        
    df_misclassified = df_results[misclassified_mask].copy()
    
    print(f"Found {len(df_misclassified)} misclassified noise instances (Memorized -> Unmemorized)")
    
    if len(df_misclassified) > 0:
        # Sort by deviation from threshold
        if direction == "Memorized > Threshold":
            df_misclassified = df_misclassified.sort_values(by="JacobianNorm")
        else:
            df_misclassified = df_misclassified.sort_values(by="JacobianNorm", ascending=False)
            
        vis_count = 0
        img_output_dir = os.path.join(OUTPUT_DIR, "misclassified_comparisons")
        os.makedirs(img_output_dir, exist_ok=True)
        
        processed_prompts = set()
        
        for idx, row in df_misclassified.iterrows():
            if vis_count >= 10: break
            
            p_idx = row['Prompt_Idx']
            s_idx = row['Sample_Idx']
            caption = row['Caption']
            j_norm = row['JacobianNorm']
            
            processed_prompts.add(p_idx)
            
            print(f"Generating comparison for Prompt {p_idx}, Sample {s_idx} (J={j_norm:.4f})...")
            
            # 1. Image with Misclassified Noise
            bad_noise = init_noises[int(s_idx)].unsqueeze(0) # [1, C, H, W]
            
            with torch.no_grad():
                # Use standard generative call
                image_bad = pipeline(caption, latents=bad_noise, num_inference_steps=50).images[0]
                
            # 2. Image with Random Comparison Noise
            rand_s_idx = np.random.randint(0, NUM_INIT_NOISES)
            while rand_s_idx == s_idx:
                rand_s_idx = np.random.randint(0, NUM_INIT_NOISES)
            
            rand_noise = init_noises[int(rand_s_idx)].unsqueeze(0)
            
            with torch.no_grad():
                image_rand = pipeline(caption, latents=rand_noise, num_inference_steps=50).images[0]
            
            fig, ax = plt.subplots(1, 2, figsize=(15, 7))
            
            ax[0].imshow(image_bad)
            ax[0].set_title(f"Misclassified Init Noise (Idx={s_idx})\nJacobian={j_norm:.4f}\n(Classified as Unmem)")
            ax[0].axis('off')
            
            ax[1].imshow(image_rand)
            ax[1].set_title(f"Random Int Noise (Idx={rand_s_idx})\nComparison")
            ax[1].axis('off')
            
            plt.suptitle(f"Memorized Prompt: {caption[:50]}...", fontsize=14)
            plt.tight_layout()
            
            save_path = os.path.join(img_output_dir, f"misclassified_p{p_idx}_s{s_idx}.png")
            plt.savefig(save_path)
            plt.close()
            
            vis_count += 1
            
        print(f"Saved {vis_count} comparison images to {img_output_dir}")

if __name__ == "__main__":
    main()
