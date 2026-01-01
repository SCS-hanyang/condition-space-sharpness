
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from diffusers import DDIMScheduler
from local_sd_pipeline import LocalStableDiffusionPipeline
import random
import argparse

# Configuration
RESULTS_DIR = "results/sharpness_c_vs_xt"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_memorized_prompts(n=10):
    try:
        df = pd.read_csv('prompts/memorized_laion_prompts.csv', sep=';')
        memorized = df['Caption'].tolist()[:n]
        return memorized
    except Exception as e:
        print(f"Error loading memorized prompts: {e}")
        return []

def compute_jacobian_norm_xt(unet, latents, t, encoder_hidden_states, num_projections=3):
    """
    Approximates ||J_xt||_F where J_xt = d(epsilon)/d(xt)
    """
    latents = latents.detach().requires_grad_(True)
    
    with torch.enable_grad():
        total_sq_norm = 0.0
        
        for _ in range(num_projections):
            # Check for OOM risk
            torch.cuda.empty_cache()
            
            noise_pred = unet(latents, t, encoder_hidden_states=encoder_hidden_states).sample
            
            v = torch.randn_like(noise_pred)
            v_dot_eps = torch.sum(noise_pred * v)
            
            grads = torch.autograd.grad(v_dot_eps, latents, create_graph=False)[0]
            total_sq_norm += torch.sum(grads ** 2).item()
            
            del grads, noise_pred, v, v_dot_eps
            
    return np.sqrt(total_sq_norm / num_projections)

def compute_jacobian_norm_c(unet, latents, t, encoder_hidden_states, num_projections=3):
    """
    Approximates ||J_c||_F where J_c = d(epsilon)/d(c)
    """
    encoder_hidden_states = encoder_hidden_states.detach().requires_grad_(True)
    
    with torch.enable_grad():
        total_sq_norm = 0.0
        
        for _ in range(num_projections):
             # Check for OOM risk
            torch.cuda.empty_cache()
            
            noise_pred = unet(latents, t, encoder_hidden_states=encoder_hidden_states).sample
            
            v = torch.randn_like(noise_pred)
            v_dot_eps = torch.sum(noise_pred * v)
            
            grads = torch.autograd.grad(v_dot_eps, encoder_hidden_states, create_graph=False)[0]
            total_sq_norm += torch.sum(grads ** 2).item()
            
            del grads, noise_pred, v, v_dot_eps
            
    return np.sqrt(total_sq_norm / num_projections)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_prompts", type=int, default=10, help="Number of memorized prompts")
    parser.add_argument("--k_seeds", type=int, default=10, help="Number of seeds")
    args = parser.parse_args()

    set_seed(SEED)
    
    print(f"Loading pipeline from {MODEL_ID}...")
    pipeline = LocalStableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        requires_safety_checker=False
    ).to(DEVICE)
    pipeline.set_progress_bar_config(disable=True)
    
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    memorized_prompts = load_memorized_prompts(args.n_prompts)
    if not memorized_prompts:
        return

    seeds = list(range(args.k_seeds))
    results = []

    print(f"Analyzing {len(memorized_prompts)} prompts with {len(seeds)} seeds...")
    
    # Pre-select random timesteps for each combination or per seed/prompt?
    # To get a good distribution, we can iterate through prompts and seeds, 
    # and for each, pick a random timestep.
    
    # We use 50 steps
    pipeline.scheduler.set_timesteps(50)
    timesteps_list = pipeline.scheduler.timesteps.tolist()
    
    # Total iterations needed for progress bar
    total_iters = len(memorized_prompts) * len(seeds)
    
    pbar = tqdm(total=total_iters, desc="Collecting Data")

    for prompt_idx, prompt in enumerate(memorized_prompts):
        # Encode prompt once
        c = pipeline._encode_prompt(prompt, DEVICE, 1, False, None)
        
        for seed in seeds:
            # Analyze at T = T_max (Initial Noise Selection)
            # This corresponds to Index 0 in the scheduler's timesteps list (if set to 50 steps, t=981 approx, but we want pure noise start)
            # Actually pipeline.prepare_latents gives pure noise.
            # The scheduler's first timestep dictates the noise level.
            
            # Use Index 0 (t=981 usually for 50 steps)
            target_t_idx = 0
            target_t = timesteps_list[target_t_idx]
            
            height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
            width = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
            
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            
            # Prepare pure random noise (x_T)
            latents = pipeline.prepare_latents(
                1, pipeline.unet.config.in_channels, height, width,
                torch.float32, torch.device(DEVICE), generator, None
            )
            
            # At T=max, x_T is just the random noise. 
            # We must scale it if the scheduler requires (usually simga included or no)
            # DDPMScheduler / DDIM usually require scaling input by sigma if prediction type is different
            # But standard SD 1.4: latents are already N(0,1).
            # The input to UNet needs scaling.
            
            lat_in = pipeline.scheduler.scale_model_input(latents, target_t)
            
            # Compute Jacobians at Initial T
            j_xt = compute_jacobian_norm_xt(pipeline.unet, lat_in, target_t, c, num_projections=3)
            j_c = compute_jacobian_norm_c(pipeline.unet, lat_in, target_t, c, num_projections=3)
            
            results.append({
                "Prompt": prompt,
                "Prompt_Idx": prompt_idx,
                "Seed": seed,
                "Timestep": target_t, 
                "Step_Index": target_t_idx,
                "J_xt": j_xt,
                "J_c": j_c
            })
            
            pbar.update(1)
            torch.cuda.empty_cache()

            
    pbar.close()
    
    # Save Raw Data
    df = pd.DataFrame(results)
    df.to_csv(f"{RESULTS_DIR}/sharpness_c_vs_xt.csv", index=False)
    
    # Analysis & Plotting
    print("Generating plots...")
    
    # 1. Scatter Plot: J_c vs J_xt
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="J_xt", y="J_c", hue="Step_Index", palette="viridis")
    plt.title(f"Relationship: Jacobian w.r.t $x_t$ vs Jacobian w.r.t $c$\n(n={args.n_prompts}, k={args.k_seeds})")
    plt.xlabel("Norm of Jacobian w.r.t Latents ($||J_{x_t}||$)")
    plt.ylabel("Norm of Jacobian w.r.t Conditioning ($||J_c||$)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{RESULTS_DIR}/scatter_xt_vs_c.png")
    plt.close()
    
    # 2. Correlation Analysis
    corr_pearson = df[["J_xt", "J_c"]].corr(method='pearson').iloc[0, 1]
    corr_spearman = df[["J_xt", "J_c"]].corr(method='spearman').iloc[0, 1]
    
    print(f"J_xt vs J_c Correlation -- Pearson: {corr_pearson:.4f}, Spearman: {corr_spearman:.4f}")
    
    # 3. Hexbin plot for density if many points, or RegPlot
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df, x="J_xt", y="J_c", scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title(f"Regression: $J_c$ vs $J_{{x_t}}$ (Corr: {corr_pearson:.2f})")
    plt.xlabel("$||J_{x_t}||$")
    plt.ylabel("$||J_c||$")
    plt.savefig(f"{RESULTS_DIR}/regression_xt_vs_c.png")
    plt.close()

    # 4. Plot over Time (Step Index)
    # Check if relationship changes over time
    # We can plot both on same axes normalized, or scatter J ratio vs Time
    
    plt.figure(figsize=(10, 6))
    # Melt for shared axis
    df_melt = df.melt(id_vars=["Step_Index"], value_vars=["J_xt", "J_c"], var_name="Type", value_name="Norm")
    sns.lineplot(data=df_melt, x="Step_Index", y="Norm", hue="Type")
    plt.title("Jacobian Norms over Diffusion Steps (Aggregated)")
    plt.xlabel("Step Index (0 = Start/Noise, 50 = End/Image)")
    plt.ylabel("Jacobian Norm")
    plt.savefig(f"{RESULTS_DIR}/norms_over_time.png")
    plt.close()

if __name__ == "__main__":
    main()
