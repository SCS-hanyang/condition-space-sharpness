
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import sys
import shutil

# Local Imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from local_sd_pipeline import LocalStableDiffusionPipeline
except ImportError:
    print("Warning: LocalStableDiffusionPipeline not found, falling back to standard.")
    from diffusers import StableDiffusionPipeline as LocalStableDiffusionPipeline
from diffusers import DDIMScheduler

# --- Helper Functions ---

def compute_jacobian_norm_batched(unet, latents, t, prompt_embeds, num_projections=1):
    """
    Batched Hutchinson estimator for ||J||_F.
    latents: [B, 4, 64, 64]
    prompt_embeds: [B, Seq, Dim]
    """
    prompt_embeds = prompt_embeds.detach()
    latents = latents.detach()
    batch_size = latents.shape[0]
    
    with torch.enable_grad():
        prompt_embeds.requires_grad_(True)
        # noise_pred: [B, 4, 64, 64]
        noise_pred = unet(latents, t, encoder_hidden_states=prompt_embeds).sample
        
        sq_norm_sum = torch.zeros(batch_size, device=latents.device)
        
        for k in range(num_projections):
            v = torch.randn_like(noise_pred)
            v_dot_eps_sum = torch.sum(noise_pred * v)
            
            grads = torch.autograd.grad(v_dot_eps_sum, prompt_embeds, create_graph=False)[0]
            # grads: [B, Seq, Dim]
            
            grads_sq_flat = grads.view(batch_size, -1).pow(2)
            sample_sq_norms = grads_sq_flat.sum(dim=1)
            sq_norm_sum += sample_sq_norms.detach()
            
        est_frob_sq = sq_norm_sum / max(1, float(num_projections))
        return est_frob_sq.sqrt()

def load_selected_prompts():
    """
    Load 50 memorized and 10 non-memorized prompts.
    """
    prompts_dir = "/home/gpuadmin/cssin/init_noise_diffusion_memorization/prompts"
    mem_path = os.path.join(prompts_dir, "memorized_laion_prompts.csv")
    unmem_path = os.path.join(prompts_dir, "unmemorized_prompts.csv")
    
    if not os.path.exists(mem_path) or not os.path.exists(unmem_path):
        raise FileNotFoundError("Prompt CSV files not found.")

    df_mem = pd.read_csv(mem_path, sep=';')
    df_unmem = pd.read_csv(unmem_path, sep=';')
    
    # Select 50 memorized, 10 non-memorized
    mem_prompts = df_mem['Caption'].tolist()[:50]
    unmem_prompts = df_unmem['Caption'].tolist()[:10]
    
    return mem_prompts, unmem_prompts

def run_analysis(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "CompVis/stable-diffusion-v1-4"
    
    print(f"Loading Model: {model_id}")
    pipeline = LocalStableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)
    
    mem_prompts, unmem_prompts = load_selected_prompts()
    print(f"Loaded {len(mem_prompts)} memorized and {len(unmem_prompts)} non-memorized prompts.")
    
    all_prompts = [("Memorized", p) for p in mem_prompts] + [("Non-memorized", p) for p in unmem_prompts]
    
    num_inference_steps = args.num_inference_steps
    num_initial_noises = args.num_initial_noises
    batch_size = args.batch_size
    
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    
    # Pre-encode Uncond
    uncond_prompt_embeds_single = pipeline._encode_prompt([""], device, 1, False, None) # [1, 77, 768]
    
    base_output_dir = "results/memorization_basin_analysis"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # --- Main Loop ---
    for p_idx, (group, prompt) in enumerate(tqdm(all_prompts, desc="Processing Prompts")):
        
        # Setup Output Directory for this prompt
        prompt_folder_name = f"{group}_{p_idx:03d}"
        prompt_dir = os.path.join(base_output_dir, prompt_folder_name)
        os.makedirs(prompt_dir, exist_ok=True)
        
        # Save Caption
        with open(os.path.join(prompt_dir, "caption.txt"), "w") as f:
            f.write(prompt)
            
        # Encode Prompt
        cond_embeds_single = pipeline._encode_prompt(prompt, device, 1, False, None)
        
        # Storage for this prompt
        # We will collect dataframes then concat
        prompt_data = []
        
        # Batch Loop over Initial Noises
        num_batches = (num_initial_noises + batch_size - 1) // batch_size
        
        for b_idx in tqdm(range(num_batches), desc=f"Prompt {p_idx+1} Batches", leave=False):
            start_n = b_idx * batch_size
            end_n = min(start_n + batch_size, num_initial_noises)
            curr_bs = end_n - start_n
            
            # 1. Generate Initial Noise (x_T)
            # Seeds: p_idx * 1000 + noise_idx (to ensure uniqueness across prompts/noises if needed, or just simple counter)
            # User said "same initial noise" (across prompts? or just fixed set?).
            # "같은 initial noise에 대해, 각각의 prompt에서..." -> likely means SAME SET of noises for ALL prompts.
            # So seeds should depend only on noise_idx.
            
            init_noises = []
            for i in range(curr_bs):
                seed = start_n + i
                g = torch.Generator(device=device).manual_seed(seed)
                n = torch.randn((1, 4, 64, 64), generator=g, device=device)
                init_noises.append(n)
            init_noises = torch.cat(init_noises, dim=0) # [B, 4, 64, 64]
            
            # Prepare Embeddings
            batch_cond = cond_embeds_single.repeat(curr_bs, 1, 1)
            batch_uncond = uncond_prompt_embeds_single.repeat(curr_bs, 1, 1)
            
            # --- Trajectory Analysis (Attraction Basin) ---
            # We follow Uncond Trajectory
            traj_latents = init_noises.clone()
            
            # --- Pure Noise Analysis ---
            # Pure Noise is static at each step (relative to x_T)
            # Actually pure noise at t is just init_noises? 
            # In pure_noise_basin_analysis.py: latent_input = pipeline.scheduler.scale_model_input(batch_noises, t)
            # We use the SAME init_noises for pure noise analysis.
            
            # Storage for batch
            # Shape: [B, T]
            traj_basin_norms = torch.zeros(curr_bs, len(timesteps))
            traj_jacob_norms = torch.zeros(curr_bs, len(timesteps))
            
            pure_basin_norms = torch.zeros(curr_bs, len(timesteps))
            pure_jacob_norms = torch.zeros(curr_bs, len(timesteps))
            
            # Loop Timesteps
            for t_idx, t in enumerate(timesteps):
                # -------------------------------------------------
                # 1. Trajectory (Attraction Basin)
                # -------------------------------------------------
                # Current State: traj_latents
                # Measure Basin Distance (Cond vs Uncond noise pred) at this point
                
                scaled_traj = pipeline.scheduler.scale_model_input(traj_latents, t)
                
                # A. Jacobian (Trajectory)
                j_traj = compute_jacobian_norm_batched(pipeline.unet, scaled_traj, t, batch_cond)
                traj_jacob_norms[:, t_idx] = j_traj.cpu()
                
                # B. Basin Distance (Trajectory)
                with torch.no_grad():
                    noise_cond_traj = pipeline.unet(scaled_traj, t, encoder_hidden_states=batch_cond).sample
                    noise_uncond_traj = pipeline.unet(scaled_traj, t, encoder_hidden_states=batch_uncond).sample
                    
                    diff_traj = (noise_cond_traj - noise_uncond_traj).reshape(curr_bs, -1).norm(dim=1)
                    traj_basin_norms[:, t_idx] = diff_traj.cpu()
                    
                # C. Step (Unconditional Generation)
                # We update traj_latents using noise_uncond_traj
                with torch.no_grad():
                    traj_latents = pipeline.scheduler.step(noise_uncond_traj, t, traj_latents).prev_sample


                # -------------------------------------------------
                # 2. Pure Noise Basin
                # -------------------------------------------------
                # Current State: init_noises (scaled)
                # Note: pure_noise_basin_analysis uses the SAME init noise for all steps?
                # Yes: batch_noises = all_init_noises[start:end]; latent_input = scale(batch_noises, t)
                
                scaled_pure = pipeline.scheduler.scale_model_input(init_noises, t)
                
                # A. Jacobian (Pure)
                j_pure = compute_jacobian_norm_batched(pipeline.unet, scaled_pure, t, batch_cond)
                pure_jacob_norms[:, t_idx] = j_pure.cpu()
                
                # B. Basin Distance (Pure)
                with torch.no_grad():
                    noise_cond_pure = pipeline.unet(scaled_pure, t, encoder_hidden_states=batch_cond).sample
                    noise_uncond_pure = pipeline.unet(scaled_pure, t, encoder_hidden_states=batch_uncond).sample
                    
                    diff_pure = (noise_cond_pure - noise_uncond_pure).reshape(curr_bs, -1).norm(dim=1)
                    pure_basin_norms[:, t_idx] = diff_pure.cpu()
                    
            # --- End Timestep Loop ---
            
            # Aggregate Batch Data
            for local_i in range(curr_bs):
                global_sample_idx = start_n + local_i
                
                # Save Individual Sample Data
                sample_dir = os.path.join(prompt_dir, f"sample_{global_sample_idx}")
                os.makedirs(sample_dir, exist_ok=True)
                
                # Build DataFrame for this sample
                df_sample = pd.DataFrame({
                    "Timestep": timesteps.cpu().numpy(),
                    "Step": np.arange(len(timesteps)),
                    "Traj_Basin_Diff": traj_basin_norms[local_i].numpy(),
                    "Traj_Jacobian": traj_jacob_norms[local_i].numpy(),
                    "Pure_Basin_Diff": pure_basin_norms[local_i].numpy(),
                    "Pure_Jacobian": pure_jacob_norms[local_i].numpy(),
                })
                
                sample_csv_path = os.path.join(sample_dir, "metrics.csv")
                df_sample.to_csv(sample_csv_path, index=False)
                
                # Also Add to Aggregate
                df_sample["Sample_Idx"] = global_sample_idx
                df_sample["Group"] = group
                prompt_data.append(df_sample)
                
        # --- End Batch Loop ---
        
        # Save Prompt Aggregate
        df_prompt_all = pd.concat(prompt_data, ignore_index=True)
        df_prompt_all.to_csv(os.path.join(prompt_dir, "summary.csv"), index=False)
        
        # Plotting for Prompt
        plot_prompt_summary(df_prompt_all, prompt_dir, group)
        
        # Cleanup
        del prompt_data, df_prompt_all
        torch.cuda.empty_cache()

def plot_prompt_summary(df, output_dir, group_name):
    """
    Generate plots comparing Trajectory vs Pure Noise Analysis.
    """
    sns.set_style("whitegrid")
    
    # 1. Basin Distance Comparison
    plt.figure(figsize=(10, 6))
    
    # Aggregate over samples
    sns.lineplot(data=df, x="Step", y="Traj_Basin_Diff", label="Attraction Basin (Traj)", ci="sd")
    sns.lineplot(data=df, x="Step", y="Pure_Basin_Diff", label="Pure Noise Basin", ci="sd", linestyle="--")
    
    plt.xlabel("Step (0=T, End=0)") # Note: Step usually goes 0->Cond, but Timestep goes T->0. 
    # In my code, Step 0 = First step of loop = Timestep 981 (approx).
    plt.ylabel("||Cond - Uncond||")
    plt.title(f"Basin Distance Comparison: {group_name}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "compare_basin_distance.png"))
    plt.close()
    
    # 2. Jacobian Comparison
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(data=df, x="Step", y="Traj_Jacobian", label="Attraction Basin (Traj)", ci="sd")
    sns.lineplot(data=df, x="Step", y="Pure_Jacobian", label="Pure Noise Basin", ci="sd", linestyle="--")
    
    plt.xlabel("Step")
    plt.ylabel("Jacobian Norm")
    plt.title(f"Jacobian Norm Comparison: {group_name}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "compare_jacobian.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--num_initial_noises", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=5)
    
    args = parser.parse_args()
    
    run_analysis(args)
