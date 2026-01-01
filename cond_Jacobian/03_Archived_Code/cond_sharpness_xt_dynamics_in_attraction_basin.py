
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

# Configuration

# Configuration
RESULTS_DIR = "results/cond_sharpness_xt_dynamics_in_attraction_basin"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
NUM_STEPS = 50

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_prompts():
    # Memorized
    try:
        # Use relative path from this script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'prompts', 'memorized_laion_prompts.csv')
        df = pd.read_csv(csv_path, sep=';')
        memorized = df['Caption'].tolist()[:10]
    except Exception as e:
        print(f"Error loading memorized prompts from {csv_path}: {e}")
        try:
             # Fallback to direct path if running from root
             df = pd.read_csv('init_noise_diffusion_memorization/prompts/memorized_laion_prompts.csv', sep=';')
             memorized = df['Caption'].tolist()[:10]
        except Exception as e2:
             print(f"Fallback failed: {e2}")
             memorized = []

    # Unmemorized (Hardcoded)
    unmemorized = [
        "A beautiful sunset over a calm ocean with a sailboat in the distance",
        "A majestic mountain range covered in snow under a blue sky",
        "A dense flower garden in full bloom with colorful tulips",
        "A quiet forest path illuminated by beams of sunlight",
        "A vast sandy beach extending to the horizon at dawn",
        "A calm lake reflecting the autumn trees and cloudy sky",
        "A flowing waterfall in the middle of a tropical rainforest",
        "A grassy field with wild flowers waving in the wind",
        "A night sky filled with bright stars and a full moon",
        "A foggy morning in a pine forest with dew on the ground",
    ]
    # Ensure 10
    while len(unmemorized) < 10:
        unmemorized.append("A random nature scene test prompt")
    unmemorized = unmemorized[:10]
    return memorized, unmemorized

def compute_jacobian_norm_xt(unet, latents, t, prompt_embeds, num_projections=3):
    """
    Approximates the Frobenius norm of the Jacobian of the noise prediction
    with respect to the input latents (x_t) using Hutchinson's estimator.
    J_xt = d(epsilon)/d(x_t)
    """
    # Detach to make it a leaf variable
    latents = latents.detach()
    
    with torch.enable_grad():
        latents.requires_grad_(True)
        
        total_sq_norm = 0.0
        
        for _ in range(num_projections):
            # Check OOM
            torch.cuda.empty_cache()
            
            # Forward pass
            noise_pred = unet(latents, t, encoder_hidden_states=prompt_embeds).sample
            
            # Random projection
            v = torch.randn_like(noise_pred)
            v_dot_eps = torch.sum(noise_pred * v)
            
            # Compute gradient w.r.t latents
            grads = torch.autograd.grad(v_dot_eps, latents, create_graph=False)[0]
            
            # Accumulate
            total_sq_norm += torch.sum(grads ** 2).item()
            
            del grads, noise_pred, v, v_dot_eps
            
        return np.sqrt(total_sq_norm / num_projections)

def main():
    set_seed(SEED)
    
    print("Loading pipeline...")
    pipeline = LocalStableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32, 
        requires_safety_checker=False
    ).to(DEVICE)
    pipeline.set_progress_bar_config(disable=True)
    
    memorized_prompts, unmemorized_prompts = load_prompts()
    print(f"Memorized: {len(memorized_prompts)}, Unmemorized: {len(unmemorized_prompts)}")
    
    # 1. Generate Unconditional Trajectory (Fixed Seed)
    print("Generating Unconditional Trajectory...")
    
    unc_prompt_embeds = pipeline._encode_prompt("", DEVICE, 1, False, None)
    
    height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    width = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    
    pipeline.scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    timesteps = pipeline.scheduler.timesteps
    
    latents = pipeline.prepare_latents(
        1, pipeline.unet.config.in_channels, height, width,
        torch.float32, torch.device(DEVICE), torch.Generator(device=DEVICE).manual_seed(SEED), None
    )
    
    trajectory_latents = [] 
    trajectory_uncond_noise = []
    
    curr_latents = latents
    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps, desc="Uncond Trajectory")):
            trajectory_latents.append(curr_latents.clone())
            
            latent_model_input = pipeline.scheduler.scale_model_input(curr_latents, t)
            noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=unc_prompt_embeds).sample
            
            trajectory_uncond_noise.append(noise_pred.clone())
            curr_latents = pipeline.scheduler.step(noise_pred, t, curr_latents, return_dict=False)[0]
            
    print("Unconditional Trajectory Generated.")
    
    results = []
    
    for group_name, prompts in [("Memorized", memorized_prompts), ("Unmemorized", unmemorized_prompts)]:
        print(f"Analyzing {group_name}...")
        
        curves = []
        
        # Step 2: Find Basins
        for prompt in tqdm(prompts, desc=f"{group_name} Curves"):
            cond_embeds = pipeline._encode_prompt(prompt, DEVICE, 1, False, None)
            diff_norms = []
            
            with torch.no_grad():
                for i, t in enumerate(timesteps):
                    lat = trajectory_latents[i]
                    latent_input = pipeline.scheduler.scale_model_input(lat, t)
                    cond_noise = pipeline.unet(latent_input, t, encoder_hidden_states=cond_embeds).sample
                    uncond_noise = trajectory_uncond_noise[i]
                    diff = torch.norm(cond_noise - uncond_noise).item()
                    diff_norms.append(diff)
            curves.append(diff_norms)
            
        curves = np.array(curves)
        
        drop_indices = []
        for i, curve in enumerate(curves):
            deriv = np.diff(curve)
            t_drop_idx = np.argmin(deriv[5:]) + 5 
            drop_indices.append(t_drop_idx)
            
        avg_drop_idx = int(np.mean(drop_indices))
        if group_name == "Unmemorized":
            # Use Memorized Average for consistency if needed, but per-sample logic for now
            # The previous logic had a placeholder. Let's use individual drops.
            pass
        else:
            shared_avg_drop_idx = avg_drop_idx
            
        # 3. Jacobian (J_xt) Calculation
        print(f"Calculating J_xt for {group_name}...")
        
        for i, prompt in enumerate(tqdm(prompts, desc="Jacobians")):
            drop_idx = drop_indices[i]
            
            inside_indices = list(range(0, drop_idx + 1))
            outside_indices = list(range(drop_idx + 1, min(drop_idx + 5, NUM_STEPS)))
            calc_indices = inside_indices + outside_indices
            
            prompt_embeds_te = pipeline._encode_prompt(prompt, DEVICE, 1, False, None)
            
            for t_idx in calc_indices:
                t = timesteps[t_idx]
                lat = trajectory_latents[t_idx]
                
                # Careful: Jacobian w.r.t 'latents' (x_t)
                # We need to pass the UNet input.
                # Note: unet takes 'latent_model_input' which is scaled.
                # But physically we want d(eps)/d(x_t).
                # scale_model_input is just a standard scaling.
                # Let's compute gradients w.r.t the 'latent_model_input' that actually goes into UNet?
                # Or w.r.t the 'latents' x_t?
                # Usually we care about the sensitivity of the diffusion state.
                # Since scaling is linear, it just adds a constant factor.
                # Let's differentiate w.r.t 'latent_input' passed to UNet for cleaner implementation,
                # or 'lat' and pass through scaler. 
                # Passing through scaler is strictly correct but 'scale_model_input' doesn't support grad easily if it's just tensor ops? 
                # It does support grad.
                # But usually we input 'latent_model_input' to UNet.
                # Let's define J_xt as d(eps) / d(x_t_input_to_unet) to avoid scaler complexity,
                # as that is the "effective" input to the neural network.
                
                latent_input = pipeline.scheduler.scale_model_input(lat, t)
                
                j_norm_xt = compute_jacobian_norm_xt(pipeline.unet, latent_input, t, prompt_embeds_te)
                
                region = "Inside Basin" if t_idx <= drop_idx else "Outside Basin"
                
                results.append({
                    "Prompt": prompt,
                    "Group": group_name,
                    "Drop_Index": drop_idx,
                    "Step_Index": t_idx,
                    "Relative_Step": t_idx - drop_idx,
                    "Region": region,
                    "J_Norm_xt": j_norm_xt,
                    "Diff_Curve": curves[i].tolist()
                })
            
    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{RESULTS_DIR}/jacobian_xt_analysis.csv", index=False)
    
    # 4. Visualization (Line Plot Aggregated)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_res, x="Relative_Step", y="J_Norm_xt", hue="Group", marker="o")
    plt.axvline(x=0, color='r', linestyle='--', label='Basin Exit')
    plt.title("Spatial Sharpness (J_xt) Dynamics relative to Basin Exit")
    plt.ylabel(r"$\| J_{x_t} \|_F$")
    plt.xlabel("Relative Step (0 = Exit)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{RESULTS_DIR}/jacobian_xt_dynamics.png")
    plt.close()
    
    print("Analysis Complete.")

if __name__ == "__main__":
    main()
