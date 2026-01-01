
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from diffusers import DDIMScheduler
from local_sd_pipeline import LocalStableDiffusionPipeline
import random

# Configuration
RESULTS_DIR = "results/cond_sharpness_dynamics_in_attraction_basin"
os.makedirs(RESULTS_DIR, exist_ok=True)
MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
NUM_STEPS = 50

# Mitigation Params
RHO = 1.0 
M = 5
GAMMAS = [0.0, 0.5, 1.0, 1.5]

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_first_memorized_prompt():
    try:
        df = pd.read_csv('prompts/memorized_laion_prompts.csv', sep=';')
        return df['Caption'].iloc[0]
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return "A photograph of an astronaut riding a horse"

def main():
    print(f"Loading pipeline from {MODEL_ID}...")
    pipeline = LocalStableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        requires_safety_checker=False
    ).to(DEVICE)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    prompt = load_first_memorized_prompt()
    print(f"Target Prompt: {prompt[:50]}...")
    
    # Analyze for each Gamma
    results = {}
    
    for gamma in GAMMAS:
        print(f"Running for Gamma = {gamma}...")
        set_seed(SEED) # Reset seed for fair comparison of initial noise
        
        # 1. Prepare Initial Noise (x_T)
        height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
        width = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
        
        generator = torch.Generator(device=DEVICE).manual_seed(SEED)
        latents = pipeline.prepare_latents(
            1, pipeline.unet.config.in_channels, height, width,
            torch.float32, torch.device(DEVICE), generator, None
        )
        
        # 2. Apply Mitigation (if Gamma > 0)
        # We need conditional embeddings for the mitigation step
        if gamma > 0:
            # Mitigation uses the PROMPT (conditional) gradient
            cond_embeds = pipeline._encode_prompt(
                prompt, DEVICE, 1, False, None # num_images_per_prompt=1, do_classifier_free_guidance=False
            )
            # The adj_init_noise_batch_wise method expects eps_cond - eps_uncond logic usually?
            # Let's check implementation of adj_init_noise_batch_wise in local_sd_pipeline.py
            # It enables grad. 
            # noise_pred = unet(latents... encoder_hidden_states=prompt_embeds)
            # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            # eps_tilde = noise_pred_text - noise_pred_uncond
            
            # So it REQUIRES prompt_embeds to consist of [uncond, cond] concatenated!
            # We must provide correct prompt_embeds shape.
            
            # Encode with CFG structure
            # But wait, _encode_prompt returns 'prompt_embeds'. 
            # If do_classifier_free_guidance=True, it returns cat([neg, pos]).
            
            full_embeds = pipeline._encode_prompt(
                prompt, DEVICE, 1, True, None # do_cfg=True to get [neg, pos]
            )
            
            # Call adjustment
            with torch.enable_grad():
                latents = pipeline.adj_init_noise_batch_wise(
                    num_inference_steps=NUM_STEPS,
                    guidance_scale=7.5, # Used inside for CFG calc in adjustment
                    latents=latents,
                    prompt_embeds=full_embeds,
                    adj_iters=M,
                    rho=RHO,
                    gamma=gamma
                )
            # Detach after optimization
            latents = latents.detach()
            
        # 3. Unconditional Generation Loop
        # We start from the (possibly adjusted) latents
        # We generate UNCONDITIONALLY (prompt="")
        
        uncond_embeds = pipeline._encode_prompt(
            "", DEVICE, 1, False, None # Just unconditional
        )
        
        pipeline.scheduler.set_timesteps(NUM_STEPS)
        timesteps = pipeline.scheduler.timesteps
        
        norms = []
        
        curr_latents = latents
        with torch.no_grad():
            for t in timesteps:
                # Expand for Unet? Unconditional is single batch usually.
                # But pipeline unet expects latent_model_input.
                
                latent_input = pipeline.scheduler.scale_model_input(curr_latents, t)
                
                # Predict Noise (Unconditional)
                noise_pred = pipeline.unet(
                    latent_input, 
                    t, 
                    encoder_hidden_states=uncond_embeds
                ).sample
                
                # Check metric: L2 Norm of predicted noise
                # This corresponds to || eps_theta(x_t, t, empty) ||
                norm_val = torch.norm(noise_pred).item()
                norms.append(norm_val)
                
                # Step
                curr_latents = pipeline.scheduler.step(
                    noise_pred, 
                    t, 
                    curr_latents, 
                    return_dict=False
                )[0]
                
        results[gamma] = (timesteps.cpu().numpy(), norms)
        
    # 4. Plotting
    plt.figure(figsize=(8, 6))
    
    colors = {0.0: 'red', 0.5: 'green', 1.0: 'orange', 1.5: 'tab:blue'}
    labels = {0.0: 'Baseline'}
    
    for gamma, (ts, ns) in results.items():
        label = labels.get(gamma, f"$\\tilde{{\\gamma}}={gamma}$")
        color = colors.get(gamma, 'black')
        plt.plot(ts, ns, label=label, color=color, linewidth=2)
        
    plt.gca().invert_xaxis() # 1000 -> 0
    plt.xlabel("Time steps")
    plt.ylabel(r"$||\tilde{\epsilon}_{\theta}(x_t, t, y)||_2$ (Unconditional)")
    plt.title("Noise Norm Dynamics (Unconditional Generation)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_path = f"{RESULTS_DIR}/mitigation_dynamics_uncond.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()
