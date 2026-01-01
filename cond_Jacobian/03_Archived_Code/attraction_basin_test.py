
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from diffusers import StableDiffusionPipeline
import shutil

def load_prompts():
    try:
        # Check standard locations
        paths = [
            'prompts/memorized_laion_prompts.csv',
            'init_noise_diffusion_memorization/prompts/memorized_laion_prompts.csv',
            '../prompts/memorized_laion_prompts.csv' # If running from subdir
        ]
        found_path = None
        for p in paths:
            if os.path.exists(p):
                found_path = p
                break
        
        if found_path:
            df = pd.read_csv(found_path, sep=';')
            memorized = df['Caption'].tolist()
        else:
            print("Warning: CSV not found in common locations.")
            memorized = ["Mothers influence on her young hippo"] * 10
            
    except Exception as e:
        print(f"Warning: Error loading CSV: {e}")
        memorized = ["Mothers influence on her young hippo"] * 10
    return memorized

def run_attraction_basin_test():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "CompVis/stable-diffusion-v1-4"
    results_dir = "init_noise_diffusion_memorization/results/attraction_basin_test"
    
    # Don't wipe if we want to run in parallel or resume, but here we wipe for clean test
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    print("Loading SD Pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    
    memorized_prompts = load_prompts()
    memorized_prompts = list(dict.fromkeys(memorized_prompts))
    count = min(len(memorized_prompts), 100)
    
    uncond_prompt_embeds = pipe._encode_prompt("", device, 1, False, None)
    
    num_inference_steps = 50
    height = pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = pipe.unet.config.sample_size * pipe.vae_scale_factor
    
    print(f"Starting Attraction Basin Test for {count} prompts...")
    
    for idx in range(count):
        prompt = memorized_prompts[idx]
        print(f"[{idx+1}/{count}] Processing: {prompt[:30]}...")
        
        sample_dir = os.path.join(results_dir, f"sample_{idx:03d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        cond_prompt_embeds = pipe._encode_prompt(prompt, device, 1, False, None)
        
        gen = torch.Generator(device).manual_seed(42)
        latents = pipe.prepare_latents(
            1, pipe.unet.config.in_channels, height, width,
            cond_prompt_embeds.dtype, device, gen, None
        )
        init_latents = latents.clone()
        
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps 
        
        basin_diff_curve = []
        latents_basin = latents.clone()

        # Wrapper for all inference to prevent OOM
        with torch.no_grad():
            # --- Find Attraction Basin Exit ---
            for t in timesteps:
                noise_cond = pipe.unet(pipe.scheduler.scale_model_input(latents_basin, t), t, encoder_hidden_states=cond_prompt_embeds).sample
                noise_uncond = pipe.unet(pipe.scheduler.scale_model_input(latents_basin, t), t, encoder_hidden_states=uncond_prompt_embeds).sample
                diff = torch.norm(noise_cond - noise_uncond).item()
                basin_diff_curve.append(diff)
                latents_basin = pipe.scheduler.step(noise_uncond, t, latents_basin, return_dict=False)[0]

            basin_exit_idx = np.argmax(basin_diff_curve)
        
            # --- Define 6 Checkpoints ---
            # 3 internal, 1 exit, 2 post
            cp_exit = basin_exit_idx
            
            # Internal points (0 to cp_exit-1)
            if cp_exit >= 3:
                cp_in_1 = int(cp_exit * 0.25)
                cp_in_2 = int(cp_exit * 0.50)
                cp_in_3 = int(cp_exit * 0.75)
            else:
                cp_in_1, cp_in_2, cp_in_3 = 0, 0, 0
                
            cp_post_1 = min(num_inference_steps - 1, cp_exit + 2)
            cp_post_2 = min(num_inference_steps - 1, cp_exit + 5)
            
            raw_points = [cp_in_1, cp_in_2, cp_in_3, cp_exit, cp_post_1, cp_post_2]
            unique_points = sorted(list(set(raw_points)))
            
            # Fill gaps to reach 6
            while len(unique_points) < 6:
                added = False
                # try finding gaps
                for k in range(len(unique_points)-1):
                    mid = (unique_points[k] + unique_points[k+1]) // 2
                    if mid != unique_points[k] and mid not in unique_points:
                        unique_points.append(mid)
                        unique_points.sort()
                        added = True
                        break
                
                if not added:
                    # Append at end
                    if unique_points[-1] < num_inference_steps - 1:
                        unique_points.append(unique_points[-1] + 1)
                    elif unique_points[0] > 0:
                        unique_points.append(unique_points[0] - 1)
                    else:
                        break # can't add more
                    unique_points.sort()
            
            checkpoints_indices = unique_points[:6]
            
            # --- Generate 8 Samples ---
            
            # (A) Fully Conditional
            latents_cond = init_latents.clone()
            for t in timesteps:
                latent_model_input = torch.cat([latents_cond] * 2)
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
                cat_embeds = torch.cat([uncond_prompt_embeds, cond_prompt_embeds])
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=cat_embeds).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
                latents_cond = pipe.scheduler.step(noise_pred, t, latents_cond, return_dict=False)[0]
                
            img_cond = pipe.vae.decode(latents_cond / pipe.vae.config.scaling_factor, return_dict=False)[0]
            img_cond = pipe.image_processor.postprocess(img_cond.detach(), output_type="pil", do_denormalize=[True])[0]
            img_cond.save(os.path.join(sample_dir, "00_Fully_Conditional.png"))
            
            # (B) Fully Unconditional (Decoded from previous run)
            img_uncond = pipe.vae.decode(latents_basin / pipe.vae.config.scaling_factor, return_dict=False)[0]
            img_uncond = pipe.image_processor.postprocess(img_uncond.detach(), output_type="pil", do_denormalize=[True])[0]
            img_uncond.save(os.path.join(sample_dir, "07_Fully_Unconditional.png"))
            
            # (C) 6 Switched Trajectories
            for i, switch_step_idx in enumerate(checkpoints_indices):
                latents_switch = init_latents.clone()
                for step_idx, t in enumerate(timesteps):
                    if step_idx < switch_step_idx:
                        # Unconditional
                        model_input = pipe.scheduler.scale_model_input(latents_switch, t)
                        noise_pred = pipe.unet(model_input, t, encoder_hidden_states=uncond_prompt_embeds).sample
                        latents_switch = pipe.scheduler.step(noise_pred, t, latents_switch, return_dict=False)[0]
                    else:
                        # Conditional (CFG)
                        latent_model_input = torch.cat([latents_switch] * 2)
                        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
                        cat_embeds = torch.cat([uncond_prompt_embeds, cond_prompt_embeds])
                        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=cat_embeds).sample
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
                        latents_switch = pipe.scheduler.step(noise_pred, t, latents_switch, return_dict=False)[0]
                
                img_sw = pipe.vae.decode(latents_switch / pipe.vae.config.scaling_factor, return_dict=False)[0]
                img_sw = pipe.image_processor.postprocess(img_sw.detach(), output_type="pil", do_denormalize=[True])[0]
                img_sw.save(os.path.join(sample_dir, f"Switch_{i+1}_at_Step_{switch_step_idx:02d}.png"))
                
            # --- Plot ---
            plt.figure(figsize=(10, 6))
            plt.plot(basin_diff_curve, label='Basin Norm Diff', color='black')
            colors = plt.cm.rainbow(np.linspace(0, 1, len(checkpoints_indices)))
            for i, idx_pt in enumerate(checkpoints_indices):
                val = basin_diff_curve[idx_pt] if idx_pt < len(basin_diff_curve) else 0
                plt.plot(idx_pt, val, 'o', color=colors[i], markersize=10, label=f'Switch {idx_pt}')
                plt.axvline(idx_pt, color=colors[i], linestyle='--', alpha=0.5)
            plt.title(f"Attraction Basin & Switches\n{prompt[:40]}...")
            plt.xlabel("Step")
            plt.ylabel("Norm Diff")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(sample_dir, "attraction_basin_plot.png"))
            plt.close()
            
        # Clean up
        torch.cuda.empty_cache()

    print("Attraction Basin Test Complete.")

if __name__ == "__main__":
    run_attraction_basin_test()
