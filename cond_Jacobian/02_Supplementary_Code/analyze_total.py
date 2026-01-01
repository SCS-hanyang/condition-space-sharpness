
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, DDIMScheduler
from typing import List, Tuple
import argparse

# --- Configuration ---
MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 1 # Process one pair at a time for deep analysis

def load_prompts() -> Tuple[List[str], List[str]]:
    prompts_dir = "/home/gpuadmin/cssin/init_noise_diffusion_memorization/prompts"
    mem_path = os.path.join(prompts_dir, "memorized_laion_prompts.csv")
    unmem_path = os.path.join(prompts_dir, "unmemorized_prompts.csv")
    
    if not os.path.exists(mem_path) or not os.path.exists(unmem_path):
        print(f"Warning: Prompt files not found in {prompts_dir}. Using dummy prompts.")
        return ["A photo of a cat"] * 5, ["A drawing of a dog"] * 5
        
    df_mem = pd.read_csv(mem_path, sep=';')
    df_unmem = pd.read_csv(unmem_path, sep=';')
    
    return df_mem['Caption'].tolist()[:100], df_unmem['Caption'].tolist()[:100]

def compute_jacobian_norm(unet, latents, t, prompt_embeds, num_projections=1):
    prompt_embeds = prompt_embeds.detach()
    latents = latents.detach()
    
    with torch.enable_grad():
        prompt_embeds.requires_grad_(True)
        noise_pred = unet(latents, t, encoder_hidden_states=prompt_embeds).sample
        v = torch.randn_like(noise_pred)
        v_dot_eps = torch.sum(noise_pred * v)
        grads = torch.autograd.grad(v_dot_eps, prompt_embeds, create_graph=False)[0]
        norm = torch.norm(grads).item()
        
    return norm

class AttentionDataStore:
    def __init__(self):
        self.records = {}
        self.layer_names = []
        self.active = False
        
    def save_map(self, layer_name, attn_map):
        if layer_name not in self.records:
            self.records[layer_name] = []
            self.layer_names.append(layer_name)
        self.records[layer_name].append(attn_map.detach().cpu())
        
    def reset(self):
        self.records = {}
        self.layer_names = []
        self.active = False

class CaptureAttnMapProcessor:
    def __init__(self, store, layer_name):
        self.store = store
        self.layer_name = layer_name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale=1.0):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        if self.store.active:
            true_batch_size = hidden_states.shape[0]
            probs_reshaped = attention_probs.view(true_batch_size, -1, attention_probs.shape[1], attention_probs.shape[2])
            avg_probs = probs_reshaped.mean(dim=1)
            self.store.save_map(self.layer_name, avg_probs[0])

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

def setup_hooks(pipe, store):
    target_layers = []
    if hasattr(pipe.unet.mid_block, "attentions"):
        for i, _ in enumerate(pipe.unet.mid_block.attentions):
            name = f"mid_block.attentions.{i}.transformer_blocks.0.attn2"
            target_layers.append((pipe.unet.mid_block.attentions[i].transformer_blocks[0].attn2, name))
            
    for i, block in enumerate(pipe.unet.up_blocks):
        if hasattr(block, "attentions"):
            for j, _ in enumerate(block.attentions):
                name = f"up_blocks.{i}.attentions.{j}.transformer_blocks.0.attn2"
                target_layers.append((block.attentions[j].transformer_blocks[0].attn2, name))
                
    for module, name in target_layers:
        processor = CaptureAttnMapProcessor(store, name)
        module.set_processor(processor)
        print(f"Hooked {name}")

def analyze_total(generation_mode="uncond", num_inference_steps=500, base_results_dir="results/total_analysis_500"):
    """
    Main Analysis Function.
    """
    print(f"--- Analysis Total Started ---")
    print(f"Mode: {generation_mode}, Steps: {num_inference_steps}, OutDir: {base_results_dir}")
    
    # 1. Load Pipeline
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    
    # 2. Setup Hooks
    store = AttentionDataStore()
    setup_hooks(pipe, store)
    
    memorized_prompts, unmemorized_prompts = load_prompts()
    count = min(len(memorized_prompts), len(unmemorized_prompts), 100)
    
    uncond_prompt_embeds = pipe._encode_prompt("", DEVICE, 1, False, None)
    
    height = pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = pipe.unet.config.sample_size * pipe.vae_scale_factor

    os.makedirs(base_results_dir, exist_ok=True)

    print(f"Starting Analysis for {count} pairs...")

    for idx in range(count):
        print(f"[{idx+1}/{count}] Processing Pair {idx}...")
        sample_root = os.path.join(base_results_dir, f"sample_{idx:03d}")
        os.makedirs(sample_root, exist_ok=True)
        
        mem_prompt = memorized_prompts[idx]
        non_mem_prompt = unmemorized_prompts[idx]
        
        summary_metrics = {}
        
        modes_to_run = ["uncond", "cond"] if generation_mode == "both" else [generation_mode]

        # Loop Prompt Types
        for p_type, prompt in [("Memorization", mem_prompt), ("Non-memorization", non_mem_prompt)]:
            p_dir = os.path.join(sample_root, p_type)
            os.makedirs(p_dir, exist_ok=True)
            
            cond_embeds = pipe._encode_prompt(prompt, DEVICE, 1, False, None)
            
            gen = torch.Generator(DEVICE).manual_seed(SEED)
            latents = pipe.prepare_latents(
                1, pipe.unet.config.in_channels, height, width,
                cond_embeds.dtype, DEVICE, gen, None
            )
            init_latents = latents.clone()

            for run_mode in modes_to_run:
                run_suffix = f"_{run_mode.capitalize()}" if generation_mode == "both" else ""
                
                print(f"  > Generating {p_type} in {run_mode} mode...")
                store.reset()
                pipe.scheduler.set_timesteps(num_inference_steps, device=DEVICE)
                timesteps = pipe.scheduler.timesteps
                
                curr_latents = init_latents.clone()
                
                basin_curve = []
                jacobian_curve = []
                
                for t in timesteps:
                    store.active = False
                    scaled_latents = pipe.scheduler.scale_model_input(curr_latents, t)
                    j_norm = compute_jacobian_norm(pipe.unet, scaled_latents, t, cond_embeds, num_projections=1)
                    jacobian_curve.append(j_norm)
                    
                    with torch.no_grad():
                        store.active = True
                        noise_cond = pipe.unet(pipe.scheduler.scale_model_input(curr_latents, t), t, encoder_hidden_states=cond_embeds).sample
                        store.active = False
                        
                        noise_uncond = pipe.unet(pipe.scheduler.scale_model_input(curr_latents, t), t, encoder_hidden_states=uncond_prompt_embeds).sample
                        
                        diff = torch.norm(noise_cond - noise_uncond).item()
                        basin_curve.append(diff)
                        
                        step_noise = noise_uncond if run_mode == "uncond" else noise_cond
                        curr_latents = pipe.scheduler.step(step_noise, t, curr_latents, return_dict=False)[0]

                # --- Attention Stability ---
                layer_stabilities = []
                if store.layer_names:
                    for name in store.layer_names:
                        maps = torch.stack(store.records[name])
                        diffs = torch.norm(maps[1:] - maps[:-1], dim=(1,2))
                        layer_stabilities.append(diffs)
                    stack_stab = torch.stack(layer_stabilities)
                    avg_stability = stack_stab.mean(dim=0).tolist()
                    avg_stability.insert(0, 0.0)
                else:
                    avg_stability = [0.0]*len(basin_curve)

                # --- Save Metrics ---
                def fix_len(lst, target=num_inference_steps):
                    if len(lst) > target: return lst[:target]
                    while len(lst) < target: lst.append(0)
                    return lst

                metric_prefix = p_type + run_suffix
                
                summary_metrics[f"{metric_prefix}_Basin"] = fix_len(basin_curve)
                summary_metrics[f"{metric_prefix}_J_TE"] = fix_len(jacobian_curve)
                summary_metrics[f"{metric_prefix}_AttnStab"] = fix_len(avg_stability)
                
                # Plot Dynamics
                fig, ax1 = plt.subplots(figsize=(10, 6))
                color = 'black'
                ax1.set_ylabel(r"Norm Diff", color=color)
                ax1.plot(basin_curve, color=color, linestyle='-', label=r"Basin Diff")
                ax1.tick_params(axis='y', labelcolor=color)

                ax1_twin = ax1.twinx()
                color = 'red'
                ax1_twin.set_ylabel(r"$||J_{TE}||_F$", color=color)
                ax1_twin.plot(jacobian_curve, color=color, linestyle='-', label=r"$J_{TE}$")
                ax1_twin.tick_params(axis='y', labelcolor=color)
                
                ax1.set_title(f"Dynamics: {p_type} ({run_mode})")
                plt.savefig(os.path.join(p_dir, f"dynamics_{run_mode}.png"))
                plt.close()

                # Save Final Image
                img = pipe.image_processor.postprocess(pipe.vae.decode(curr_latents/pipe.vae.config.scaling_factor, return_dict=False)[0].detach(), output_type="pil", do_denormalize=[True])[0]
                img.save(os.path.join(p_dir, f"final_{run_mode}.png"))

            torch.cuda.empty_cache()
            
        df_pair = pd.DataFrame(summary_metrics)
        df_pair.to_csv(os.path.join(sample_root, "metrics.csv"), index=False)

    print("Analysis Total Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="uncond", choices=["uncond", "cond", "both"], help="Generation trajectory mode")
    parser.add_argument("--steps", type=int, default=500, help="Number of inference steps")
    parser.add_argument("--outdir", type=str, default="results/total_analysis_500", help="Base output directory")
    args = parser.parse_args()
    
    analyze_total(generation_mode=args.mode, num_inference_steps=args.steps, base_results_dir=args.outdir)
