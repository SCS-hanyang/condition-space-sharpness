import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from diffusers import DDIMScheduler
from diffusers.models.attention_processor import Attention

# Try absolute import for local pipeline
import sys
# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from local_sd_pipeline import LocalStableDiffusionPipeline
except ImportError:
    print("Warning: LocalStableDiffusionPipeline not found, falling back to standard.")
    from diffusers import StableDiffusionPipeline as LocalStableDiffusionPipeline

# --- 1. Helper Classes & Functions ---

def compute_jacobian_norm_batched(unet, latents, t, prompt_embeds, num_projections=1):
    """
    Batched Hutchinson estimator for ||J||_F.
    Returns a tensor of shape (batch_size,) containing the Jacobian norm for each sample.
    """
    # latents: [B, 4, 64, 64]
    # prompt_embeds: [B, Seq, Dim]
    
    prompt_embeds = prompt_embeds.detach()
    latents = latents.detach()
    
    batch_size = latents.shape[0]
    
    with torch.enable_grad():
        prompt_embeds.requires_grad_(True)
        # Forward pass
        # Output noise_pred: [B, 4, 64, 64]
        noise_pred = unet(latents, t, encoder_hidden_states=prompt_embeds).sample
        
        sq_norm_sum = torch.zeros(batch_size, device=latents.device)
        
        for k in range(num_projections):
            # Random probe v: [B, 4, 64, 64]
            v = torch.randn_like(noise_pred)
            
            # Dot product sum. We rely on autograd to handle per-sample independence if structure allows.
            # But sum(v_dot_eps) -> sum output.
            # grad(sum, inputs) -> if inputs are batched [B, ...], grad is [B, ...].
            # Elements of grad[i] only depend on loss components related to inputs[i].
            v_dot_eps_sum = torch.sum(noise_pred * v)
            
            retain = (k < num_projections - 1)
            grads = torch.autograd.grad(v_dot_eps_sum, prompt_embeds, retain_graph=retain, create_graph=False)[0]
            
            # Squared norm per sample
            grads_sq_flat = grads.view(batch_size, -1).pow(2)
            sample_sq_norms = grads_sq_flat.sum(dim=1) # [B]
            
            sq_norm_sum += sample_sq_norms.detach()
            
        est_frob_sq = sq_norm_sum / max(1, float(num_projections))
        return est_frob_sq.sqrt() # [B]

class AttentionDataStore:
    def __init__(self):
        self.records = {} # {layer_name: [map_batch]}
        self.active = False
        
    def save_map(self, layer_name, attn_map):
        if layer_name not in self.records:
            self.records[layer_name] = []
        self.records[layer_name].append(attn_map)
        
    def reset(self):
        self.records = {}
        self.active = False

class CaptureAttnMapProcessor:
    def __init__(self, store, layer_name):
        self.store = store
        self.layer_name = layer_name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0):
        # hidden_states: [Batch * Heads, Seq, Dim]
        
        # Depending on diffusers version, signature might vary. 
        # Typically hidden_states: [Batch*Heads, Seq, Dim]
        
        batch_size_heads, sequence_length, _ = hidden_states.shape
        batch_size = batch_size_heads // attn.heads
        
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

        # Standard Attention
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        if self.store.active:
            # Reshape to [Batch, Heads, Q, K]
            probs_reshaped = attention_probs.view(batch_size, attn.heads, attention_probs.shape[1], attention_probs.shape[2])
            # Mean over Heads -> [Batch, Q, K]
            avg_probs = probs_reshaped.mean(dim=1)
            # Save batch to store. Keep on Device.
            self.store.save_map(self.layer_name, avg_probs)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

def setup_hooks(pipe, store):
    def register_recursive(module, name_prefix):
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            if 'attn2' in name and isinstance(child, Attention):
                processor = CaptureAttnMapProcessor(store, full_name)
                child.set_processor(processor)
            else:
                register_recursive(child, full_name)
    register_recursive(pipe.unet, "")
    
def load_prompts():
    prompts_dir = "/home/gpuadmin/cssin/init_noise_diffusion_memorization/prompts"
    mem_path = os.path.join(prompts_dir, "memorized_laion_prompts.csv")
    unmem_path = os.path.join(prompts_dir, "unmemorized_prompts.csv")
    
    if not os.path.exists(mem_path) or not os.path.exists(unmem_path):
        print("Warning: CSV files not found. Using dummies.")
        return ["cat"]*5, ["dog"]*5

    df_mem = pd.read_csv(mem_path, sep=';')
    df_unmem = pd.read_csv(unmem_path, sep=';')
    
    return df_mem['Caption'].tolist()[:50], df_unmem['Caption'].tolist()[:5]

def analyze_pure_noise(args, pipeline, device):
    mem_prompts, unmem_prompts = load_prompts()
    
    uncond_prompt = [""]
    # Single Uncond Embed: [1, Seq, Dim]
    uncond_embeds_single = pipeline._encode_prompt(uncond_prompt, device, 1, False, None)
                                            
    pipeline.scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    
    # Seeds
    num_samples = args.num_samples
    ############################################################################
    # OOM Fix: Reduced batch_size from 50 to 10.
    ############################################################################
    batch_size = 10 
    
    # 1. Generate All Init Noises
    print("Generating Init Noises...")
    all_init_noises = []
    for seed in range(num_samples):
        gen = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(
            (1, pipeline.unet.config.in_channels, 64, 64),
            generator=gen,
            device=device,
            dtype=uncond_embeds_single.dtype
        )
        all_init_noises.append(noise)
    all_init_noises = torch.cat(all_init_noises, dim=0)

    # 2. Main Analysis (Batched, On-the-fly Uncond)
    groups = [("Memorized", mem_prompts), ("Unmemorized", unmem_prompts)]
    store = AttentionDataStore()
    setup_hooks(pipeline, store)

    output_dir = "./results/pure_noise_basin_analysis"
    os.makedirs(output_dir, exist_ok=True)
    # all_results = [] # Removed
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for group_name, prompts in groups:
        for p_idx, prompt in enumerate(tqdm(prompts, desc=f"Processing {group_name}")):
            
            cond_embeds_single = pipeline._encode_prompt(prompt, device, 1, False, None)
            
            # Buffers
            prompt_diffs = torch.zeros((num_samples, len(timesteps)), device="cpu")
            prompt_jacob = torch.zeros((num_samples, len(timesteps)), device="cpu")
            prompt_attn = torch.zeros((num_samples, len(timesteps)), device="cpu")
            
            # Storage for prev maps: layer_name -> Tensor[Num_Samples, Q, K] (on GPU)
            prev_maps_storage = {} 
            
            for t_idx, t in enumerate(tqdm(timesteps, desc="Sampling", leave=False)):
                
                for b in range(num_batches):
                    start = b * batch_size
                    end = min(start + batch_size, num_samples)
                    if start >= end: break
                    current_bs = end - start
                    
                    batch_noises = all_init_noises[start:end] 
                    
                    # Prepare Embeddings [B, Seq, Dim]
                    batch_uncond_embeds = uncond_embeds_single.repeat(current_bs, 1, 1)
                    batch_cond_embeds = cond_embeds_single.repeat(current_bs, 1, 1)
                    
                    latent_input = pipeline.scheduler.scale_model_input(batch_noises, t)
                    
                    # A. Jacobian
                    if args.compute_jacobian:
                        j_norms = compute_jacobian_norm_batched(pipeline.unet, latent_input, t, batch_cond_embeds)
                        prompt_jacob[start:end, t_idx] = j_norms.cpu()
                    
                    # B. Conditional Forward + Attn
                    store.active = True
                    store.reset() # clear prev batch data
                    
                    with torch.no_grad():
                        noise_pred_cond = pipeline.unet(latent_input, t, encoder_hidden_states=batch_cond_embeds).sample
                    store.active = False
                    
                    # C. Unconditional Forward (On-the-fly)
                    with torch.no_grad():
                        noise_pred_uncond = pipeline.unet(latent_input, t, encoder_hidden_states=batch_uncond_embeds).sample
                    
                    # D. Diff
                    diffs = (noise_pred_cond - noise_pred_uncond).view(current_bs, -1).norm(dim=1)
                    prompt_diffs[start:end, t_idx] = diffs.cpu()
                    
                    # E. Attention Stability
                    batch_attn_diffs = torch.zeros(current_bs, device=device)
                    layer_count = 0
                    
                    for layer_name, maps_list in store.records.items():
                        if not maps_list: continue
                        curr_batch_map = maps_list[-1] # [B, Q, K]
                        
                        # Init storage if needed
                        if layer_name not in prev_maps_storage:
                            prev_maps_storage[layer_name] = torch.zeros(
                                (num_samples, curr_batch_map.shape[1], curr_batch_map.shape[2]),
                                device=device, dtype=curr_batch_map.dtype
                            )
                        
                        if t_idx > 0:
                            prev_batch_map = prev_maps_storage[layer_name][start:end]
                            d = (curr_batch_map - prev_batch_map).view(current_bs, -1).norm(dim=1)
                            batch_attn_diffs += d
                        
                        prev_maps_storage[layer_name][start:end] = curr_batch_map.clone()
                        layer_count += 1
                        
                    if layer_count > 0 and t_idx > 0:
                        batch_attn_diffs /= layer_count
                        prompt_attn[start:end, t_idx] = batch_attn_diffs.cpu()
                    else:
                        prompt_attn[start:end, t_idx] = 0.0

            # Collect Results per Prompt
            prompt_results = []
            for s_idx in range(num_samples):
                for t_i, t_val in enumerate(timesteps):
                    prompt_results.append({
                        "Group": group_name,
                        "Prompt_Idx": p_idx,
                        "Sample_Idx": s_idx,
                        "Step": t_i,
                        "Timestep": t_val.item(),
                        "Diff": prompt_diffs[s_idx, t_i].item(),
                        "Jacobian": prompt_jacob[s_idx, t_i].item(),
                        "AttnStability": prompt_attn[s_idx, t_i].item()
                    })
            
            # Save & Plot per Prompt
            prompt_dir = os.path.join(output_dir, f"{group_name}_{p_idx}")
            os.makedirs(prompt_dir, exist_ok=True)
            
            df_prompt = pd.DataFrame(prompt_results)
            save_path = os.path.join(prompt_dir, "metrics.csv")
            df_prompt.to_csv(save_path, index=False)
            
            # Save caption text for reference
            with open(os.path.join(prompt_dir, "caption.txt"), "w") as f:
                f.write(prompt)
                
            plot_results(df_prompt, prompt_dir)
            
            # Cleanup
            del prompt_diffs, prompt_jacob, prompt_attn, df_prompt, prompt_results
            del prev_maps_storage
            torch.cuda.empty_cache()

    return None, output_dir

def plot_results(df, output_dir):
    print("Plotting...")
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y="Diff", hue="Group")
    plt.title("Average Norm Diff (Cond Result - Uncond Result) over Time")
    plt.savefig(os.path.join(output_dir, "avg_diff_over_time.png"))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y="Jacobian", hue="Group")
    plt.title("Average Jacobian Norm over Time")
    plt.savefig(os.path.join(output_dir, "avg_jacobian_over_time.png"))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y="AttnStability", hue="Group")
    plt.title("Average Attention Stability over Time")
    plt.savefig(os.path.join(output_dir, "avg_attn_stability_over_time.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Diff", y="Jacobian", hue="Group", alpha=0.3)
    plt.title("Diff vs Jacobian Norm")
    plt.savefig(os.path.join(output_dir, "scatter_diff_vs_jacobian.png"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_inference_steps", type=int, default=500, help="Number of denoising steps")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples per prompt")
    parser.add_argument("--compute_jacobian", action="store_true", default=True, help="Compute Jacobian Norm")
    
    args = parser.parse_args()
    
    model_id = "CompVis/stable-diffusion-v1-4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        pipeline = LocalStableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    except Exception as e:
        print(f"Failed to load local pipeline: {e}")
        pipeline = LocalStableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
        
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    _, out_dir = analyze_pure_noise(args, pipeline, device)
    # plot_results(df, out_dir)
