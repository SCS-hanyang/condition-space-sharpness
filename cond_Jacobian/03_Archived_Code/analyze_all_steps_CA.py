
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import Attention
import seaborn as sns
import shutil

# --- 1. Capture Logic ---

class CaptureAttnProcessor:
    def __init__(self, store, layer_name):
        self.store = store
        self.layer_name = layer_name

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        temb: torch.Tensor = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        
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

        # Attention Scores
        attention_scores = attn.scale * torch.bmm(query, key.transpose(-1, -2))
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = attention_scores.softmax(dim=-1)
        
        # Capture
        true_batch_size = hidden_states.shape[0]
        if true_batch_size == 2:
            probs_reshaped = attention_probs.view(true_batch_size, -1, attention_probs.shape[1], attention_probs.shape[2])
            cond_probs = probs_reshaped[1] 
        else:
            cond_probs = attention_probs.view(true_batch_size, -1, attention_probs.shape[1], attention_probs.shape[2])[0]

        # Average over heads -> (Spatial, Tokens)
        avg_probs = cond_probs.mean(dim=0)
        
        # Store raw map (Process later to save VRAM/Compute during inference)
        # We need to minimally process to save memory if maps are huge?
        # Spatial dims can be large (64x64=4096). 4096*77 floats is small.
        # But we want to resize to 768x77 eventually. Doing it here might be slow.
        # Let's just store (Spatial, Tokens) on CPU.
        self.store.capture(self.layer_name, avg_probs)
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

class AttentionStore:
    def __init__(self):
        # {step: {layer_name: tensor(Spatial, Tokens)}}
        self.data = {} 
        self.current_step = 0
        self.layer_names = []
        
    def capture(self, layer_name, attn_map):
        if self.current_step not in self.data:
            self.data[self.current_step] = {}
        
        self.data[self.current_step][layer_name] = attn_map.detach().cpu()
        if layer_name not in self.layer_names:
            self.layer_names.append(layer_name)
        
    def step(self):
        self.current_step += 1
        
    def reset(self):
        self.data = {}
        self.current_step = 0
        
    def get_layer_map(self, step, layer_name):
        if step in self.data:
            return self.data[step].get(layer_name, None)
        return None

def setup_attention_hooks(pipe, store):
    def register_recursive(module, name_prefix):
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            is_cross = 'attn2' in name and isinstance(child, Attention)
            if is_cross:
                processor = CaptureAttnProcessor(store, full_name)
                child.set_processor(processor)
            else:
                register_recursive(child, full_name)
    register_recursive(pipe.unet, "")
    return store

# --- 2. Helper Functions ---
def resize_map(attn_map, target_dim=768):
    # attn_map: (Spatial, Tokens)
    # Resize Spatial -> target_dim
    T = attn_map.shape[1]
    reshaped = attn_map.t().unsqueeze(0) # (1, T, Spatial)
    # Linear interpolate
    resized = F.interpolate(reshaped, size=target_dim, mode='linear', align_corners=False)
    # (1, T, target_dim) -> (target_dim, T)
    final_map = resized.squeeze(0).t()
    return final_map

# --- 3. Main Execution ---

def run_full_analysis():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "CompVis/stable-diffusion-v1-4"
    base_results_dir = "init_noise_diffusion_memorization/results/memorization_on_CA/all_steps"
    
    # Clean up previous run if exists to avoid confusion
    if os.path.exists(base_results_dir):
        shutil.rmtree(base_results_dir)
    os.makedirs(base_results_dir, exist_ok=True)
    
    print("Loading SD Pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    pipe.set_progress_bar_config(disable=True)
    
    mem_prompt = "Mothers influence on her young hippo"
    non_mem_prompt = "A futuristic city with flying cars and neon lights"
    
    store = AttentionStore()
    setup_attention_hooks(pipe, store)
    
    num_inference_steps = 50
    
    # Store results for both prompts
    # Structure: Prompts -> Steps -> Layers
    all_results = {} 
    
    for p_type, prompt in [("Memorization", mem_prompt), ("Non-memorization", non_mem_prompt)]:
        print(f"Running Inference: {p_type}")
        store.reset()
        
        def callback(step, t, latents):
            store.step()
            
        generator = torch.Generator(device).manual_seed(42)
        pipe(prompt, num_inference_steps=num_inference_steps, generator=generator, callback=callback, callback_steps=1)
        
        # Save reference
        all_results[p_type] = store.data # Copy ref
        # Store layer order
        layer_order = store.layer_names

    print("Inference Complete. Generating Plots per Step...")
    
    # Iterate Steps
    for step_idx in range(num_inference_steps):
        step_dir = os.path.join(base_results_dir, f"step_{step_idx:02d}")
        os.makedirs(step_dir, exist_ok=True)
        print(f"Processing Step {step_idx}/{num_inference_steps}...")
        
        # --- A. Token Comparison Logic ---
        # Collect stats for this step
        stats_data = []
        
        for lname in layer_order:
            for p_type in ["Memorization", "Non-memorization"]:
                raw_map = all_results[p_type][step_idx].get(lname)
                if raw_map is not None:
                    # Token Scores: Average over spatial dim
                    token_scores = raw_map.mean(dim=0) # (77,)
                    s0 = token_scores[0].item()
                    s_rest = token_scores[1:].sum().item()
                    
                    stats_data.append({
                        "Type": p_type,
                        "Layer": lname,
                        "FirstToken": s0,
                        "RestTokens": s_rest
                    })
        
        # Save Stats CSV
        df_step = pd.DataFrame(stats_data)
        df_step.to_csv(os.path.join(step_dir, f"token_stats_step_{step_idx}.csv"), index=False)
        
        # Plot Comparison
        # Map layers to indices
        df_step['LayerIndex'] = df_step['Layer'].map({name: i for i, name in enumerate(layer_order)})
        df_step.sort_values('LayerIndex', inplace=True)
        
        plt.figure(figsize=(12, 6))
        mem_df = df_step[df_step['Type'] == 'Memorization']
        non_df = df_step[df_step['Type'] == 'Non-memorization']
        
        x_vals = mem_df['LayerIndex'].values
        plt.plot(x_vals, mem_df['FirstToken'], 'r-o', label='Mem: First', linewidth=2)
        plt.plot(x_vals, mem_df['RestTokens'], 'r--x', label='Mem: Rest', linewidth=2)
        plt.plot(x_vals, non_df['FirstToken'], 'b-o', label='Non-Mem: First', linewidth=2)
        plt.plot(x_vals, non_df['RestTokens'], 'b--x', label='Non-Mem: Rest', linewidth=2)
        
        plt.title(f"Comparison: First Token vs Rest (Step {step_idx})")
        plt.xlabel("Layer Index")
        plt.ylabel("Attention Score Sum")
        
        short_ticks = [n.split('.')[0] + '.' + n.split('.')[1] for n in mem_df['Layer']]
        plt.xticks(x_vals, short_ticks, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(step_dir, f"comparison_plot_step_{step_idx}.png"), dpi=150)
        plt.close()
        
        # --- B. Heatmaps Logic ---
        # Generate side-by-side heatmaps for each layer
        heatmap_dir = os.path.join(step_dir, "heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)
        
        for lname in layer_order:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            for i, p_type in enumerate(["Memorization", "Non-memorization"]):
                raw_map = all_results[p_type][step_idx].get(lname)
                if raw_map is not None:
                    # Resize to (768, 77)
                    final_map = resize_map(raw_map, target_dim=768)
                    
                    sns.heatmap(final_map.numpy(), ax=axes[i], cmap="YlGnBu", cbar=True, vmin=0, vmax=1.0)
                    axes[i].set_title(f"{p_type}", fontsize=14)
                    axes[i].set_xlabel("Token Index")
                    if i == 0:
                        axes[i].set_ylabel("Image Representation (0-768)")
                    else:
                        axes[i].set_ylabel("")
                        axes[i].set_yticks([])
            
            plt.suptitle(f"Layer: {lname} (Step {step_idx})", fontsize=16)
            plt.tight_layout()
            safe_lname = lname.replace(".", "_")
            plt.savefig(os.path.join(heatmap_dir, f"{safe_lname}.png"), dpi=100) # Lower dpi for speed/storage
            plt.close()

    print(f"All steps completed. Results saved in {base_results_dir}")

if __name__ == "__main__":
    run_full_analysis()
