
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import Attention
import seaborn as sns

# Reuse the processor logic
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
        
        # Standard Attention Logic
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

        attention_scores = attn.scale * torch.bmm(query, key.transpose(-1, -2))
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = attention_scores.softmax(dim=-1)
        
        # --- Capture Logic ---
        true_batch_size = hidden_states.shape[0]
        
        if true_batch_size == 2:
            probs_reshaped = attention_probs.view(true_batch_size, -1, attention_probs.shape[1], attention_probs.shape[2])
            cond_probs = probs_reshaped[1] # (Heads, Q, K)
        else:
            cond_probs = attention_probs.view(true_batch_size, -1, attention_probs.shape[1], attention_probs.shape[2])[0]

        # Average over heads
        avg_probs = cond_probs.mean(dim=0) 
        
        # Spatial Mean: (Tokens,)
        token_scores = avg_probs.mean(dim=0)
        
        self.store.capture(self.layer_name, token_scores)
        # ---------------------

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class AttentionStatsStore:
    def __init__(self):
        self.data = {} 
        self.current_step = 0
        self.layer_names = []
        
    def capture(self, layer_name, token_scores):
        if self.current_step not in self.data:
            self.data[self.current_step] = {}
        
        self.data[self.current_step][layer_name] = token_scores.detach().cpu()
        if layer_name not in self.layer_names:
            self.layer_names.append(layer_name)
        
    def step(self):
        self.current_step += 1
        
    def reset(self):
        self.data = {}
        self.current_step = 0
        self.layer_names = [] # Reset order if needed, but usually consistent
        
    def get_layer_scores(self, step_index, layer_name):
        if step_index not in self.data:
            return None
        return self.data[step_index].get(layer_name, None)

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

def analyze_and_plot():
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "CompVis/stable-diffusion-v1-4"
    results_dir = "init_noise_diffusion_memorization/results/memorization_on_CA/first_token_analysis"
    os.makedirs(results_dir, exist_ok=True)
    
    print("Loading SD Pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    pipe.set_progress_bar_config(disable=True)
    
    mem_prompt = "Mothers influence on her young hippo"
    non_mem_prompt = "A futuristic city with flying cars and neon lights"
    
    prompts = {
        "Memorization": mem_prompt,
        "Non-memorization": non_mem_prompt
    }
    
    store = AttentionStatsStore()
    setup_attention_hooks(pipe, store)
    
    num_inference_steps = 50
    target_steps = {"T": 0, "T_div_4": 37}
    
    collected_data = []

    # 2. Inference
    for p_type, prompt in prompts.items():
        print(f"Processing {p_type}...")
        store.reset()
        
        def callback(step, t, latents):
            store.step()
            
        generator = torch.Generator(device).manual_seed(42)
        pipe(prompt, num_inference_steps=num_inference_steps, generator=generator, callback=callback, callback_steps=1)
        
        for label, step_idx in target_steps.items():
            for lname in store.layer_names:
                sc = store.get_layer_scores(step_idx, lname) # Tensor (77,)
                if sc is not None:
                    # Calculate Stats
                    s0 = sc[0].item()
                    s_rest = sc[1:].sum().item()
                    
                    collected_data.append({
                        "Type": p_type,
                        "TimeStep": label,
                        "Layer": lname,
                        "FirstToken": s0,
                        "RestTokens": s_rest
                    })

    # 3. Create DataFrame
    df = pd.DataFrame(collected_data)
    
    # Save raw data
    df.to_csv(os.path.join(results_dir, "first_token_analysis.csv"), index=False)
    
    # 4. Plotting
    # Maintain layer order from UNet forward pass
    # store.layer_names contains the order of execution for the last run
    layer_order = store.layer_names
    
    # Map layers to indices 0..15 for plotting
    layer_indices = {name: i for i, name in enumerate(layer_order)}
    df['LayerIndex'] = df['Layer'].map(layer_indices)
    df.sort_values('LayerIndex', inplace=True)
    
    # Generate Plot
    # 2 Subplots: T and T/4
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    for i, (label, title_suffix) in enumerate([("T", "Step 0"), ("T_div_4", "Step 37")]):
        ax = axes[i]
        subset = df[df['TimeStep'] == label]
        
        # Mem Data
        mem = subset[subset['Type'] == 'Memorization']
        # NonMem Data
        non_mem = subset[subset['Type'] == 'Non-memorization']
        
        layers_x = mem['LayerIndex'].values
        
        # Plot Lines
        ax.plot(layers_x, mem['FirstToken'], marker='o', color='red', label='Mem: First Token ($S_0$)', linewidth=2)
        ax.plot(layers_x, mem['RestTokens'], marker='x', color='salmon', linestyle='--', label='Mem: Rest ($S_{rest}$)', linewidth=2)
        
        ax.plot(layers_x, non_mem['FirstToken'], marker='o', color='blue', label='Non-Mem: First Token ($S_0$)', linewidth=2)
        ax.plot(layers_x, non_mem['RestTokens'], marker='x', color='skyblue', linestyle='--', label='Non-Mem: Rest ($S_{rest}$)', linewidth=2)
        
        ax.set_title(f"First Token vs Rest Attention Sum ({title_suffix})", fontsize=16)
        ax.set_ylabel("Sum of Attention Scores", fontsize=14)
        ax.set_xlabel("UNet Layer Index (Default Execution Order)", fontsize=14)
        
        ax.set_xticks(layers_x)
        # Shorten layer names for X labels
        short_names = [n.split('.')[-4] + '.' + n.split('.')[-1] for n in mem['Layer'].values] # heuristic shortening
        # Better heuristic: block type + index
        def simplify_chem(n):
            parts = n.split('.')
            return f"{parts[0]}.{parts[1]}" # e.g., down_blocks.0
            
        ax.set_xticklabels([simplify_chem(n) for n in mem['Layer'].values], rotation=45, ha='right')
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "first_token_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    analyze_and_plot()
