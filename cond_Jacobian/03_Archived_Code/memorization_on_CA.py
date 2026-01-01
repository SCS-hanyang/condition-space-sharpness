
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import Attention
import seaborn as sns

# Define the custom Attention Processor
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

        # Compute Attention Weights
        attention_scores = attn.scale * torch.bmm(query, key.transpose(-1, -2))
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = attention_scores.softmax(dim=-1)
        
        # --- Capture Logic ---
        true_batch_size = hidden_states.shape[0]
        
        if true_batch_size == 2:
            # Split into [Uncond, Cond]
            probs_reshaped = attention_probs.view(true_batch_size, -1, attention_probs.shape[1], attention_probs.shape[2])
            cond_probs = probs_reshaped[1] # (Heads, Q, K)
        else:
            cond_probs = attention_probs.view(true_batch_size, -1, attention_probs.shape[1], attention_probs.shape[2])[0]

        # Average over heads
        avg_probs = cond_probs.mean(dim=0) # (Q, K)
        
        # Store in our collector
        self.store.capture(self.layer_name, avg_probs)
        # ---------------------

        # Continue computation
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class AttentionStore:
    def __init__(self, target_dim=768):
        # Storage: {step_index: {layer_name: map_tensor}}
        self.data = {} 
        self.current_step = 0
        self.target_dim = target_dim # Target length for Y-axis (Image Rep Dim)
        self.layer_names = []
        
    def capture(self, layer_name, attn_map):
        # attn_map: (HW, T)
        # We want to reshape/interpolate HW to target_dim (768)
        
        T = attn_map.shape[1]
        
        # Transpose to (T, HW) -> Treat T as Batch/Channel, HW as Length for interpolation
        # F.interpolate 1d expects (N, C, L). We want to resize L.
        # Let's map: N=1, C=T, L=HW
        
        reshaped = attn_map.t().unsqueeze(0) # (1, T, HW)
        
        # Interpolate 1D
        # Use linear mode for 1D
        resized = F.interpolate(reshaped, size=self.target_dim, mode='linear', align_corners=False)
        
        # Result: (1, T, Target_Dim)
        # We want (Target_Dim, T) for plotting (Y=Dim, X=Tokens)
        final_map = resized.squeeze(0).t() # (Target_Dim, T)
        
        if self.current_step not in self.data:
            self.data[self.current_step] = {}
            
        self.data[self.current_step][layer_name] = final_map.detach().cpu()
        
        if layer_name not in self.layer_names:
            self.layer_names.append(layer_name)
        
    def step(self):
        self.current_step += 1
        
    def reset(self):
        self.data = {}
        self.current_step = 0
        
    def get_layer_map(self, step_index, layer_name):
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
    print("Attention hooks registered.")

def run_visualization():
    # 1. Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "CompVis/stable-diffusion-v1-4"
    results_dir = "init_noise_diffusion_memorization/results/memorization_on_CA/per_layer"
    os.makedirs(results_dir, exist_ok=True)
    
    # 2. Load Model
    print("Loading SD Pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    pipe.set_progress_bar_config(disable=True)
    
    # 3. Select Prompts
    mem_prompt = "Mothers influence on her young hippo"
    non_mem_prompt = "A futuristic city with flying cars and neon lights"
    
    prompts = {
        "Memorization": mem_prompt,
        "Non-memorization": non_mem_prompt
    }
    
    store = AttentionStore(target_dim=768)
    setup_attention_hooks(pipe, store)
    
    # 4. Run Inference
    num_inference_steps = 50
    target_steps = {
        "T": 0,
        "T_div_4": 37 
    } 
    
    saved_data = []
    
    for p_type, prompt in prompts.items():
        print(f"Processing {p_type}: {prompt}")
        store.reset()
        
        def callback(step, timestep, latents):
            store.step()
            
        generator = torch.Generator(device).manual_seed(42)
        
        pipe(
            prompt, 
            num_inference_steps=num_inference_steps, 
            generator=generator,
            callback=callback, 
            callback_steps=1
        )
        
        for label, step_idx in target_steps.items():
            layer_maps = {}
            for lname in store.layer_names:
                m = store.get_layer_map(step_idx, lname)
                if m is not None:
                    layer_maps[lname] = m
            
            saved_data.append({
                "type": p_type,
                "label": label,
                "maps": layer_maps,
                "prompt": prompt
            })

    # 5. Plotting per Layer
    all_layers = store.layer_names
    print(f"Found {len(all_layers)} attention layers.")
    
    order = [
        ("Memorization", "T"),
        ("Memorization", "T_div_4"),
        ("Non-memorization", "T"),
        ("Non-memorization", "T_div_4")
    ]
    
    for layer_name in all_layers:
        print(f"Plotting layer: {layer_name}")
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        
        for i, (ptype, plabel) in enumerate(order):
            ax = axes[i]
            
            data_item = next((d for d in saved_data if d["type"] == ptype and d["label"] == plabel), None)
            
            if data_item and layer_name in data_item["maps"]:
                attn_map = data_item["maps"][layer_name].numpy() # (768, 77)
                
                # Plot full token range (77)
                sns.heatmap(attn_map, ax=ax, cmap="YlGnBu", cbar=True, vmin=0, vmax=1.0)
                
                t_val = "T" if plabel == "T" else "T/4"
                ax.set_title(f"({chr(97+i)}) {ptype}\n($t = {t_val}$)", fontsize=16)
                
                # Setup axes
                ax.set_xlabel("Token Index", fontsize=12)
                
                # We show major ticks 
                ax.set_xticks(np.arange(0, 77, 10))
                ax.set_xticklabels(np.arange(0, 77, 10))
                
                if i == 0:
                    ax.set_ylabel("Image Representation Dimension (0-768)", fontsize=12)
                    # Set Y ticks to show 0 to 768
                    ax.set_yticks(np.arange(0, 768, 100))
                    ax.set_yticklabels(np.arange(0, 768, 100))
                else:
                    ax.set_ylabel("")
                    ax.set_yticks([])
            else:
                ax.text(0.5, 0.5, "Data Missing", ha='center')
        
        safe_name = layer_name.replace(".", "_")
        plt.suptitle(f"Layer: {layer_name}", fontsize=20, y=1.05)
        plt.tight_layout()
        save_path = os.path.join(results_dir, f"{safe_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {len(all_layers)} plots to {results_dir}")

if __name__ == "__main__":
    run_visualization()
