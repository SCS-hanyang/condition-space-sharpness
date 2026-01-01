
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import Attention
import seaborn as sns

# Reuse the processor logic but simplified for statistics
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

        # Average over heads -> (Q, K) = (Spatial, Tokens)
        avg_probs = cond_probs.mean(dim=0) 
        
        # Calculate Spatial Mean immediately: (Tokens,)
        # sums to 1 across tokens
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
        # {step: {layer: tensor(77,)}}
        self.data = {} 
        self.current_step = 0
        self.layer_names = []
        
    def capture(self, layer_name, token_scores):
        if self.current_step not in self.data:
            self.data[self.current_step] = {}
        
        # If we have multiple calls per step (unlikely for standard CFG loops unless deeper recursion?), just overwrite or mean?
        # Usually called once per layer per step.
        self.data[self.current_step][layer_name] = token_scores.detach().cpu()
        
        if layer_name not in self.layer_names:
            self.layer_names.append(layer_name)
        
    def step(self):
        self.current_step += 1
        
    def reset(self):
        self.data = {}
        self.current_step = 0
        
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
    results_dir = "init_noise_diffusion_memorization/results/memorization_on_CA/distribution_analysis"
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
    
    # Data storage: [ {type, label, scores: {layer: tensor}} ]
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
            layer_scores = {}
            for lname in store.layer_names:
                sc = store.get_layer_scores(step_idx, lname)
                if sc is not None:
                    layer_scores[lname] = sc
            
            collected_data.append({
                "type": p_type,
                "label": label,
                "scores": layer_scores
            })

    # 3. Analysis & Plotting
    all_layers = store.layer_names
    
    # Metric Storage for overall analysis
    # {layer: {mem_max_T: val, non_mem_max_T: val, ...}}
    analysis_stats = {}

    for layer_name in all_layers:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Subplot 1: T, Subplot 2: T/4
        for i, (label, step_name) in enumerate([("T", "Step 0"), ("T_div_4", "Step 37")]):
            ax = axes[i]
            
            # Get Mem scores
            mem_data = next(d for d in collected_data if d["type"] == "Memorization" and d["label"] == label)
            mem_scores = mem_data["scores"].get(layer_name, torch.zeros(77)).numpy()
            
            # Get Non-Mem scores
            non_mem_data = next(d for d in collected_data if d["type"] == "Non-memorization" and d["label"] == label)
            non_mem_scores = non_mem_data["scores"].get(layer_name, torch.zeros(77)).numpy()
            
            # Plot
            x = np.arange(len(mem_scores))
            ax.plot(x, mem_scores, label='Memorization', color='red', alpha=0.8, linewidth=2)
            ax.plot(x, non_mem_scores, label='Non-memorization', color='blue', alpha=0.8, linewidth=1.5, linestyle='--')
            
            # Styling
            ax.set_title(f"Attention Score Distribution ({step_name})", fontsize=14)
            ax.set_xlabel("Token Index", fontsize=12)
            ax.set_ylabel("Mean Attention Score", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Stats recording
            if layer_name not in analysis_stats: analysis_stats[layer_name] = {}
            analysis_stats[layer_name][f"{label}_Mem_Max"] = mem_scores.max()
            analysis_stats[layer_name][f"{label}_NonMem_Max"] = non_mem_scores.max()
            analysis_stats[layer_name][f"{label}_Mem_Entropy"] = -np.sum(mem_scores * np.log(mem_scores + 1e-9))
            analysis_stats[layer_name][f"{label}_NonMem_Entropy"] = -np.sum(non_mem_scores * np.log(non_mem_scores + 1e-9))

        safe_name = layer_name.replace(".", "_")
        plt.suptitle(f"Layer: {layer_name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{safe_name}_dist.png"), dpi=300)
        plt.close()

    # 4. Save Analysis CSV
    df_stats = pd.DataFrame.from_dict(analysis_stats, orient='index')
    csv_path = os.path.join(results_dir, "attention_stats.csv")
    df_stats.to_csv(csv_path)
    print(f"Saved plots and stats to {results_dir}")
    print(df_stats.head())

if __name__ == "__main__":
    analyze_and_plot()
