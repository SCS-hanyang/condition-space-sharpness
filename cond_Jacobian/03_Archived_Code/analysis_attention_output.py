
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

# --- 1. Capture Processor ---
class CaptureAttnMapProcessor:
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

        # QK^T
        attention_scores = attn.scale * torch.bmm(query, key.transpose(-1, -2))
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = attention_scores.softmax(dim=-1)
        
        # --- Capture Map ---
        # attention_probs: (Batch*Heads, Q, K)
        # We want the map corresponding to the Conditional Prompt pass
        # Batch size is usually 1 (Manual Loop). Heads = 8.
        
        true_batch_size = hidden_states.shape[0]
        
        # Reshape to (Batch, Heads, Q, K)
        probs_reshaped = attention_probs.view(true_batch_size, -1, attention_probs.shape[1], attention_probs.shape[2])
        
        # Average heads -> (Batch, Q, K)
        avg_probs = probs_reshaped.mean(dim=1)
        
        # Take batch 0
        current_map = avg_probs[0] # (Spatial, Tokens)
        
        # Check if store is active
        if self.store.active:
            self.store.save_map(self.layer_name, current_map)
        
        # -------------------
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class AttentionDataStore:
    def __init__(self):
        # Data Structure: {layer_name: [map_t0, map_t1, ...]}
        self.records = {}
        self.layer_names = []
        self.active = False
        
    def save_map(self, layer_name, attn_map):
        if layer_name not in self.records:
            self.records[layer_name] = []
            self.layer_names.append(layer_name)
        # Store on CPU to save VRAM
        self.records[layer_name].append(attn_map.detach().cpu())
        
    def reset(self):
        self.records = {}
        self.layer_names = []
        self.active = False

def setup_hooks(pipe, store):
    def register_recursive(module, name_prefix):
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            is_cross = 'attn2' in name and isinstance(child, Attention)
            if is_cross:
                processor = CaptureAttnMapProcessor(store, full_name)
                child.set_processor(processor)
            else:
                register_recursive(child, full_name)
    register_recursive(pipe.unet, "")
    print(f"Registered hooks for {len(store.layer_names)} layers (initially empty).")

# --- Helper: Load Prompts ---
def load_prompts():
    try:
        path = 'prompts/memorized_laion_prompts.csv'
        if not os.path.exists(path):
            path = 'init_noise_diffusion_memorization/' + path
        df = pd.read_csv(path, sep=';')
        memorized = df['Caption'].tolist()
    except:
        memorized = ["Mothers influence on her young hippo"] * 100

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
        "A desert landscape with sand dunes and a cactus",
        "A river winding through a green valley with mountains",
        "A close up of a drop of water on a green leaf",
        "A cherry blossom tree shedding petals in the spring breeze",
        "A winter landscape with a frozen lake and snowy pines",
        "A cute cat sitting on a windowsill looking at the rain",
        "A golden retriever dog playing fetch in a park",
        "A majestic lion resting under the shade of a tree",
        "A colorful parrot sitting on a branch in the jungle",
        "A small squirrel holding a nut in its paws",
        "A herd of elephants walking across the savannah",
        "A panda eating bamboo in a bamboo forest",
        "A dolphin jumping out of the water in the ocean",
        "A group of penguins standing on an iceberg",
        "A brown bear fishing for salmon in a river",
        "A butterfly resting on a bright red flower",
        "A white horse galloping through a green meadow",
        "A sloth hanging from a tree branch",
        "A koala sleeping in a eucalyptus tree",
        "A red fox walking through the snow",
        "Modern office interior with large windows and plants",
        "A futuristic city skyline at night with neon lights",
        "A busy street market in Tokyo with colorful signs",
        "A vintage car parked on a cobblestone street in Paris",
        "A cozy living room with a fireplace and books on shelves",
        "A tall skyscraper reflecting the clouds on its glass surface",
        "A bridge spanning across a wide river at twilight",
        "A quiet library hall with rows of wooden bookshelves",
        "A busy coffee shop with people working on laptops",
        "An empty subway station with tiled walls and lights",
        "A classic European village street with flower boxes",
        "A modern kitchen with stainless steel appliances",
        "A rustic wooden cabin in the middle of the woods",
        "An aerial view of a busy city intersection",
        "A large stadium filled with cheering crowd",
        "A red apple on a wooden table with a knife",
        "A plate of pasta with tomato sauce and fresh basil",
        "A delicious pizza with pepperoni and melted cheese",
        "A cup of steaming hot coffee next to a notebook",
        "A colorful bowl of fruit salad on a picnic blanket",
        "A vintage camera sitting on a stack of old books",
        "A pair of glasses resting on a newspaper",
        "A wooden guitar leaning against a brick wall",
        "A bouquet of roses in a glass vase",
        "A freshly baked loaf of bread on a cutting board",
        "A glass of red wine and cheese on a platter",
        "A bicycle parked against a fence",
        "A set of colorful pencils in a cup",
        "A lit candle on a dark table",
        "A pair of running shoes on a pavement",
        "Abstract painting with blue and orange geometric shapes",
        "A surreal landscape with floating islands and waterfalls",
        "A digital art illustration of a cyberpunk character",
        "A minimalist design with black lines on white background",
        "A colorful explosion of paint in slow motion",
        "A 3d render of a shiny metallic sphere",
        "A pattern of colorful polka dots",
        "A dreamlike sequence of clouds forming shapes",
        "An artistic representation of the solar system",
        "A fractal image with infinite spiraling patterns",
        "A smiling woman holding a cup of coffee looking out window",
        "A man reading a book on a park bench",
        "Children playing soccer in a grass field",
        "A lone astronaut walking on the surface of Mars",
        "A robot working in a factory assembly line",
        "A artist painting on a canvas in a studio",
        "A chef cooking in a busy restaurant kitchen",
        "A musician playing violin on a stage",
        "A student studying late at night with a lamp",
        "A gardener watering plants in a greenhouse",
        "A group of friends hiking up a mountain trail",
        "A person riding a bicycle at sunset",
        "A dancer performing ballet on a stage",
        "A photographer taking pictures of nature",
        "A old man feeding pigeons in a square",
        "A blue luxury sedan driving on a coastal road",
        "A sailboat regatta on a windy day",
        "A close up of a complex mechanical watch mechanism",
        "A drone flying over a forest",
        "A pile of autumn leaves in a park",
        "A snowman with a carrot nose and a scarf",
        "A beautifully decorated christmas tree",
        "A bowl of ramen with egg and pork",
        "A slice of chocolate cake on a white plate",
        "A glass of orange juice with ice cubes",
        "A stack of gold coins on a table",
        "A pair of headphones on a mixing desk",
        "A opened gift box with colorful wrapping paper",
        "A antique clock on a mantelpiece",
        "A thunderstorm viewed from a window"
    ]
    unmemorized = list(dict.fromkeys(unmemorized))
    return memorized, unmemorized

def analyze_and_plot():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "CompVis/stable-diffusion-v1-4"
    results_dir = "init_noise_diffusion_memorization/results/analysis_attention_output"
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
    
    memorized_prompts, unmemorized_prompts = load_prompts()
    target_idx = 20
    mem_prompt = memorized_prompts[target_idx]
    non_mem_prompt = unmemorized_prompts[target_idx]
    
    # Pre-compute unconditional embeddings
    uncond_prompt_embeds = pipe._encode_prompt("", device, 1, False, None)
    
    # Store for analysis
    # {p_type: {layer_name: [diff_t0, diff_t1, ...]}}
    layer_changes = {} 
    
    store = AttentionDataStore()
    setup_hooks(pipe, store)
    
    num_inference_steps = 50
    
    for p_type, prompt in [("Memorization", mem_prompt), ("Non-memorization", non_mem_prompt)]:
        print(f"Processing {p_type}...")
        store.reset()
        
        cond_prompt_embeds = pipe._encode_prompt(prompt, device, 1, False, None)
        
        height = pipe.unet.config.sample_size * pipe.vae_scale_factor
        width = pipe.unet.config.sample_size * pipe.vae_scale_factor
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps
        
        gen = torch.Generator(device).manual_seed(42)
        latents = pipe.prepare_latents(
            1, pipe.unet.config.in_channels, height, width,
            cond_prompt_embeds.dtype, device, gen, None
        )
        
        # Loop
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                # Measure Attention (Conditional)
                store.active = True
                pipe.unet(
                    pipe.scheduler.scale_model_input(latents, t),
                    t,
                    encoder_hidden_states=cond_prompt_embeds
                )
                store.active = False
                
                # Uncond Update
                noise_pred = pipe.unet(
                    pipe.scheduler.scale_model_input(latents, t),
                    t,
                    encoder_hidden_states=uncond_prompt_embeds
                ).sample
                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
        # Analysis: Calculate changes between steps for each layer
        # store.records: {layer: [map_0, map_1, ... map_49]}
        # We calculate Diff(t) = Norm(Map_t - Map_{t-1})
        # For t=0, diff is 0 or undefined. We'll start from t=1.
        
        layer_changes[p_type] = {}
        
        for layer_name in store.records:
            maps = store.records[layer_name] # List of tensors
            diffs = []
            
            # Change from t to t+1 ?
            # Map sequence is t=Max to t=0 (Trajectory order)
            # Step index 0 is T_start. Step 49 is T_end.
            
            for t in range(1, len(maps)):
                m_prev = maps[t-1].float()
                m_curr = maps[t].float()
                
                # Metric: L2 norm of difference / Normalized by size?
                # Just L2 norm for now
                diff = torch.norm(m_curr - m_prev).item()
                diffs.append(diff)
            
            # Pad first one to match x-axis length (50)
            # t=0 has no prev, so 0 val
            if len(diffs) < 50:
                diffs = [0] + diffs
            layer_changes[p_type][layer_name] = diffs
            
    # --- Visualization ---
    print("Generating Plots...")
    
    # 1. Aggregated Change Dynamics (Avg over all layers)
    plt.figure(figsize=(10, 6))
    
    for p_type in ["Memorization", "Non-memorization"]:
        # Avg across layers
        all_diffs = [layer_changes[p_type][lname] for lname in layer_changes[p_type]]
        avg_diffs = np.mean(all_diffs, axis=0) # (50,)
        
        lbl = p_type
        plt.plot(np.arange(len(avg_diffs)), avg_diffs, label=lbl, linewidth=2)
        
    # Highlight Basin Interval (8-12)
    plt.axvspan(8, 12, color='gray', alpha=0.2, label='Basin Exit Interval (8-12)')
    
    plt.title("Run-time Attention Stability (Avg L2 Diff between consecutive steps)")
    plt.xlabel("Step Index")
    plt.ylabel("Avg L2 Difference (Map_t - Map_{t-1})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, "avg_attention_change.png"), dpi=300)
    plt.close()
    
    # 2. Per-Layer Heatmap of Changes
    # X: Step, Y: Layer
    # We want to see WHICH layer contributes most to the change at step 10
    
    for p_type in ["Memorization", "Non-memorization"]:
        # Prepare Matrix: (Layers, Steps)
        # Using sorted layer names
        sorted_layers = sorted(layer_changes[p_type].keys())
        matrix = []
        for lname in sorted_layers:
            matrix.append(layer_changes[p_type][lname])
        
        matrix = np.array(matrix) # (L, S)
        
        plt.figure(figsize=(15, 10))
        sns.heatmap(matrix, cmap="YlOrRd", cbar_kws={'label': 'L2 Difference'})
        
        # Labels
        # Shorten layer names
        short_names = [n.split('.')[-2]+"."+n.split('.')[-1] for n in sorted_layers] 
        # Actually standard names are like down_blocks.0.attentions.0...
        # Let's clean up
        clean_names = []
        for n in sorted_layers:
            parts = n.split('.')
            if 'down_blocks' in n: pre = f"D{parts[1]}"
            elif 'up_blocks' in n: pre = f"U{parts[1]}"
            elif 'mid_block' in n: pre = "M"
            else: pre = "?"
            # parts[-1] is usually 'transformer_blocks', then '0', 'attn2'
            # The structure is ...attentions.X.transformer_blocks.0.attn2...
            # This is complex. Standard shortening:
            clean_names.append(n[-30:]) # Just take last 30 chars
            
        plt.yticks(np.arange(len(clean_names))+0.5, clean_names, rotation=0, fontsize=8)
        plt.xlabel("Step Index")
        plt.title(f"Layer-wise Attention Stability: {p_type}")
        
        # Highlight Interval
        # In heatmap, x-axis is 0..49
        plt.axvline(8, color='blue', linestyle='--')
        plt.axvline(12, color='blue', linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"heatmap_attention_change_{p_type}.png"), dpi=300)
        plt.close()

    print("Analysis Complete.")

if __name__ == "__main__":
    analyze_and_plot()
