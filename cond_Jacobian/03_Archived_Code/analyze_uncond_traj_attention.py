
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
        
        # Capture logic for Unconditional Trajectory
        
        true_batch_size = hidden_states.shape[0]
        
        # Flatten Heads
        # Reshape: (Batch, Heads, Q, K)
        probs_reshaped = attention_probs.view(true_batch_size, -1, attention_probs.shape[1], attention_probs.shape[2])
        
        # Take mean over heads
        avg_probs = probs_reshaped.mean(dim=1) # (Batch, Q, K)
        
        # If batch=1, just take 0
        final_map = avg_probs[0] # (Q, K) -> (Spatial, Tokens)
        
        # Store
        self.store.capture(self.layer_name, final_map)
        
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

def resize_map(attn_map, target_dim=768):
    T = attn_map.shape[1]
    reshaped = attn_map.t().unsqueeze(0)
    resized = F.interpolate(reshaped, size=target_dim, mode='linear', align_corners=False)
    final_map = resized.squeeze(0).t()
    return final_map

# --- Helper: Load Prompts ---
def load_prompts():
    # Memorized
    try:
        # Adjust path if running from root or inside dir
        if os.path.exists('prompts/memorized_laion_prompts.csv'):
            path = 'prompts/memorized_laion_prompts.csv'
        elif os.path.exists('init_noise_diffusion_memorization/prompts/memorized_laion_prompts.csv'):
            path = 'init_noise_diffusion_memorization/prompts/memorized_laion_prompts.csv'
        else:
            # Fallback path if neither exists
            path = 'prompts/memorized_laion_prompts.csv'
            
        df = pd.read_csv(path, sep=';')
        memorized = df['Caption'].tolist()
    except Exception as e:
        print(f"Error loading memorized prompts: {e}")
        memorized = ["Mothers influence on her young hippo"] * 100

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
    # Ensure distinct
    unmemorized = list(dict.fromkeys(unmemorized))
    return memorized, unmemorized

# --- 3. Main Execution ---

def run_uncond_trajectory_analysis():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "CompVis/stable-diffusion-v1-4"
    base_results_dir = "init_noise_diffusion_memorization/results/memorization_on_CA/uncond_trajectory"
    
    if os.path.exists(base_results_dir):
        shutil.rmtree(base_results_dir)
    os.makedirs(base_results_dir, exist_ok=True)
    
    print("Loading SD Pipeline...")
    # --- FIX: Disable Safety Checker ---
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    
    # Load Prompts
    memorized_prompts, unmemorized_prompts = load_prompts()
    
    # Target Index
    target_idx = 20
    
    # Ensure bounds
    if target_idx >= len(memorized_prompts): target_idx = 0
    if target_idx >= len(unmemorized_prompts): target_idx = 0
    
    mem_prompt = memorized_prompts[target_idx]
    non_mem_prompt = unmemorized_prompts[target_idx]
    
    print(f"Selected Index: {target_idx}")
    print(f"Memorized Prompt: {mem_prompt}")
    print(f"Non-memorized Prompt: {non_mem_prompt}")
    
    store = AttentionStore()
    setup_attention_hooks(pipe, store)
    
    num_inference_steps = 50
    
    # Pre-compute unconditional embeddings (Empty string)
    uncond_embeds_fixed = pipe._encode_prompt("", device, 1, False, None)
    
    all_results = {}
    
    # Create sample directory
    sample_dir = os.path.join(base_results_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    all_basin_curves = {}

    for p_type, prompt in [("Memorization", mem_prompt), ("Non-memorization", non_mem_prompt)]:
        print(f"Running Unconditional Trajectory for: {p_type}")
        store.reset()
        
        # Prepare Conditional Embeds
        cond_embeds = pipe._encode_prompt(prompt, device, 1, False, None)

        # Prepare Latents
        height = pipe.unet.config.sample_size * pipe.vae_scale_factor
        width = pipe.unet.config.sample_size * pipe.vae_scale_factor
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps
        
        gen = torch.Generator(device).manual_seed(42)
        
        latents = pipe.prepare_latents(
            1, pipe.unet.config.in_channels, height, width,
            cond_embeds.dtype, device, gen, None
        )
        
        init_latents = latents.clone()
        basin_curve = []
        
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                # 1. Measure Attention & Get Conditional Noise
                # We need the output now for difference calc
                latent_input = pipe.scheduler.scale_model_input(latents, t)
                
                noise_cond = pipe.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=cond_embeds
                ).sample
                
                # Capture hook step
                store.step()
                
                # 2. Get Unconditional Noise & Update Step
                noise_uncond = pipe.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=uncond_embeds_fixed
                ).sample
                
                # Calculate Basin Metric (Sharpness/Difference)
                diff = torch.norm(noise_cond - noise_uncond).item()
                basin_curve.append(diff)
                
                latents = pipe.scheduler.step(noise_uncond, t, latents, return_dict=False)[0]
                
        all_results[p_type] = store.data
        all_basin_curves[p_type] = basin_curve
        layer_order = store.layer_names
        
        # --- Save Images ---
        image_uncond = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        image_uncond = image_uncond.detach() # Explicit detach fix
        image_uncond = pipe.image_processor.postprocess(image_uncond, output_type="pil", do_denormalize=[True])[0]
        image_uncond.save(os.path.join(sample_dir, f"{p_type}_Unconditional.png"))
        
        print(f"Generating Conditional Image for {p_type}...")
        image_cond = pipe(
            prompt, 
            num_inference_steps=num_inference_steps, 
            latents=init_latents, 
            guidance_scale=7.5
        ).images[0]
        image_cond.save(os.path.join(sample_dir, f"{p_type}_Conditional.png"))

    # --- Plot Attraction Basin Dynamics ---
    print("Generating Attraction Basin Plot...")
    plt.figure(figsize=(10, 6))
    
    for p_type, curve in all_basin_curves.items():
        color = 'red' if p_type == "Memorization" else 'blue'
        x_axis = np.arange(len(curve)) # Dynamic x-axis
        plt.plot(x_axis, curve, label=p_type, color=color, linewidth=2)
        
    plt.title("Attraction Basin Dynamics (Unconditional Trajectory)")
    plt.xlabel("Step Index (0=T -> 50=0)")
    plt.ylabel(r"$\|\epsilon(x_t, c) - \epsilon(x_t, \emptyset)\|_2$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(base_results_dir, "attraction_basin_dynamics.png"), dpi=300)
    plt.close()
    
    # Save Curve Data
    # Pad if lengths differ (unlikely here but safe)
    max_len = max(len(c) for c in all_basin_curves.values())
    df_curves = pd.DataFrame({k: v + [None]*(max_len-len(v)) for k,v in all_basin_curves.items()})
    df_curves['Step'] = np.arange(max_len)
    df_curves.to_csv(os.path.join(base_results_dir, "attraction_basin_curves.csv"), index=False)

    print("Inference Complete. Generated Verification Samples in 'samples/'. Generating Plots...")
    
    # Iterate Steps
    for step_idx in range(num_inference_steps):
        step_dir = os.path.join(base_results_dir, f"step_{step_idx:02d}")
        os.makedirs(step_dir, exist_ok=True)
        print(f"Processing Step {step_idx}/{num_inference_steps}...")
        
        # --- Stats Logic ---
        stats_data = []
        for lname in layer_order:
            for p_type in ["Memorization", "Non-memorization"]:
                raw_map = all_results[p_type][step_idx].get(lname)
                if raw_map is not None:
                    # Resize/Process
                    # final_map = resize_map(raw_map, target_dim=768) 
                    # Actually stats on raw spatial is accurate for "sum" or "mean".
                    
                    token_scores = raw_map.mean(dim=0) # (77,)
                    s0 = token_scores[0].item()
                    s_rest = token_scores[1:].sum().item()
                    
                    stats_data.append({
                        "Type": p_type,
                        "Layer": lname,
                        "FirstToken": s0,
                        "RestTokens": s_rest
                    })
        
        # Save Stats
        df_step = pd.DataFrame(stats_data)
        df_step.to_csv(os.path.join(step_dir, f"token_stats_step_{step_idx}.csv"), index=False)
        
        # Plot Comparison
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
        
        plt.title(f"[Uncond Trajectory] First vs Rest (Step {step_idx})")
        plt.xlabel("Layer Index")
        plt.ylabel("Attention (on Cond Prompt)")
        
        short_ticks = [n.split('.')[0] + '.' + n.split('.')[1] for n in mem_df['Layer']]
        plt.xticks(x_vals, short_ticks, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(step_dir, f"comparison_plot_step_{step_idx}.png"), dpi=150)
        plt.close()
        
        # --- Heatmaps Logic ---
        heatmap_dir = os.path.join(step_dir, "heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)
        
        for lname in layer_order:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            for i, p_type in enumerate(["Memorization", "Non-memorization"]):
                raw_map = all_results[p_type][step_idx].get(lname)
                if raw_map is not None:
                    final_map = resize_map(raw_map, target_dim=768)
                    sns.heatmap(final_map.numpy(), ax=axes[i], cmap="YlGnBu", cbar=True, vmin=0, vmax=1.0)
                    axes[i].set_title(p_type)
                    axes[i].set_xlabel("Token Index")
                    if i == 0: axes[i].set_ylabel("Spatial Dim (0-768)")
                    else: axes[i].set_yticks([])
            
            plt.suptitle(f"Layer: {lname} (Step {step_idx}) - Uncond Traj", fontsize=16)
            plt.tight_layout()
            safe_lname = lname.replace(".", "_")
            plt.savefig(os.path.join(heatmap_dir, f"{safe_lname}.png"), dpi=100)
            plt.close()

    print("Unconditional Analysis Complete.")

if __name__ == "__main__":
    run_uncond_trajectory_analysis()
