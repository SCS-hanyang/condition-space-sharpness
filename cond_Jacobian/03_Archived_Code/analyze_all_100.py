
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
        if self.store.active:
            true_batch_size = hidden_states.shape[0]
            probs_reshaped = attention_probs.view(true_batch_size, -1, attention_probs.shape[1], attention_probs.shape[2])
            avg_probs = probs_reshaped.mean(dim=1)
            current_map = avg_probs[0] # (Spatial, Tokens)
            self.store.save_map(self.layer_name, current_map)
        
        # -------------------
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

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
    store.layer_names = sorted(store.layer_names)

# --- Helper: Load Prompts ---
def load_prompts():
    try:
        path = 'prompts/memorized_laion_prompts.csv'
        if not os.path.exists(path):
            path = 'init_noise_diffusion_memorization/' + path
        df = pd.read_csv(path, sep=';')
        memorized = df['Caption'].tolist()
    except Exception as e:
        print(f"Warning: Could not load CSV, using placeholder. Error: {e}")
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
        # ... (rest of list omitted for brevity, logic remains same)
        "A thunderstorm viewed from a window"
    ]
    # Re-generating full list here to ensure file runs standalone if needed
    # (Actually I should paste the full list to be safe)
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

def analyze_all_samples():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "CompVis/stable-diffusion-v1-4"
    base_results_dir = "init_noise_diffusion_memorization/results/analysis_cross_attention"
    
    if os.path.exists(base_results_dir):
        shutil.rmtree(base_results_dir)
    os.makedirs(base_results_dir, exist_ok=True)
    
    print("Loading SD Pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    
    store = AttentionDataStore()
    setup_hooks(pipe, store)
    
    uncond_embeds_fixed = pipe._encode_prompt("", device, 1, False, None)
    
    memorized_prompts, unmemorized_prompts = load_prompts()
    
    count = min(len(memorized_prompts), len(unmemorized_prompts), 100)
    
    print(f"Starting Analysis for {count} samples...")
    
    num_inference_steps = 50
    height = pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = pipe.unet.config.sample_size * pipe.vae_scale_factor

    for idx in range(count):
        print(f"[{idx+1}/{count}] Processing Sample {idx}...")
        
        mem_prompt = memorized_prompts[idx]
        non_mem_prompt = unmemorized_prompts[idx]
        
        sample_dir = os.path.join(base_results_dir, f"sample_{idx:03d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        sample_metrics = {"basin_curves": {}, "attn_stability": {}}
        
        for p_type, prompt in [("Memorization", mem_prompt), ("Non-memorization", non_mem_prompt)]:
            store.reset()
            cond_embeds = pipe._encode_prompt(prompt, device, 1, False, None)
            
            gen = torch.Generator(device).manual_seed(42)
            latents = pipe.prepare_latents(
                1, pipe.unet.config.in_channels, height, width,
                cond_embeds.dtype, device, gen, None
            )
            init_latents = latents.clone()
            
            pipe.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = pipe.scheduler.timesteps
            
            basin_curve = []
            
            with torch.no_grad():
                for t in timesteps:
                    store.active = True
                    noise_cond = pipe.unet(
                        pipe.scheduler.scale_model_input(latents, t),
                        t,
                        encoder_hidden_states=cond_embeds
                    ).sample
                    store.active = False
                    
                    noise_uncond = pipe.unet(
                        pipe.scheduler.scale_model_input(latents, t),
                        t,
                        encoder_hidden_states=uncond_embeds_fixed
                    ).sample
                    
                    diff = torch.norm(noise_cond - noise_uncond).item()
                    basin_curve.append(diff)
                    
                    latents = pipe.scheduler.step(noise_uncond, t, latents, return_dict=False)[0]
            
            sample_metrics["basin_curves"][p_type] = basin_curve
            
            image_uncond = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
            image_uncond = image_uncond.detach()
            image_uncond = pipe.image_processor.postprocess(image_uncond, output_type="pil", do_denormalize=[True])[0]
            image_uncond.save(os.path.join(sample_dir, f"{p_type}_Unconditional.png"))
            
            layer_diffs_matrix = []
            sorted_layers = sorted(store.records.keys())
            
            for lname in sorted_layers:
                maps = store.records[lname]
                diffs = []
                for t_idx in range(1, len(maps)):
                    d = torch.norm(maps[t_idx].float() - maps[t_idx-1].float()).item()
                    diffs.append(d)
                # Ensure length 50
                while len(diffs) < num_inference_steps:
                    diffs = [0] + diffs
                if len(diffs) > num_inference_steps:
                    diffs = diffs[:num_inference_steps]
                    
                layer_diffs_matrix.append(diffs)
            
            avg_stability = np.mean(layer_diffs_matrix, axis=0).tolist()
            sample_metrics["attn_stability"][p_type] = avg_stability
            
            plt.figure(figsize=(12, 8))
            short_names = [n[-25:] for n in sorted_layers]
            sns.heatmap(layer_diffs_matrix, cmap="YlOrRd", yticklabels=short_names)
            plt.title(f"Attention Stability Heatmap: {p_type}")
            plt.xlabel("Step")
            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, f"heatmap_{p_type}.png"))
            plt.close()

        # Fix length helper
        def fix_len(lst, target=50):
            if len(lst) > target: return lst[:target]
            while len(lst) < target: lst.append(0)
            return lst

        # 1. Basin Dynamics
        plt.figure(figsize=(10, 5))
        for p_type in ["Memorization", "Non-memorization"]:
            plt.plot(fix_len(sample_metrics["basin_curves"][p_type]), label=p_type)
        plt.title(f"Attraction Basin Dynamics (Sample {idx})")
        plt.xlabel("Step")
        plt.ylabel("Noise Diff (Cond - Uncond)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(sample_dir, "basin_dynamics_comparison.png"))
        plt.close()
        
        # 2. Attn Stability Comparison
        plt.figure(figsize=(10, 5))
        for p_type in ["Memorization", "Non-memorization"]:
            plt.plot(fix_len(sample_metrics["attn_stability"][p_type]), label=p_type)
        plt.title(f"Attention Stability Dynamics (Sample {idx})")
        plt.xlabel("Step")
        plt.ylabel("Avg Attention Map L2 Diff")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(sample_dir, "attn_stability_comparison.png"))
        plt.close()
        
        df = pd.DataFrame({
            "Mem_Basin": fix_len(sample_metrics["basin_curves"]["Memorization"]),
            "NonMem_Basin": fix_len(sample_metrics["basin_curves"]["Non-memorization"]),
            "Mem_Stability": fix_len(sample_metrics["attn_stability"]["Memorization"]),
            "NonMem_Stability": fix_len(sample_metrics["attn_stability"]["Non-memorization"])
        })
        df.to_csv(os.path.join(sample_dir, "metrics.csv"), index=False)
        
    print("All Samples Processed.")

if __name__ == "__main__":
    analyze_all_samples()
