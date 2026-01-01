
import os
import random
import argparse
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from diffusers import DDIMScheduler
from local_sd_pipeline import LocalStableDiffusionPipeline
from tqdm import tqdm

# Ensure results directory exists
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set random seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def permute_words(text):
    """Randomly shuffle words in the text."""
    words = text.split()
    if len(words) > 1:
        # Fisher-Yates shuffle equivalent
        random.shuffle(words)
    return " ".join(words)

def analyze_vectors_normalized(vec_base, vec_target):
    """
    Computes absolute and relative distances between two vectors.
    """
    v1 = vec_base.flatten().float()
    v2 = vec_target.flatten().float()
    
    # Base Norms for Normalization
    norm_v1_l2 = torch.norm(v1, p=2).item()
    norm_v1_l1 = torch.norm(v1, p=1).item()
    
    # Avoid division by zero
    norm_v1_l2 = norm_v1_l2 if norm_v1_l2 > 1e-6 else 1.0
    norm_v1_l1 = norm_v1_l1 if norm_v1_l1 > 1e-6 else 1.0
    
    # 1. Cosine Similarity (Already normalized by definition)
    cos_sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
    
    # 2. L2 Analysis
    l2_diff = torch.norm(v1 - v2, p=2).item()
    rel_l2 = l2_diff / norm_v1_l2
    
    # 3. L1 Analysis
    l1_diff = torch.norm(v1 - v2, p=1).item()
    rel_l1 = l1_diff / norm_v1_l1
    
    # 4. Max Absolute Difference
    max_diff = torch.max(torch.abs(v1 - v2)).item()
    # Relative Max Diff (vs average absolute value or max value of base?)
    # Let's use max value of base as reference
    max_val_v1 = torch.max(torch.abs(v1)).item()
    rel_max = max_diff / max_val_v1 if max_val_v1 > 1e-6 else 0.0
    
    return {
        "CosSim": cos_sim,
        "L2_Diff": l2_diff,
        "Relative_L2": rel_l2,
        "L1_Diff": l1_diff,
        "Relative_L1": rel_l1,
        "Max_Diff": max_diff,
        "Relative_Max": rel_max
    }

def get_jacobian_norm_normalized(pipe, prompt, num_projections=1):
    """
    Estimates Frobenius norm of Jacobian. 
    Also returns Relative Jacobian Norm (Sensitivity / Output Norm).
    """
    device = pipe.device
    dtype = pipe.unet.dtype
    
    # Get Embeddings
    text_inputs = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    input_ids = text_inputs.input_ids.to(device)
    
    with torch.no_grad():
        prompt_embeds = pipe.text_encoder(input_ids)[0].to(dtype=dtype)
    prompt_embeds.requires_grad_(True)
    
    # Prepare Inputs
    height = pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = pipe.unet.config.sample_size * pipe.vae_scale_factor
    
    # Use fixed generator for noise
    gen = torch.Generator(device=device).manual_seed(42)
    latents = torch.randn((1, pipe.unet.config.in_channels, height//8, width//8), device=device, generator=gen, dtype=dtype)
    t = torch.tensor([999], device=device).long()
    
    # Forward
    noise_pred = pipe.unet(latents, t, encoder_hidden_states=prompt_embeds).sample
    
    # Compute Output Norm for Normalization
    output_norm = torch.norm(noise_pred.detach().float(), p='fro').item()
    
    # Jacobian Estimation
    total_sq_norm = 0.0
    for _ in range(num_projections):
        v = torch.randn_like(noise_pred)
        grads = torch.autograd.grad(noise_pred, prompt_embeds, grad_outputs=v, retain_graph=True, create_graph=False)[0]
        total_sq_norm += grads.pow(2).sum().item()
        
    jacobian_norm = (total_sq_norm / num_projections) ** 0.5
    prompt_embeds.requires_grad_(False)
    
    # Normalize: ||J|| / ||epsilon||
    # Ideally we normalization by input norm too ||J|| * (||c|| / ||epsilon||) for unitless elasticity?
    # Simply dividing by output norm gives "Percentage change in output per unit change in input"
    relative_jacobian = jacobian_norm / output_norm if output_norm > 1e-6 else 0.0
    
    return jacobian_norm, relative_jacobian

def analyze_output_set_properties_normalized(pipe, base_prompt, num_variations=10):
    """
    Set Analysis with Normalization.
    Returns Spread (Absolute) and Coefficient of Variation (Spread / AvgNorm).
    """
    device = pipe.device
    dtype = pipe.unet.dtype

    variations = [base_prompt]
    words = base_prompt.split()
    for _ in range(num_variations - 1):
        if len(words) > 1:
            p_words = words.copy()
            random.shuffle(p_words)
            variations.append(" ".join(p_words))
        else:
            variations.append(base_prompt)
            
    # Use fixed noise for all variations to isolate text effect
    local_gen = torch.Generator(device=device).manual_seed(42)
    height = pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = pipe.unet.config.sample_size * pipe.vae_scale_factor
    latents = torch.randn((1, pipe.unet.config.in_channels, height // 8, width // 8), device=device, generator=local_gen, dtype=dtype)
    t = torch.tensor([999], device=device).long()

    vectors = []
    for txt in variations:
        try:
            text_input = pipe.tokenizer(txt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            with torch.no_grad():
                embeds = pipe.text_encoder(text_input.input_ids.to(device))[0].to(dtype=dtype)
                cond_tensor = pipe.unet(latents, t, encoder_hidden_states=embeds).sample
                vectors.append(cond_tensor.detach().cpu().float())
        except:
            continue
            
    if not vectors:
        return 0.0, 0.0, 0.0
        
    stack = torch.stack(vectors) # [N, C, H, W]
    N = stack.shape[0]
    stack_flat = stack.view(N, -1)
    
    # Metrics
    centroid = stack_flat.mean(dim=0, keepdim=True)
    dists = torch.norm(stack_flat - centroid, p=2, dim=1)
    spread = dists.mean().item()
    
    norms = torch.norm(stack_flat, p=2, dim=1)
    avg_norm = norms.mean().item()
    
    # Relative Spread (Coefficient of Variation equivalent)
    # How large is the spread compared to the magnitude of the vectors?
    rel_spread = spread / avg_norm if avg_norm > 1e-6 else 0.0
    
    return spread, avg_norm, rel_spread


# Pipeline Wrapper (Optional reuse of existing class structure)
class FirstStepAnalysisPipeline(LocalStableDiffusionPipeline):
    # Just a wrapper to be consistent with previous code structure, 
    # but actual analysis calls use manual components as enacted above.
    pass

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    print(f"Using device: {device}")
    
    # 1. Load Data
    # A. Memorized
    try:
        df = pd.read_csv('prompts/memorized_laion_prompts.csv', sep=';')
        memorized_prompts = df['Caption'].tolist()
        # Ensure we don't have too many if testing
        # memorized_prompts = memorized_prompts[:500] 
        print(f"Loaded {len(memorized_prompts)} memorized prompts.")
    except Exception as e:
        print(f"Error loading memorized prompts: {e}")
        return

    # B. Unmemorized (Expanded List - ~100 items)
    unmemorized_prompts = [
        # Nature & Scenery
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
        
        # Animals
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
        
        # Urban & Architecture
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
        
        # Objects & Food
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

        # Abstract & Concepts
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
        
        # People & Routine
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
        
        # Random / Mixed
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
    # Ensure unique and roughly 100
    unmemorized_prompts = list(set(unmemorized_prompts))
    print(f"Using {len(unmemorized_prompts)} unmemorized prompts (Expanded).")

    # 4. Normalize Counts to be Equal (e.g. 100 vs 100)
    target_count = 100
    
    # Shuffle and slice memorized
    random.shuffle(memorized_prompts)
    memorized_subset = memorized_prompts[:target_count]
    
    # Shuffle and slice unmemorized (if we have enough, otherwise take all)
    random.shuffle(unmemorized_prompts)
    unmemorized_subset = unmemorized_prompts[:target_count]
    
    # Check if we have enough
    real_min = min(len(memorized_subset), len(unmemorized_subset))
    memorized_subset = memorized_subset[:real_min]
    unmemorized_subset = unmemorized_subset[:real_min]
    
    print(f"Balanced Analysis: Using {len(memorized_subset)} Memorized vs {len(unmemorized_subset)} Unmemorized prompts.")

    # 2. Load Model
    model_id = "CompVis/stable-diffusion-v1-4"
    print(f"Loading model: {model_id}...")
    
    # We only need components for manual pipeline
    pipeline = LocalStableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    # We use this 'pipeline' object as a container for parts (unet, text_encoder...)
    print("Model loaded.")

    # 3. Run Analysis
    results = []
    
    experiments = []
    for p in memorized_subset:
        experiments.append((p, "Memorized"))
    for p in unmemorized_subset:
        experiments.append((p, "Unmemorized"))
        
    print(f"Starting comprehensive analysis on {len(experiments)} prompts...")
    
    # Common generator for Sensitivity comparison
    gen_common = torch.Generator(device=device).manual_seed(42)
    height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    width = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    
    # Pre-generate noise for sensitivity test (same noise for base and permuted)
    base_latents = torch.randn((1, pipeline.unet.config.in_channels, height // 8, width // 8), device=device, generator=gen_common, dtype=pipeline.unet.dtype)
    t = torch.tensor([999], device=device).long()

    # Pre-compute Uncond (since latents are fixed)
    tokens_uncond = pipeline.tokenizer("", padding="max_length", max_length=pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeds_uncond = pipeline.text_encoder(tokens_uncond.input_ids.to(device))[0].to(dtype=pipeline.unet.dtype)
        vec_uncond = pipeline.unet(base_latents, t, encoder_hidden_states=embeds_uncond).sample

    for prompt, p_type in tqdm(experiments):
        try:
            # A. Normalized Sensitivity (Permuted vs Uncond)
            # Permuted
            perm_text = permute_words(prompt)
            tokens_perm = pipeline.tokenizer(perm_text, padding="max_length", max_length=pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            with torch.no_grad():
                embeds_perm = pipeline.text_encoder(tokens_perm.input_ids.to(device))[0].to(dtype=pipeline.unet.dtype)
                vec_perm = pipeline.unet(base_latents, t, encoder_hidden_states=embeds_perm).sample
                
            metrics_sens = analyze_vectors_normalized(vec_uncond.detach().cpu(), vec_perm.detach().cpu())
            
            # B. Jacobian Norm (Normalized)
            jac_norm, rel_jac_norm = get_jacobian_norm_normalized(pipeline, prompt, num_projections=1)
            
            # C. Set Spread (Normalized)
            spread, avg_norm, rel_spread = analyze_output_set_properties_normalized(pipeline, prompt, num_variations=10)
            
            # Aggregate Results
            row = {
                "Prompt": prompt,
                "Type": p_type,
                "JacobianNorm": jac_norm,
                "RelativeJacobian": rel_jac_norm,
                "SetSpread": spread,
                "RelativeSpread": rel_spread  # This is the Coeff of Variation
            }
            # Add sensitivity metrics (CosSim, RelL2, RelL1...)
            row.update(metrics_sens)
            
            results.append(row)
            
        except Exception as e:
            # print(f"Skipping {prompt[:20]}: {e}")
            continue

    # 4. Save Results
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "sensitivity_metrics_normalized.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # 5. Generate Plots
    generate_plots(results_df)

def generate_plots(df):
    # Metrics to plot: Focus on Normalized ones for fair comparison
    plot_targets = {
        "CosSim": "Cosine Similarity",
        "Relative_L2": "Relative L2 Distance (Norm Normalized)",
        "Relative_L1": "Relative L1 Distance (Norm Normalized)",
        "RelativeJacobian": "Relative Jacobian Norm (Sensitivity%)",
        "RelativeSpread": "Relative Set Spread (Coeff of Variation)"
    }
    
    # Filter cleanup
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=plot_targets.keys())
    
    mem_df = df[df["Type"] == "Memorized"]
    unmem_df = df[df["Type"] == "Unmemorized"]
    
    print("Generating normalized comparison plots...")
    
    for metric, title in plot_targets.items():
        # 1. Histogram
        plt.figure(figsize=(10, 6))
        # Use common bins range
        # remove outliers for plotting range if needed, or just auto
        plt.hist(mem_df[metric], bins=30, alpha=0.5, label='Memorized', density=True, color='red')
        plt.hist(unmem_df[metric], bins=30, alpha=0.5, label='Unmemorized', density=True, color='blue')
        
        plt.title(f"Distribution of {title}", fontsize=14)
        plt.xlabel(title, fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(RESULTS_DIR, f"hist_{metric}.png"))
        plt.close()
        
        # 2. Boxplot
        plt.figure(figsize=(8, 6))
        plt.boxplot([mem_df[metric], unmem_df[metric]], labels=['Memorized', 'Unmemorized'], patch_artist=True)
        plt.title(f"Comparison of {title}", fontsize=14)
        plt.ylabel(title, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(RESULTS_DIR, f"boxplot_{metric}.png"))
        plt.close()
        
    print("Plots saved.")

if __name__ == "__main__":
    main()
