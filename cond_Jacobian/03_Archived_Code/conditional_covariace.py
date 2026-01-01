
import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from local_sd_pipeline import LocalStableDiffusionPipeline
from tqdm import tqdm

# Ensure results directory exists
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_balanced_prompts(n_samples=50):
    # 1. Memorized Prompts
    try:
        df = pd.read_csv('prompts/memorized_laion_prompts.csv', sep=';')
        memorized_all = df['Caption'].tolist()
    except FileNotFoundError:
        print("Warning: memorized_laion_prompts.csv not found. Using placeholders.")
        memorized_all = [f"Memorized prompt placeholder {i}" for i in range(100)]
    
    # 2. Unmemorized Prompts (Hardcoded extended list)
    unmemorized_all = [
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
        "A vintage camera sitting on a stack of old books"
    ]
    
    # Shuffle and Sample
    random.shuffle(memorized_all)
    random.shuffle(unmemorized_all)
    
    # Ensure equal size
    limit = min(len(memorized_all), len(unmemorized_all), n_samples)
    
    return memorized_all[:limit], unmemorized_all[:limit]

def analyze_population_covariance(prompts, pipe, latents, t, device, tag="Group"):
    """
    Computes the eigenvalues of the covariance matrix of the set of prompts.
    X = [v_1, v_2, ..., v_N] where v_i = E(c_i) - E(empty)
    Cov ~ X_centered * X_centered^T
    """
    print(f"[{tag}] Collecting vectors for {len(prompts)} prompts...")
    dtype = pipe.unet.dtype
    
    # 1. Compute Fixed Unconditional Reference
    tokens_uncond = pipe.tokenizer("", padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeds_uncond = pipe.text_encoder(tokens_uncond.input_ids.to(device))[0].to(dtype=dtype)
        noise_pred_uncond = pipe.unet(latents, t, encoder_hidden_states=embeds_uncond).sample
    vec_uncond = noise_pred_uncond.flatten().float().cpu() # Move to CPU
    
    # 2. Collect Conditional Vectors
    vectors = []
    
    for prompt in tqdm(prompts):
        text_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeds = pipe.text_encoder(text_input.input_ids.to(device))[0].to(dtype=dtype)
            noise_pred = pipe.unet(latents, t, encoder_hidden_states=embeds).sample
            
        v = noise_pred.flatten().float().cpu() # Move to CPU
        diff = v - vec_uncond
        vectors.append(diff)
        
    # 3. Construct Matrix X [D, N]
    # D ~ 16384, N ~ 50
    X = torch.stack(vectors).T # [D, N]
    
    # Center the matrix (Subtract Mean Vector of the population)
    # Because we want Covariance: E[(v - mu)(v - mu)^T]
    mu = X.mean(dim=1, keepdim=True)
    X_centered = X - mu
    
    # 4. SVD eigenvalues
    # Covariance C = (1 / (N-1)) * X_centered * X_centered^T
    # Singular values 's' of X_centered satisfy: Eigenvalues of C = s^2 / (N-1)
    
    print(f"[{tag}] Doing SVD on shape {X_centered.shape}...")
    try:
        # Move to GPU for SVD if possible (usually faster)
        # Check if D*N fits in memory. 16k * 50 * 4 bytes is small (~3MB).
        X_gpu = X_centered.to(device)
        U, S, Vh = torch.linalg.svd(X_gpu, full_matrices=False)
        singular_values = S.cpu()
        del X_gpu, U, Vh
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"GPU SVD failed ({e}), falling back to CPU...")
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        singular_values = S
        
    N = X.shape[1]
    eigenvalues = (singular_values ** 2) / (N - 1)
    
    # 5. Summary Metrics
    max_eig = eigenvalues[0].item()
    trace_eig = eigenvalues.sum().item()
    
    # effective rank (number of eigenvalues that explain 95% variance)
    total_var = eigenvalues.sum()
    cumsum = torch.cumsum(eigenvalues, dim=0)
    explained_ratio = cumsum / total_var
    eff_rank_95 = (explained_ratio < 0.95).sum().item() + 1
    
    print(f"[{tag}] Results:")
    print(f"  Max Eigenvalue:   {max_eig:.4f}")
    print(f"  Trace (Sum):      {trace_eig:.4f}")
    print(f"  Eff. Rank (95%):  {eff_rank_95}/{N}")
    
    return eigenvalues.numpy()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    print(f"Using device: {device}")
    
    # Load Model
    model_id = "CompVis/stable-diffusion-v1-4"
    print(f"Loading model: {model_id}...")
    pipeline = LocalStableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    print("Model loaded.")
    
    # Load Balanced Prompt Sets
    mem_prompts, unmem_prompts = get_balanced_prompts(n_samples=50)
    print(f"Analyzing {len(mem_prompts)} Memorized vs {len(unmem_prompts)} Unmemorized prompts.")
    
    # Prepare Physics
    height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    width = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    gen = torch.Generator(device=device).manual_seed(42)
    latents = torch.randn((1, pipeline.unet.config.in_channels, height // 8, width // 8), device=device, generator=gen, dtype=pipeline.unet.dtype)
    t = torch.tensor([999], device=device).long()
    
    # Analyze
    eigs_mem = analyze_population_covariance(mem_prompts, pipeline, latents, t, device, tag="Memorized")
    eigs_unmem = analyze_population_covariance(unmem_prompts, pipeline, latents, t, device, tag="Unmemorized")
    
    # Visualization (Spectrum Plot)
    plt.figure(figsize=(10, 6))
    plt.plot(eigs_mem, label='Memorized', marker='o', markersize=3, alpha=0.7)
    plt.plot(eigs_unmem, label='Unmemorized', marker='x', markersize=3, alpha=0.7)
    plt.yscale('log')
    plt.title('Covariance Eigenvalue Spectrum (Population Analysis)')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue (log scale)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    out_path = os.path.join(RESULTS_DIR, "population_covariance_spectrum.png")
    plt.savefig(out_path)
    print(f"Spectrum plot saved to {out_path}")

    # CSV Export
    df = pd.DataFrame({
        "Index": range(len(eigs_mem)),
        "Eigenvalue_Mem": eigs_mem,
        "Eigenvalue_Unmem": eigs_unmem if len(eigs_unmem) == len(eigs_mem) else np.pad(eigs_unmem, (0, len(eigs_mem)-len(eigs_unmem)), constant_values=np.nan)
    })
    df.to_csv(os.path.join(RESULTS_DIR, "covariance_spectrum.csv"), index=False)

if __name__ == "__main__":
    main()
