
import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from diffusers import DDIMScheduler
from local_sd_pipeline import LocalStableDiffusionPipeline
import nltk
from nltk.corpus import wordnet

# Configuration
RESULTS_DIR = "results/compare_exp_cond_and_uncond"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
NUM_PERTURBATIONS = 100  # Number of perturbations per prompt
NUM_PROMPTS = 100        # Number of prompts per group (Mem/Unmem)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def download_nltk_resources():
    try:
        nltk.data.find('corpora/wordnet.zip')
        nltk.data.find('tokenizers/punkt.zip')
        nltk.data.find('taggers/averaged_perceptron_tagger.zip')
    except LookupError:
        print("Downloading NLTK resources...")
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

def get_synonym(word):
    """Get a synonym for a word using WordNet."""
    synonyms = []
    try:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                name = lemma.name().lower()
                if name != word.lower() and '_' not in name:
                    synonyms.append(name)
    except Exception:
        pass
    
    if synonyms:
        return random.choice(list(set(synonyms)))
    return word

def permute_prompt_shuffle(text):
    """Randomly shuffle words in the text."""
    words = text.split()
    if len(words) > 1:
        random.shuffle(words)
    return " ".join(words)

def permute_prompt_synonym(text):
    """Replace nouns with synonyms."""
    try:
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        new_words = []
        for word, tag in tagged:
            if tag.startswith('NN'): # Noun
                if random.random() < 0.5: # 50% chance
                    new_words.append(get_synonym(word))
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        return " ".join(new_words)
    except Exception:
        return text

def get_noise_pred(pipeline, latents, t, encoder_hidden_states):
    """Get noise prediction from UNet."""
    with torch.no_grad():
        noise_pred = pipeline.unet(latents, t, encoder_hidden_states=encoder_hidden_states).sample
    return noise_pred

def compute_metrics(target_noise, ref_noise):
    """
    Compute similarity metrics between target noise and reference noise.
    target_noise: The noise vector to compare (e.g., Conditional Expectation)
    ref_noise: The reference noise vector (e.g., Unconditional Noise)
    """
    v1 = ref_noise.flatten().float()
    v2 = target_noise.flatten().float()
    
    cos_sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
    l2_diff = torch.norm(v1 - v2, p=2).item()
    l1_diff = torch.norm(v1 - v2, p=1).item()
    max_diff = torch.max(torch.abs(v1 - v2)).item()
    
    return cos_sim, l2_diff, l1_diff, max_diff

def main():
    set_seed(SEED)
    download_nltk_resources()
    print(f"Using device: {DEVICE}")

    # ==========================================
    # 1. Load Data
    # ==========================================
    # Memorized Prompts
    try:
        df = pd.read_csv('prompts/memorized_laion_prompts.csv', sep=';')
        memorized_prompts = df['Caption'].tolist()
        print(f"Loaded {len(memorized_prompts)} memorized prompts.")
    except Exception as e:
        print(f"Error loading memorized prompts: {e}")
        return

    # Unmemorized Prompts (Hardcoded for stability)
    unmemorized_prompts = [
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
    unmemorized_prompts = list(set(unmemorized_prompts))
    
    # Randomly sample to ensure balanced classes
    random.shuffle(memorized_prompts)
    random.shuffle(unmemorized_prompts)
    
    memorized_samples = memorized_prompts[:NUM_PROMPTS]
    unmemorized_samples = unmemorized_prompts[:NUM_PROMPTS]
    
    print(f"Selected {len(memorized_samples)} Memorized and {len(unmemorized_samples)} Unmemorized prompts.")

    # ==========================================
    # 2. Load Model & Setup
    # ==========================================
    print(f"Loading model: {MODEL_ID}")
    pipeline = LocalStableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        requires_safety_checker=False
    ).to(DEVICE)
    pipeline.set_progress_bar_config(disable=True)

    # Prepare common latents
    height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    width = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    g_cpu = torch.Generator(device=DEVICE).manual_seed(SEED)
    latents = torch.randn(
        (1, pipeline.unet.config.in_channels, height // 8, width // 8),
        device=DEVICE, generator=g_cpu, dtype=pipeline.unet.dtype
    )
    t = torch.tensor([999], device=DEVICE).long() # High noise level

    # ==========================================
    # 3. Compute Unconditional Noise
    # ==========================================
    print("Computing Unconditional Noise...")
    tokens_uncond = pipeline.tokenizer("", padding="max_length", max_length=pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeds_uncond = pipeline.text_encoder(tokens_uncond.input_ids.to(DEVICE))[0].to(dtype=pipeline.unet.dtype)
        noise_uncond = get_noise_pred(pipeline, latents, t, embeds_uncond)

    # ==========================================
    # 4. Global Expectation Analysis
    # (Do different prompts average to Uncond?)
    # ==========================================
    print("\n[Analysis 1] Global Expectation: Average of MANY different prompts vs Uncond")
    
    global_results = []
    
    def compute_global_average_noise(prompt_list, label):
        noise_preds = []
        for p in tqdm(prompt_list, desc=f"Global {label}"):
            tok = pipeline.tokenizer(p, padding="max_length", max_length=pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            with torch.no_grad():
                emb = pipeline.text_encoder(tok.input_ids.to(DEVICE))[0].to(dtype=pipeline.unet.dtype)
                noise_preds.append(get_noise_pred(pipeline, latents, t, emb))
        
        avg_noise = torch.stack(noise_preds).mean(dim=0)
        metrics = compute_metrics(avg_noise, noise_uncond)
        return metrics

    mem_global_metrics = compute_global_average_noise(memorized_samples, "Memorized")
    unmem_global_metrics = compute_global_average_noise(unmemorized_samples, "Unmemorized")
    
    global_results.append({"Group": "Memorized", "CosSim": mem_global_metrics[0], "L2": mem_global_metrics[1], "L1": mem_global_metrics[2], "MaxDiff": mem_global_metrics[3]})
    global_results.append({"Group": "Unmemorized", "CosSim": unmem_global_metrics[0], "L2": unmem_global_metrics[1], "L1": unmem_global_metrics[2], "MaxDiff": unmem_global_metrics[3]})
    
    df_global = pd.DataFrame(global_results)
    print("\nGlobal Expectation Results:")
    print(df_global)
    df_global.to_csv(os.path.join(RESULTS_DIR, "global_expectation_vs_uncond.csv"), index=False)

    # ==========================================
    # 5. Local Expectation Analysis (Perturbation)
    # (Does local perturbation average to Uncond?)
    # ==========================================
    print("\n[Analysis 2] Local Expectation: Average of perturbing ONE prompt vs Uncond")
    
    local_results = []
    prompt_dataset = [(p, "Memorized") for p in memorized_samples] + [(p, "Unmemorized") for p in unmemorized_samples]
    
    # To save time if needed, we can sample a subset, but user asked for 100 each.
    # We proceed with full set.
    
    for prompt, p_type in tqdm(prompt_dataset, desc="Local Perturbation"):
        # 0. Base Embedding
        tokens_base = pipeline.tokenizer(prompt, padding="max_length", max_length=pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeds_base = pipeline.text_encoder(tokens_base.input_ids.to(DEVICE))[0].to(dtype=pipeline.unet.dtype)
            
        # Method 1: Shuffle
        noise_preds_shuffle = []
        for _ in range(NUM_PERTURBATIONS):
            p_shuf = permute_prompt_shuffle(prompt)
            tok = pipeline.tokenizer(p_shuf, padding="max_length", max_length=pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            with torch.no_grad():
                emb = pipeline.text_encoder(tok.input_ids.to(DEVICE))[0].to(dtype=pipeline.unet.dtype)
                noise_preds_shuffle.append(get_noise_pred(pipeline, latents, t, emb))
        avg_noise_shuffle = torch.stack(noise_preds_shuffle).mean(dim=0)
        m_shuffle = compute_metrics(avg_noise_shuffle, noise_uncond)
        
        # Method 2: Synonym
        noise_preds_syn = []
        for _ in range(NUM_PERTURBATIONS):
            p_syn = permute_prompt_synonym(prompt)
            tok = pipeline.tokenizer(p_syn, padding="max_length", max_length=pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            with torch.no_grad():
                emb = pipeline.text_encoder(tok.input_ids.to(DEVICE))[0].to(dtype=pipeline.unet.dtype)
                noise_preds_syn.append(get_noise_pred(pipeline, latents, t, emb))
        avg_noise_syn = torch.stack(noise_preds_syn).mean(dim=0)
        m_syn = compute_metrics(avg_noise_syn, noise_uncond)
        
        # Method 3: Embedding noise
        noise_preds_emb = []
        for _ in range(NUM_PERTURBATIONS):
            noise = torch.randn_like(embeds_base) * 0.1
            emb_noisy = embeds_base + noise
            noise_preds_emb.append(get_noise_pred(pipeline, latents, t, emb_noisy))
        avg_noise_emb = torch.stack(noise_preds_emb).mean(dim=0)
        m_emb = compute_metrics(avg_noise_emb, noise_uncond)
        
        local_results.append({
            "Prompt": prompt,
            "Type": p_type,
            "Shuffle_CosSim": m_shuffle[0], "Shuffle_L2": m_shuffle[1], "Shuffle_L1": m_shuffle[2], "Shuffle_MaxDiff": m_shuffle[3],
            "Synonym_CosSim": m_syn[0], "Synonym_L2": m_syn[1], "Synonym_L1": m_syn[2], "Synonym_MaxDiff": m_syn[3],
            "EmbNoise_CosSim": m_emb[0], "EmbNoise_L2": m_emb[1], "EmbNoise_L1": m_emb[2], "EmbNoise_MaxDiff": m_emb[3],
        })
        
    df_local = pd.DataFrame(local_results)
    df_local.to_csv(os.path.join(RESULTS_DIR, "local_expectation_vs_uncond.csv"), index=False)
    print("Local results saved.")
    
    # ==========================================
    # 6. Plotting
    # ==========================================
    plot_results(df_global, df_local)

def plot_results(df_global, df_local):
    """Generate visualizations for both global and local analyses."""
    sns.set_theme(style="whitegrid")
    
    # --- Plot 1: Global Average Similarity ---
    # Bar chart comparing Mem/Unmem global average distance to Uncond
    metrics = ["CosSim", "L2", "L1", "MaxDiff"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, metric in enumerate(metrics):
        sns.barplot(x="Group", y=metric, data=df_global, ax=axes[i], palette="viridis")
        axes[i].set_title(f"Global Avg Noise vs Uncond\n({metric})")
        axes[i].set_ylabel(metric)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "global_expectation_comparison.png"))
    plt.close()
    
    # --- Plot 2: Local Perturbation Distributions ---
    # Boxplots showing distribution of distances for each method
    methods = ["Shuffle", "Synonym", "EmbNoise"]
    
    for method in methods:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        for i, metric in enumerate(metrics):
            col_name = f"{method}_{metric}"
            sns.boxplot(x="Type", y=col_name, data=df_local, ax=axes[i], palette="Set2")
            axes[i].set_title(f"{method} Expectation vs Uncond\n({metric})")
            axes[i].set_ylabel(metric)
            
        plt.suptitle(f"Local Perturbation Method: {method}", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"local_expectation_distribution_{method}.png"))
        plt.close()

if __name__ == "__main__":
    main()
