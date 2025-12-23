import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from diffusers import DDIMScheduler
from local_sd_pipeline import LocalStableDiffusionPipeline
import nltk
from nltk.corpus import wordnet

# Ensure results directory exists
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

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
                # Avoid same word and multi-word synonyms for simplicity
                if lemma.name().lower() != word.lower() and '_' not in lemma.name():
                    synonyms.append(lemma.name())
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
                if random.random() < 0.5: # 50% chance to replace
                    new_words.append(get_synonym(word))
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        # Detokenize simply
        return " ".join(new_words)
    except Exception as e:
        # Fallback if NLTK fails or not setup
        # print(f"Synonym failed: {e}")
        return text

def get_noise_pred(pipeline, latents, t, encoder_hidden_states):
    """Get noise prediction from UNet."""
    with torch.no_grad():
        noise_pred = pipeline.unet(latents, t, encoder_hidden_states=encoder_hidden_states).sample
    return noise_pred

def compute_metrics(target_noise, ref_noise):
    """
    Compute Cos Sim, L2, L1, Max Diff between target (Average Conditional) and reference (Unconditional).
    """
    v1 = ref_noise.flatten().float()    # Unconditional
    v2 = target_noise.flatten().float() # Average Conditional
    
    cos_sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
    l2_diff = torch.norm(v1 - v2, p=2).item()
    l1_diff = torch.norm(v1 - v2, p=1).item()
    max_diff = torch.max(torch.abs(v1 - v2)).item()
    
    return cos_sim, l2_diff, l1_diff, max_diff

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    download_nltk_resources()
    
    print(f"Using device: {device}")
    
    # 1. Load Data
    # Memorized
    try:
        df = pd.read_csv('prompts/memorized_laion_prompts.csv', sep=';')
        memorized_prompts = df['Caption'].tolist()
        print(f"Loaded {len(memorized_prompts)} memorized prompts.")
    except Exception as e:
        print(f"Error loading memorized prompts: {e}")
        return

    # Unmemorized (Expanded List from prompt_sensitivity.py)
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
    unmemorized_prompts = list(set(unmemorized_prompts)) # Unique
    
    # Balance Counts: 100 vs 100
    target_count = 100
    random.shuffle(memorized_prompts)
    memorized_subset = memorized_prompts[:target_count]
    
    random.shuffle(unmemorized_prompts)
    unmemorized_subset = unmemorized_prompts[:target_count]
    
    # Ensure equal if less than 100
    real_min = min(len(memorized_subset), len(unmemorized_subset))
    memorized_subset = memorized_subset[:real_min]
    unmemorized_subset = unmemorized_subset[:real_min]
    
    print(f"Balanced Analysis: {len(memorized_subset)} Memorized vs {len(unmemorized_subset)} Unmemorized prompts.")
    
    # 2. Load Model
    model_id = "CompVis/stable-diffusion-v1-4"
    print(f"Loading model: {model_id}...")
    pipeline = LocalStableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    print("Model loaded.")
    
    # Common variables
    height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    width = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    # Fixed noise for fair comparison
    gen_common = torch.Generator(device=device).manual_seed(42)
    latents = torch.randn((1, pipeline.unet.config.in_channels, height // 8, width // 8), device=device, generator=gen_common, dtype=pipeline.unet.dtype)
    t = torch.tensor([999], device=device).long()
    
    # 3. Compute Unconditional Noise
    print("Computing Unconditional Noise...")
    tokens_uncond = pipeline.tokenizer("", padding="max_length", max_length=pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeds_uncond = pipeline.text_encoder(tokens_uncond.input_ids.to(device))[0].to(dtype=pipeline.unet.dtype)
        # Uncond noise for the SAME latent
        noise_uncond = get_noise_pred(pipeline, latents, t, embeds_uncond)
        
    results = []
    
    experiments = []
    for p in memorized_subset:
        experiments.append((p, "Memorized"))
    for p in unmemorized_subset:
        experiments.append((p, "Unmemorized"))
        
    print(f"Starting analysis on {len(experiments)} prompts with 100 perturbations each...")
    
    for prompt, p_type in tqdm(experiments):
        try:
            # Base Embedding (for Embedding Perturbation)
            tokens_base = pipeline.tokenizer(prompt, padding="max_length", max_length=pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            with torch.no_grad():
                embeds_base = pipeline.text_encoder(tokens_base.input_ids.to(device))[0].to(dtype=pipeline.unet.dtype)
            
            # --- Experiment 1: Prompt Perturbation (Shuffle) ---
            # Generate 100 shuffled prompts -> Average Conditional Noise
            noise_preds_shuffle = []
            for _ in range(100):
                p_shuf = permute_prompt_shuffle(prompt)
                tok = pipeline.tokenizer(p_shuf, padding="max_length", max_length=pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    emb = pipeline.text_encoder(tok.input_ids.to(device))[0].to(dtype=pipeline.unet.dtype)
                    noise_preds_shuffle.append(get_noise_pred(pipeline, latents, t, emb))
            
            avg_noise_shuffle = torch.stack(noise_preds_shuffle).mean(dim=0)
            metrics_shuffle = compute_metrics(avg_noise_shuffle, noise_uncond) # Target=AvgPred, Ref=Uncond
            
            # --- Experiment 2: Prompt Perturbation (Synonym) ---
            noise_preds_syn = []
            for _ in range(100):
                p_syn = permute_prompt_synonym(prompt)
                tok = pipeline.tokenizer(p_syn, padding="max_length", max_length=pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    emb = pipeline.text_encoder(tok.input_ids.to(device))[0].to(dtype=pipeline.unet.dtype)
                    noise_preds_syn.append(get_noise_pred(pipeline, latents, t, emb))
            
            avg_noise_syn = torch.stack(noise_preds_syn).mean(dim=0)
            metrics_syn = compute_metrics(avg_noise_syn, noise_uncond)

            # --- Experiment 3: Text Embedding Perturbation (Gaussian) ---
            noise_preds_emb = []
            for _ in range(100):
                # Inject Gaussian noise into text embedding
                # using scale 0.1 as a heuristic for perturbation
                noise = torch.randn_like(embeds_base) * 0.1 
                emb_noisy = embeds_base + noise
                noise_preds_emb.append(get_noise_pred(pipeline, latents, t, emb_noisy))
            
            avg_noise_emb = torch.stack(noise_preds_emb).mean(dim=0)
            metrics_emb = compute_metrics(avg_noise_emb, noise_uncond)
            
            # Store results
            row = {
                "Prompt": prompt,
                "Type": p_type,
                # Shuffle
                "Shuffle_CosSim": metrics_shuffle[0],
                "Shuffle_L2": metrics_shuffle[1],
                "Shuffle_L1": metrics_shuffle[2],
                "Shuffle_MaxDiff": metrics_shuffle[3],
                # Synonym
                "Synonym_CosSim": metrics_syn[0],
                "Synonym_L2": metrics_syn[1],
                "Synonym_L1": metrics_syn[2],
                "Synonym_MaxDiff": metrics_syn[3],
                # Embedding Noise
                "EmbNoise_CosSim": metrics_emb[0],
                "EmbNoise_L2": metrics_emb[1],
                "EmbNoise_L1": metrics_emb[2],
                "EmbNoise_MaxDiff": metrics_emb[3],
            }
            results.append(row)
            
        except Exception as e:
            # print(f"Error processing {prompt[:20]}: {e}")
            continue

    # Save to CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "compare_condition_expectation_vs_uncond.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # 4. Plotting
    plot_results(results_df)

def plot_results(df):
    """Plot histograms comparing Memorized vs Unmemorized for each Metric and Method."""
    methods = ["Shuffle", "Synonym", "EmbNoise"]
    metrics = ["CosSim", "L2", "L1", "MaxDiff"]
    
    print("Generating plots...")
    
    for method in methods:
        for metric in metrics:
            col_name = f"{method}_{metric}"
            if col_name not in df.columns: continue
            
            plt.figure(figsize=(10, 6))
            
            mem_data = df[df["Type"] == "Memorized"][col_name]
            unmem_data = df[df["Type"] == "Unmemorized"][col_name]
            
            # Histogram
            plt.hist(mem_data, bins=30, alpha=0.5, label='Memorized', density=True, color='red')
            plt.hist(unmem_data, bins=30, alpha=0.5, label='Unmemorized', density=True, color='blue')
            
            plt.title(f"{method} Perturbation: {metric} (Avg Cond vs Uncond)", fontsize=14)
            plt.xlabel(metric, fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            filename = f"hist_avg_cond_vs_uncond_{method}_{metric}.png"
            plt.savefig(os.path.join(RESULTS_DIR, filename))
            plt.close()
            
    # Also Boxplots for summary
    for method in methods:
        plt.figure(figsize=(12, 8))
        # Create subplots for 4 metrics
        for i, metric in enumerate(metrics, 1):
            col_name = f"{method}_{metric}"
            plt.subplot(2, 2, i)
            data_to_plot = [df[df["Type"] == "Memorized"][col_name], df[df["Type"] == "Unmemorized"][col_name]]
            plt.boxplot(data_to_plot, labels=['Memorized', 'Unmemorized'], patch_artist=True)
            plt.title(metric)
            plt.grid(True, alpha=0.3)
        
        plt.suptitle(f"{method} Perturbation Summary", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"boxplot_avg_cond_vs_uncond_{method}.png"))
        plt.close()

    print("Plots saved.")

if __name__ == "__main__":
    main()
