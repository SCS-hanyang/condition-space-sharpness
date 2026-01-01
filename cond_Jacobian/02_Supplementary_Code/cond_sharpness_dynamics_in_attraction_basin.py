
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from diffusers import DDIMScheduler
from local_sd_pipeline import LocalStableDiffusionPipeline
import random

# Configuration
RESULTS_DIR = "results/cond_sharpness_dynamics_in_attraction_basin"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
NUM_STEPS = 50

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_prompts():
    # Memorized
    try:
        df = pd.read_csv('prompts/memorized_laion_prompts.csv', sep=';')
        memorized = df['Caption'].tolist()[:10]
    except Exception as e:
        print(f"Error loading memorized prompts: {e}")
        memorized = []

    # Unmemorized (Hardcoded from prompt_sensitivity.ipynb / compare_exp_cond_and_uncond.py)
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
    unmemorized = list(set(unmemorized))[:10]
    return memorized, unmemorized

def compute_jacobian_norm(unet, latents, t, prompt_embeds, num_projections=3):
    """
    Approximates the Frobenius norm of the Jacobian of the unconditional noise prediction
    with respect to the prompt embeddings using Hutchinson's estimator.
    """
    # Detach to make it a leaf variable
    prompt_embeds = prompt_embeds.detach()
    
    with torch.enable_grad():
        prompt_embeds.requires_grad_(True)
        
        total_sq_norm = 0.0
        
        # Hutchinson estimator
        for _ in range(num_projections):
            # Forward pass
            noise_pred = unet(latents, t, encoder_hidden_states=prompt_embeds).sample
            
            # Predict noise component associated with text (conditional part)
            # Actually, the unet outputs the noise residual. 
            # We want Jacobian of epsilon_cond w.r.t c.
            
            # Create random projection vector v
            v = torch.randn_like(noise_pred)
            
            # Compute v^T * epsilon
            v_dot_eps = torch.sum(noise_pred * v)
            
            # Compute gradient w.r.t prompt_embeds
            grads = torch.autograd.grad(v_dot_eps, prompt_embeds, create_graph=False)[0]
            
            # Accumulate squared norm
            total_sq_norm += torch.sum(grads ** 2).item()
            
        # No need to reset requires_grad for detached variable
        
        # Average and take sqrt
        # E[ ||grad(v^T f)||^2 ] = ||J||_F^2
        avg_sq_norm = total_sq_norm / num_projections
        return np.sqrt(avg_sq_norm)

def get_input_embeddings(pipeline, prompt, device):
    """
    Get the initial token embeddings (before Transformer layers) for a prompt.
    Returns: input_ids, input_embeddings (requires_grad=False initially)
    """
    # Tokenize
    text_inputs = pipeline.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    
    # Get Input Embeddings (from the embedding layer of the text encoder)
    # CLIPTextModel -> text_model -> embeddings -> token_embedding
    input_embeddings = pipeline.text_encoder.text_model.embeddings.token_embedding(input_ids)
    
    return input_ids, input_embeddings

def compute_jacobian_norm_IE(pipeline, latents, t, input_embeddings, input_ids, num_projections=3):
    """
    Approximates the Frobenius norm of the Jacobian of the unconditional noise prediction
    with respect to the Input Token Embeddings (IE) using Hutchinson's estimator.
    """
    # We need to forward pass through the Text Encoder (CLIP) first, then UNet
    # To do this, we need to manually call the text encoder with 'inputs_embeds' 
    # instead of 'input_ids' to allow gradient flow from input_embeddings.
    
    # Detach to make it a leaf variable
    input_embeddings = input_embeddings.detach()
    
    with torch.enable_grad():
        input_embeddings.requires_grad_(True)
        
        total_sq_norm = 0.0
        
        # We need the position embeddings as well, which are added in the text encoder.
        # But `text_model.embeddings(inputs_embeds=...)` handles this if we call it correctly.
        # However, huggingface CLIPTextModel `forward` takes `input_ids` usually. 
        # But it also accepts `inputs_embeds`.
        
        # Note: We assume classifier_free_guidance=False here for simplicity of Jacobian definintion,
        # or we look at the cond part only. The prompt we pass is the conditional one.
        
        for _ in range(num_projections):
            # 1. Pass through Text Encoder
            # We use the underlying text_model to pass inputs_embeds
            # We also need to handle position_ids if necessary, but usually it's auto-generated.
            
            # CLIPTextTransformer forward:
            # inputs_embeds -> + pos_embeds -> encoder -> final_layer_norm
            
            # NOTE: We can't just call pipeline.text_encoder(inputs_embeds=...) because 
            # the wrapper might expect input_ids for some checks.
            # Let's try calling it directly.
            
            # We need to construct the attention mask as well for padding tokens.
            # But here we used "max_length" padding, so we have padding tokens.
            # The tokenizer returns attention_mask usually.
            
            # Re-create attention mask since we don't have it passed in comfortably.
            # 1 for not padding, 0 for padding. 
            # In CLIP, padding token is 0 or 49407 (eot)? 
            # Actually pipeline.tokenizer.pad_token_id usually.
            
            bsz, seq_len = input_ids.shape
            attention_mask = torch.zeros((bsz, seq_len), device=input_ids.device)
            # Find where input_ids IS NOT padding. 
            # But wait, we want differentiation through embedding.
            # input_ids is constant. Mask is constant.
            
            # Assuming standard padding (0 or eos) behavior.
            # Let's create a proper mask from the original input_ids.
            # However, we only have embeddings here. But we passed input_ids as arg too.
            attention_mask = (input_ids != pipeline.tokenizer.pad_token_id).long()
            # Also CLIP usually uses causal mask? No, text encoder is bidirectional? 
            # Actually CLIP text encoder uses causal mask.
            # The diffusers pipeline relies on the model to handle it.
            
            # Manual Forward Pass through CLIP Text Encoder
            # 1. Embeddings (Word + Position)
            text_model = pipeline.text_encoder.text_model
            seq_len = input_ids.shape[1]
            position_ids = text_model.embeddings.position_ids[:, :seq_len]
            hidden_states = text_model.embeddings(inputs_embeds=input_embeddings, position_ids=position_ids)
            
            # 2. Causal Mask
            bsz = input_ids.shape[0]
            # Standard causal mask for CLIP (triangular)
            mask = torch.full((bsz, 1, seq_len, seq_len), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype)
            mask = torch.triu(mask, diagonal=1)
            
            # 3. Encoder
            encoder_outputs = text_model.encoder(
                hidden_states,
                attention_mask=None,
                causal_attention_mask=mask 
            )
            
            # 4. Final Layer Norm
            last_hidden_state = text_model.final_layer_norm(encoder_outputs[0])
            
            # In standard SD, we verify normally it uses penultimate or last hidden state.
            # SD 1.4 uses the last_hidden_state (but normalized?).
            # Actually SD 1.4 uses 'output_hidden_states=True' and takes [0]? 
            # No, standard CLIPTextModel returns BaseModelOutputWithPooling.
            # outputs[0] is last_hidden_state.
            # The pipeline usually does:
            # prompt_embeds = self.text_encoder(text_input_ids, output_hidden_states=True)
            # prompt_embeds = prompt_embeds[0]
            
            # However, we need to be careful about the normalization. 
            # For CLIP, the text_model(inputs_embeds=...) output IS the last hidden state 
            # BEFORE the final projection (which SD doesn't use) or AFTER?
            # SD uses the hidden states from the text model directly.
            
            prompt_embeds = last_hidden_state
            
            # Normalization (LayerNorm) is essentially done inside text_model forward. 
            # So `last_hidden_state` is ready for UNet.
            
            # 2. Forward UNet
            noise_pred = pipeline.unet(latents, t, encoder_hidden_states=prompt_embeds).sample
            
            # 3. Projection
            v = torch.randn_like(noise_pred)
            v_dot_eps = torch.sum(noise_pred * v)
            
            # 4. Grad w.r.t Input Embeddings
            grads = torch.autograd.grad(v_dot_eps, input_embeddings, create_graph=False)[0]
            
            total_sq_norm += torch.sum(grads ** 2).item()
            
        # No need to reset requires_grad for detached variable
        return np.sqrt(total_sq_norm / num_projections)


def main():
    set_seed(SEED)
    
    print("Loading pipeline...")
    pipeline = LocalStableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32, # Use float32 for gradient accuracy
        requires_safety_checker=False
    ).to(DEVICE)
    pipeline.set_progress_bar_config(disable=True)
    
    memorized_prompts, unmemorized_prompts = load_prompts()
    print(f"Memorized: {len(memorized_prompts)}, Unmemorized: {len(unmemorized_prompts)}")
    
    # 1. Generate Unconditional Trajectory (Fixed Seed)
    print("Generating Unconditional Trajectory...")
    
    unc_prompt_embeds = pipeline._encode_prompt("", DEVICE, 1, False, None)
    
    # Prepare Latents
    height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    width = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    
    pipeline.scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    timesteps = pipeline.scheduler.timesteps
    
    latents = pipeline.prepare_latents(
        1, pipeline.unet.config.in_channels, height, width,
        torch.float32, torch.device(DEVICE), torch.Generator(device=DEVICE).manual_seed(SEED), None
    )
    
    # Store trajectory
    trajectory_latents = [] # List of latents at each step input
    trajectory_uncond_noise = []
    
    # Run Unconditional Loop
    curr_latents = latents
    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps, desc="Uncond Trajectory")):
            trajectory_latents.append(curr_latents.clone())
            
            # Predict noise
            latent_model_input = pipeline.scheduler.scale_model_input(curr_latents, t)
            noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=unc_prompt_embeds).sample
            
            trajectory_uncond_noise.append(noise_pred.clone())
            
            # Step (Unconditional)
            curr_latents = pipeline.scheduler.step(noise_pred, t, curr_latents, return_dict=False)[0]
            
    
    print("Unconditional Trajectory Generated.")
    
    # 2. Analyze Prompts
    results = []
    
    for group_name, prompts in [("Memorized", memorized_prompts), ("Unmemorized", unmemorized_prompts)]:
        print(f"Analyzing {group_name}...")
        
        curves = []
        
        # Batch processing or single? Single for safety with grad.
        # First pass: Calculate curves to find basin drop.
        # Ideally we do this efficiently.
        
        for prompt in tqdm(prompts, desc=f"{group_name} Curves"):
            # Encode prompt
            cond_embeds = pipeline._encode_prompt(prompt, DEVICE, 1, False, None)
            
            diff_norms = []
            
            with torch.no_grad():
                for i, t in enumerate(timesteps):
                    lat = trajectory_latents[i]
                    latent_input = pipeline.scheduler.scale_model_input(lat, t)
                    
                    # Cond Noise
                    cond_noise = pipeline.unet(latent_input, t, encoder_hidden_states=cond_embeds).sample
                    uncond_noise = trajectory_uncond_noise[i]
                    
                    diff = torch.norm(cond_noise - uncond_noise).item()
                    diff_norms.append(diff)
            
            curves.append(diff_norms)
            
        # Convert to array
        curves = np.array(curves)
        
        # Detect basins
        # Heuristic: Find the point of steepest drop after the initial rise?
        # Or simply argmin of gradient (most negative derivative)
        # We expect a drop from high (~10) to low (~2).
        
        final_data = []
        
        # Calculate avg drop index for memorized to use for unmemorized if needed
        # But we process groups sequentially.
        
        drop_indices = []
        
        for i, curve in enumerate(curves):
            # Simple derivative
            deriv = np.diff(curve)
            # Find steepest drop (min derivative)
            # Ignore first few steps (warmup)
            t_drop_idx = np.argmin(deriv[5:]) + 5 # Offset 5
            drop_indices.append(t_drop_idx)
            
        avg_drop_idx = int(np.mean(drop_indices))
        if group_name == "Unmemorized":
            # For Unmemorized, user says: "if not found, use avg of memorized".
            # Unmemorized curves are usually flat or smooth. 
            # We will just force use the Memorized Average IF the drop is not significant?
            # Or just use the Memorized Average for ALL unmemorized (as a baseline comparison).
            # The prompt says: "memorized prompt에서 찾은 basin들의 평균으로 계산해라" (calculate using average of basins found in memorized prompts).
            # So for Unmemorized, we override `t_start`
            # I need the avg from memorized. I should store it.
            pass
        else:
            shared_avg_drop_idx = avg_drop_idx
            
        target_indices_list = []
        
        if group_name == "Unmemorized":
            # Use shared avg from Memorized
            use_indices = [shared_avg_drop_idx] * len(prompts)
        else:
             # Use individual indices for Memorized (or shared avg? Instruction says "Find attraction basin... using same method... if not found use average". Memorized should have it).
             # I will use individual for Memorized.
             use_indices = drop_indices

        # 3. Jacobian Calculation
        print(f"Calculating Jacobians for {group_name}...")
        
        for i, prompt in enumerate(tqdm(prompts, desc="Jacobians")):
            drop_idx = use_indices[i]
            
            # Define indices to calculate
            # Inside: 0 to drop_idx (inclusive)
            # Outside: drop_idx + 1 to drop_idx + 4 (inclusive)
            
            inside_indices = list(range(0, drop_idx + 1))
            outside_indices = list(range(drop_idx + 1, min(drop_idx + 5, NUM_STEPS)))
            
            calc_indices = inside_indices + outside_indices
            
            # Obtain initial embeddings for IE calculation (reuse for all steps)
            input_ids, input_embeddings = get_input_embeddings(pipeline, prompt, DEVICE)
            prompt_embeds_te = pipeline._encode_prompt(prompt, DEVICE, 1, False, None) # For TE
            
            for t_idx in calc_indices:
                t = timesteps[t_idx]
                lat = trajectory_latents[t_idx]
                latent_input = pipeline.scheduler.scale_model_input(lat, t)
                
                # A. Stanard Jacobian (TE)
                j_norm_te = compute_jacobian_norm(pipeline.unet, latent_input, t, prompt_embeds_te)
                
                # B. Input Embedding Jacobian (IE)
                j_norm_ie = compute_jacobian_norm_IE(pipeline, latent_input, t, input_embeddings, input_ids)
                
                # Region tag
                region = "Inside Basin" if t_idx <= drop_idx else "Outside Basin"
                
                results.append({
                    "Prompt": prompt,
                    "Group": group_name,
                    "Drop_Index": drop_idx,
                    "Step_Index": t_idx,
                    "Relative_Step": t_idx - drop_idx,
                    "Region": region,
                    "J_Norm_TE": j_norm_te,
                    "J_Norm_IE": j_norm_ie,
                    "Diff_Curve": curves[i].tolist() # Store for plotting specific ones
                })
            
    # Save Results
    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{RESULTS_DIR}/jacobian_analysis.csv", index=False)
    
    # 4. Visualization
    # Plot 1: Boxplot of Jacobian Change (Prev vs Post)
    # We want to show "Change". Maybe (Post - Prev) or just distribution.
    
    # 4. Visualization
    # Plot 1: Dynamics of Jacobian Norm (Line Plot)
    
    plt.figure(figsize=(10, 6))
    
    # Melt? No, it's already in long format but split by J_Norm_TE and J_Norm_IE columns.
    # Let's melt specifically for the plotting if we want to facet by TE/IE.
    
    # We want columns: Group, Region, Relative_Step, J_Norm, Metric
    # Current: Group, Step, Relative_Step, J_Norm_TE, J_Norm_IE
    
    df_plot_te = df_res.copy()
    df_plot_te["J_Norm"] = df_plot_te["J_Norm_TE"]
    df_plot_te["Metric"] = "CLIP Output"
    
    df_plot_ie = df_res.copy()
    df_plot_ie["J_Norm"] = df_plot_ie["J_Norm_IE"]
    df_plot_ie["Metric"] = "Input Embedding"
    
    df_plot = pd.concat([df_plot_te, df_plot_ie], ignore_index=True)
    
    # Visualize using catplot/pointplot isn't ideal for continuous time series?
    # Relative_Step is our x-axis.
    
    g = sns.relplot(
        data=df_plot, 
        x="Relative_Step", 
        y="J_Norm", 
        hue="Group", 
        col="Metric", 
        kind="line", 
        style="Group",
        markers=True,
        facet_kws={'sharey': False},
        height=6, 
        aspect=1.2
    )
    
    # Add vertical line at 0 (Drop Index)
    for ax in g.axes.flat:
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Attraction Basin Exit')
        
    g.fig.suptitle("Jacobian Frobenius Norm Dynamics relative to Basin Exit", y=1.02)
    plt.savefig(f"{RESULTS_DIR}/jacobian_dynamics_lineplot.png")
    plt.close()
    
    # Log Scale Version
    # We can just change the scale
    try:
        g = sns.relplot(
            data=df_plot, 
            x="Relative_Step", 
            y="J_Norm", 
            hue="Group", 
            col="Metric", 
            kind="line", 
            style="Group",
            markers=True,
            facet_kws={'sharey': False},
            height=6, 
            aspect=1.2
        )
        for ax in g.axes.flat:
            ax.set_yscale("log")
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        
        g.fig.suptitle("Jacobian Frobenius Norm Dynamics (Log Scale)", y=1.02)
        plt.savefig(f"{RESULTS_DIR}/jacobian_dynamics_lineplot_log.png")
        plt.close()
    except Exception as e:
        print(f"Log plot failed: {e}")

    
    # Plot 2: Specific Prompts Function
    def plot_specific_prompts(indices, save_name="specific_prompts"):
        plt.figure(figsize=(15, 5))
        
        # Extract unique memorized prompts from results
        mem_rows = [r for r in results if r["Group"] == "Memorized"]
        seen_prompts = set()
        unique_data = []
        
        for r in mem_rows:
            if r["Prompt"] not in seen_prompts:
                seen_prompts.add(r["Prompt"])
                unique_data.append(r)
        
        for k in indices:
            if k >= len(unique_data): continue
            res = unique_data[k]
            curve = res["Diff_Curve"]
            drop_idx = res["Drop_Index"]
            
            plt.plot(curve, label=f"Prompt {k}")
            plt.axvline(x=drop_idx, linestyle="--", alpha=0.5)
            
        plt.title("Attraction Basin Dynamics (Cond-Uncond Diff)")
        plt.xlabel("Step Index (0=Start, 50=End)")
        plt.ylabel("L2 Norm Difference")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{RESULTS_DIR}/{save_name}.png")
        plt.close()

    # Plot for first 3 memorized
    plot_specific_prompts([0, 1, 2], "sample_dynamics")
    
    print("Analysis Complete.")

if __name__ == "__main__":
    main()
