
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

# Configuration
RESULTS_DIR = "results/cond_Eigen_dynamics_in_attraction_basin"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
NUM_STEPS = 50

from scipy.sparse.linalg import svds, LinearOperator

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_prompts():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'prompts', 'memorized_laion_prompts.csv')
        df = pd.read_csv(csv_path, sep=';')
        memorized = df['Caption'].tolist()[:10]
    except Exception:
        # Fallback
        try:
             df = pd.read_csv('init_noise_diffusion_memorization/prompts/memorized_laion_prompts.csv', sep=';')
             memorized = df['Caption'].tolist()[:10]
        except:
             memorized = []

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
        "A foggy morning in a pine forest with dew on the ground"
    ]
    return memorized, unmemorized

def get_jacobian_linear_operator(unet, latents, t, prompt_embeds_leaf):
    """
    Creates a LinearOperator for the Jacobian J = d(epsilon)/d(c).
    J shape: (dim_out, dim_in)
    dim_out = 4 * 64 * 64 = 16384
    dim_in = 77 * 768 = 59136 (TE)
    """
    # Shapes
    dim_out = latents.numel() # epsilon shape is same as latents (1, 4, 64, 64)
    dim_in = prompt_embeds_leaf.numel()
    
    # Pre-compute noise_pred once to build graph? No, we need graph for VJP.
    # But for 'matvec' (J @ v), we can use jvp (forward mode) or simply standard autograd if we don't have jvp easily (SD 1.4 pipeline).
    # Since `torch.func.jvp` is available in newer torch, let's stick to `autograd.grad` (VJP) which gives J^T @ u.
    # SVDs needs both J @ v and J^T @ u.
    # J @ v (Forward AD) is efficient if v is one vector.
    # J^T @ u (Reverse AD) is efficient if u is one vector.
    
    # PyTorch VJP (Reverse Mode): efficient for J^T @ u.
    # PyTorch JVP (Forward Mode): efficient for J @ v.
    
    # We will assume prompt_embeds_leaf requires grad.
    
    def matvec(v_np):
        # v is (dim_in,) numpy array. Convert to tensor.
        v_tensor = torch.from_numpy(v_np).to(device=DEVICE, dtype=torch.float32).reshape(prompt_embeds_leaf.shape)
        
        # J @ v = directional derivative in direction v.
        # We can approximate this with finite difference or use Forward AD.
        # Use simple finite diff for stability/simplicity if Forward AD is complex to setup with unet wrapper?
        # No, let's use the exact trick:
        # g(epsilon) w.r.t c. No.
        # Finite Difference: (f(c + eps*v) - f(c - eps*v)) / 2eps
        
        epsilon = 1e-3
        # Check memory - might OOM if we clone graph.
        # Actually standard forward pass is cheap.
        
        with torch.no_grad():
            c_plus = prompt_embeds_leaf + epsilon * v_tensor
            c_minus = prompt_embeds_leaf - epsilon * v_tensor
            
            out_plus = unet(latents, t, encoder_hidden_states=c_plus).sample
            out_minus = unet(latents, t, encoder_hidden_states=c_minus).sample
            
            j_v = (out_plus - out_minus) / (2 * epsilon)
            
        return j_v.flatten().cpu().numpy()

    def rmatvec(u_np):
        # u is (dim_out,) numpy array.
        # J^T @ u = VJP. d(u^T * f)/dc.
        u_tensor = torch.from_numpy(u_np).to(device=DEVICE, dtype=torch.float32).reshape(latents.shape)
        
        # We need to re-run forward pass to get graph
        # This is slow but necessary for exact VJP without keeping graph 
        # (or retain_graph=True in main loop but that consumes memory).
        
        with torch.enable_grad():
            # Run forward
            noise_pred = unet(latents, t, encoder_hidden_states=prompt_embeds_leaf).sample
            
            # VJP
            vjp_val = torch.autograd.grad(
                outputs=noise_pred,
                inputs=prompt_embeds_leaf,
                grad_outputs=u_tensor,
                create_graph=False,
                retain_graph=False
            )[0]
            
        return vjp_val.flatten().cpu().numpy()

    return LinearOperator((dim_out, dim_in), matvec=matvec, rmatvec=rmatvec)

def compute_top_k_eigenvalues(unet, latents, t, prompt_embeds, k=10):
    """
    Computes top-k singular values (which are sqrt of eigenvalues of J^T J)
    using scipy.sparse.linalg.svds
    """
    prompt_embeds = prompt_embeds.detach().requires_grad_(True)
    
    # Build wrapper
    A = get_jacobian_linear_operator(unet, latents, t, prompt_embeds)
    
    # SVDS
    # k must be < min(shape)
    try:
        u, s, vt = svds(A, k=k)
        # s is sorted ascending. Reverse it.
        return s[::-1]
    except Exception as e:
        print(f"SVDs failed: {e}. Returning zeros.")
        return np.zeros(k)

# For IE, we need a separate LinearOperator that wraps the manual TextEncoder forward
def get_jacobian_linear_operator_IE(pipeline, latents, t, input_embeddings, input_ids):
    dim_out = latents.numel()
    dim_in = input_embeddings.numel()
    
    def manual_forward(embeds):
         # Copy-paste manual forward logic from previous script
         # Need to handle this cleanly.
         text_model = pipeline.text_encoder.text_model
         bsz, seq_len = input_ids.shape
         
         # 1. Embeddings
         position_ids = text_model.embeddings.position_ids[:, :seq_len]
         hidden_states = text_model.embeddings(inputs_embeds=embeds, position_ids=position_ids)
         
         # 2. Mask
         mask = torch.full((bsz, 1, seq_len, seq_len), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype)
         mask = torch.triu(mask, diagonal=1)
         
         # 3. Encoder
         encoder_outputs = text_model.encoder(
             hidden_states,
             attention_mask=None,
             causal_attention_mask=mask 
         )
         last_hidden_state = text_model.final_layer_norm(encoder_outputs[0])
         
         # 4. UNet
         noise_pred = pipeline.unet(latents, t, encoder_hidden_states=last_hidden_state).sample
         return noise_pred

    def matvec(v_np):
        v_tensor = torch.from_numpy(v_np).to(device=DEVICE, dtype=torch.float32).reshape(input_embeddings.shape)
        epsilon = 1e-3
        with torch.no_grad():
            out_plus = manual_forward(input_embeddings + epsilon * v_tensor)
            out_minus = manual_forward(input_embeddings - epsilon * v_tensor)
            j_v = (out_plus - out_minus) / (2 * epsilon)
        return j_v.flatten().cpu().numpy()

    def rmatvec(u_np):
        u_tensor = torch.from_numpy(u_np).to(device=DEVICE, dtype=torch.float32).reshape(latents.shape)
        with torch.enable_grad():
             # We assume input_embeddings requires grad
             if not input_embeddings.requires_grad:
                 input_embeddings.requires_grad_(True)
             
             noise_pred = manual_forward(input_embeddings)
             vjp_val = torch.autograd.grad(noise_pred, input_embeddings, grad_outputs=u_tensor)[0]
        return vjp_val.flatten().cpu().numpy()

    return LinearOperator((dim_out, dim_in), matvec=matvec, rmatvec=rmatvec)

def compute_top_k_eigenvalues_IE(pipeline, latents, t, input_embeddings, input_ids, k=10):
    input_embeddings = input_embeddings.detach().requires_grad_(True)
    A = get_jacobian_linear_operator_IE(pipeline, latents, t, input_embeddings, input_ids)
    try:
        u, s, vt = svds(A, k=k)
        return s[::-1]
    except Exception as e:
        print(f"IE SVDs failed: {e}")
        return np.zeros(k)

def get_input_embeddings(pipeline, prompt, device):
    text_inputs = pipeline.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    input_embeddings = pipeline.text_encoder.text_model.embeddings.token_embedding(input_ids)
    return input_ids, input_embeddings

def main():
    set_seed(SEED)
    print("Loading pipeline...")
    pipeline = LocalStableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, requires_safety_checker=False
    ).to(DEVICE)
    pipeline.set_progress_bar_config(disable=True)
    
    memorized_prompts, unmemorized_prompts = load_prompts()
    
    # 1. Uncond Trajectory
    unc_prompt_embeds = pipeline._encode_prompt("", DEVICE, 1, False, None)
    height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    width = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    pipeline.scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    timesteps = pipeline.scheduler.timesteps
    
    latents = pipeline.prepare_latents(
        1, pipeline.unet.config.in_channels, height, width,
        torch.float32, torch.device(DEVICE), torch.Generator(device=DEVICE).manual_seed(SEED), None
    )
    
    trajectory_latents = [] 
    trajectory_uncond_noise = []
    
    curr_latents = latents
    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps, desc="Uncond Trajectory")):
            trajectory_latents.append(curr_latents.clone())
            latent_model_input = pipeline.scheduler.scale_model_input(curr_latents, t)
            noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=unc_prompt_embeds).sample
            trajectory_uncond_noise.append(noise_pred.clone())
            curr_latents = pipeline.scheduler.step(noise_pred, t, curr_latents, return_dict=False)[0]
            
    # 2. Analyze
    results = []
    
    for group_name, prompts in [("Memorized", memorized_prompts), ("Unmemorized", unmemorized_prompts)]:
        print(f"Analyzing {group_name}...")
        
        # A. Find Basins first
        curves = []
        for prompt in tqdm(prompts, desc=f"{group_name} Curves"):
            cond_embeds = pipeline._encode_prompt(prompt, DEVICE, 1, False, None)
            diff_norms = []
            with torch.no_grad():
                for i, t in enumerate(timesteps):
                    lat = trajectory_latents[i]
                    latent_input = pipeline.scheduler.scale_model_input(lat, t)
                    cond_noise = pipeline.unet(latent_input, t, encoder_hidden_states=cond_embeds).sample
                    diff = torch.norm(cond_noise - trajectory_uncond_noise[i]).item()
                    diff_norms.append(diff)
            curves.append(diff_norms)
        
        curves = np.array(curves)
        drop_indices = []
        for c in curves:
            drop_indices.append(np.argmin(np.diff(c)[5:]) + 5)
            
        # B. Compute Eigenvalues
        print(f"Calculating Eigenvalues for {group_name}...")
        for i, prompt in enumerate(tqdm(prompts, desc="Eigenvalues")):
            drop_idx = drop_indices[i]
            # Calculate for inside (0~drop) + outside (drop+1~drop+5)
            calc_indices = list(range(0, drop_idx + 1)) + list(range(drop_idx + 1, min(drop_idx + 5, NUM_STEPS)))
            
            # Common Inputs
            input_ids, input_embeddings = get_input_embeddings(pipeline, prompt, DEVICE)
            prompt_embeds_te = pipeline._encode_prompt(prompt, DEVICE, 1, False, None)
            
            for t_idx in calc_indices:
                t = timesteps[t_idx]
                lat = trajectory_latents[t_idx]
                latent_input = pipeline.scheduler.scale_model_input(lat, t)
                
                # Top 10 Singular Values (SVD of J) => sqrt(Eigenvalues of J^T J)
                # TE
                s_te = compute_top_k_eigenvalues(pipeline.unet, latent_input, t, prompt_embeds_te, k=10)
                
                # IE
                s_ie = compute_top_k_eigenvalues_IE(pipeline, latent_input, t, input_embeddings, input_ids, k=10)
                
                region = "Inside Basin" if t_idx <= drop_idx else "Outside Basin"
                
                results.append({
                    "Prompt": prompt,
                    "Group": group_name,
                    "Drop_Index": drop_idx,
                    "Step_Index": t_idx,
                    "Relative_Step": t_idx - drop_idx,
                    "Region": region,
                    "Top10_Eigenvalues_TE": s_te.tolist(),
                    "Top10_Eigenvalues_IE": s_ie.tolist(),
                    "Diff_Curve": curves[i].tolist()
                })
                
                torch.cuda.empty_cache()

    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{RESULTS_DIR}/eigen_analysis.csv", index=False)
    print("Analysis Complete.")

if __name__ == "__main__":
    main()
