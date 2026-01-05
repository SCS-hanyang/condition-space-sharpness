
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from diffusers import UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
import argparse

class Config:
    image_size = 32
    clip_name = "openai/clip-vit-base-patch32"
    # Updated to point to the test training output directory
    out_dir = "./results/condition_Jacobian_dynamics_during_training_diff_prompts_test/"
    # UNet config
    in_channels = 3
    out_channels = 3
    layers_per_block = 1  # Note: logic in test training script might differ, let's check
    block_out_channels = (32, 64, 64) # Note: Checked test script, it uses (32, 64, 64) and layers_per_block=1
    down_block_types = ("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D")
    up_block_types = ("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D")

def get_sorted_checkpoints(out_dir):
    if not os.path.exists(out_dir):
        return []
    files = os.listdir(out_dir)
    ckpt_files = [f for f in files if f.startswith("ckpt_step_") and f.endswith(".pt")]
    try:
        steps_files = []
        for f in ckpt_files:
            step_str = f.split("_")[-1].split(".")[0]
            steps_files.append((int(step_str), f))
        steps_files.sort(key=lambda x: x[0])
        return steps_files
    except ValueError:
        print("Error parsing checkpoint filenames.")
        return []

def load_model(ckpt_idx, sorted_ckpts, device):
    try:
        step, filename = sorted_ckpts[ckpt_idx]
    except IndexError:
        print(f"Index {ckpt_idx} out of range.")
        return None, None, None, None, None

    ckpt_path = os.path.join(Config.out_dir, filename)
    # print(f"Loading checkpoint index {ckpt_idx}: {filename} (Step {step}) ...")
    
    tokenizer = CLIPTokenizer.from_pretrained(Config.clip_name)
    text_encoder = CLIPTextModel.from_pretrained(Config.clip_name).to(device)
    
    cross_attn_dim = text_encoder.config.hidden_size
    unet = UNet2DConditionModel(
        sample_size=Config.image_size,
        in_channels=Config.in_channels,
        out_channels=Config.out_channels,
        layers_per_block=Config.layers_per_block,
        block_out_channels=Config.block_out_channels,
        down_block_types=Config.down_block_types,
        up_block_types=Config.up_block_types,
        cross_attention_dim=cross_attn_dim,
    ).to(device)
    
    state_dict = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(state_dict)
    unet.eval()
    
    return unet, tokenizer, text_encoder, step, filename

def compute_jacobian_norm_batched(unet, latents, t, prompt_embeds, num_projections=4):
    prompt_embeds = prompt_embeds.detach()
    latents = latents.detach()
    batch_size = latents.shape[0]
    
    with torch.enable_grad():
        prompt_embeds.requires_grad_(True)
        noise_pred = unet(latents, t, encoder_hidden_states=prompt_embeds).sample
        
        sq_norm_sum = torch.zeros(batch_size, device=latents.device)
        
        for k in range(num_projections):
            v = torch.randn_like(noise_pred)
            v_dot_eps_sum = torch.sum(noise_pred * v)
            
            retain = (k < num_projections - 1)
            grads = torch.autograd.grad(v_dot_eps_sum, prompt_embeds, retain_graph=retain, create_graph=False)[0]
            
            grads_sq_flat = grads.view(batch_size, -1).pow(2)
            sample_sq_norms = grads_sq_flat.sum(dim=1)
            
            sq_norm_sum += sample_sq_norms.detach()
            
        est_frob_sq = sq_norm_sum / max(1, float(num_projections))
        return est_frob_sq.sqrt()

@torch.no_grad()
def encode_prompt(tokenizer, text_encoder, prompts, device, max_length=77):
    tokens = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    attn_mask = tokens.attention_mask.to(device)
    out = text_encoder(input_ids=input_ids, attention_mask=attn_mask)
    return out.last_hidden_state

def prepare_prompts():
    # 1. Memorized Prompts (cifar10_prompts.csv)
    df_mem = pd.read_csv("cifar10_prompts.csv")
    
    # 2. Non-Memorized Prompts (cifar10_prompts_nm.csv)
    df_nm = pd.read_csv("cifar10_prompts_nm.csv")

    # Filter for classes 1, 3, 5
    target_indices = [1, 3, 5]
    df_mem = df_mem[df_mem['class_index'].isin(target_indices)]
    df_nm = df_nm[df_nm['class_index'].isin(target_indices)]
    
    # Classes
    classes = df_mem['class_name'].unique()
    
    prompt_data = {} # {class_name: {'Memorized': [10], 'SameNoun': [5], 'Synonym': [5]}}
    
    for cls in classes:
        prompt_data[cls] = {}
        
        # Memorized: 10 simple_prompts
        mem_prompts = df_mem[df_mem['class_name'] == cls]['simple_prompt'].tolist()
        # Take first 10, repeating if necessary
        if len(mem_prompts) < 10:
            mem_prompts = (mem_prompts * (10 // len(mem_prompts) + 1))[:10]
        else:
            mem_prompts = mem_prompts[:10]
        prompt_data[cls]['Memorized'] = mem_prompts
        
        # Non-Memorized SameNoun: 5
        nm_sn_prompts = df_nm[df_nm['class_name'] == cls]['same_noun'].tolist()
        if len(nm_sn_prompts) < 5:
            nm_sn_prompts = (nm_sn_prompts * (5 // len(nm_sn_prompts) + 1))[:5]
        else:
            nm_sn_prompts = nm_sn_prompts[:5]
        prompt_data[cls]['SameNoun'] = nm_sn_prompts
        
        # Non-Memorized Synonym: 5
        nm_syn_prompts = df_nm[df_nm['class_name'] == cls]['synonym'].tolist()
        if len(nm_syn_prompts) < 5:
            nm_syn_prompts = (nm_syn_prompts * (5 // len(nm_syn_prompts) + 1))[:5]
        else:
            nm_syn_prompts = nm_syn_prompts[:5]
        prompt_data[cls]['Synonym'] = nm_syn_prompts
        
    return classes, prompt_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_init_noise", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--output_file", type=str, default="./results/calculate_condition_Jacobin_norm_CJDDT_test/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stride", type=int, default=1, help="Stride for checkpoint processing")
    parser.add_argument("--calculate_timestep", type=int, default=999, help="Timestep to calculate Jacobian norm")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    file_name = f"jacobian_analysis_results_{args.calculate_timestep}.csv"
    print(f"Using device: {device}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    output_file = os.path.join(args.output_file, file_name)
    print(output_file)
    
    # 1. Prepare Data
    classes, prompt_map = prepare_prompts()
    strongly_mem_prompt = "Memorized sample made by changsu shin, the student of university student"
    
    # 2. Checkpoints
    sorted_ckpts = get_sorted_checkpoints(Config.out_dir)
    if not sorted_ckpts:
        print(f"No checkpoints found in {Config.out_dir}.")
        return
        
    indices_to_analyze = list(range(0, len(sorted_ckpts), args.stride))
    if (len(sorted_ckpts)-1) not in indices_to_analyze:
        indices_to_analyze.append(len(sorted_ckpts)-1)
    
    # 3. Fixed Init Noise
    rng = torch.Generator(device=device).manual_seed(42)
    init_noises = torch.randn((args.num_init_noise, Config.in_channels, Config.image_size, Config.image_size), generator=rng, device=device)
    t_init = torch.full((args.num_init_noise,), args.calculate_timestep, device=device, dtype=torch.long)
    
    all_results = []
    
    # 4. Processing Loop
    for idx in tqdm(indices_to_analyze, desc="Processing Checkpoints"):
        unet, tokenizer, text_encoder, step, _ = load_model(idx, sorted_ckpts, device)
        if unet is None: continue
        
        # Helper to process a batch of prompts
        def process_prompts(prompts, condition_label, cls_label, prompt_indices):
            # Encode prompts: [NumPrompts, 77, 768]
            
            cond_embeds_all = encode_prompt(tokenizer, text_encoder, prompts, device) # [P, 77, Dim]
            
            for p_i, p_embed in enumerate(cond_embeds_all):
                # p_embed: [77, Dim]
                # Repeat for num_init_noise
                batch_embeds = p_embed.unsqueeze(0).repeat(args.num_init_noise, 1, 1) # [N, 77, Dim]
                
                # Compute Jacobian Norms in batches of noises
                # Batched processing of noises
                norms_list = []
                for b_start in range(0, args.num_init_noise, args.batch_size):
                    b_end = min(b_start + args.batch_size, args.num_init_noise)
                    
                    bn_norms = compute_jacobian_norm_batched(
                        unet, 
                        init_noises[b_start:b_end], 
                        t_init[b_start:b_end], 
                        batch_embeds[b_start:b_end],
                        num_projections=4
                    )
                    norms_list.append(bn_norms.detach().cpu())
                
                norms = torch.cat(norms_list).numpy()
                
                # Store
                for n_i, norm_val in enumerate(norms):
                    all_results.append({
                        "ModelIdx": idx,
                        "Step": step,
                        "Class": cls_label,
                        "Condition": condition_label,
                        "PromptIdx": prompt_indices[p_i],
                        "Prompt": prompts[p_i],
                        "NoiseIdx": n_i,
                        "JacobianNorm": norm_val
                    })

        # A. Per-Class Prompts
        for cls in classes:
            # Memorized
            pm = prompt_map[cls]['Memorized']
            process_prompts(pm, "Memorized", cls, list(range(len(pm))))
            
            # SameNoun
            psn = prompt_map[cls]['SameNoun']
            process_prompts(psn, "SameNoun", cls, list(range(len(psn))))
            
            # Synonym
            psy = prompt_map[cls]['Synonym']
            process_prompts(psy, "Synonym", cls, list(range(len(psy))))
            
        # B. Strongly Memorized (Global, but we can assign class "None" or "StronglyMem")
        process_prompts([strongly_mem_prompt], "StronglyMemorized", "N/A", [0])
        
    # 5. Save Results
    print(f"Saving {len(all_results)} results to {args.output_file}...")
    df = pd.DataFrame(all_results)
    df.to_csv(args.output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
