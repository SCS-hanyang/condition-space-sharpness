import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
import os
import argparse

# Random Seed Setting
def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

seed_everything()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def load_prompts(file_path):
    with open(file_path, 'r') as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts

base_path = "/home/gpuadmin/cssin/cond_Jacobian"

mem_prompts_path = os.path.join(base_path, "prompts/sd1_mem.txt")
nmem_prompts_path = os.path.join(base_path, "prompts/sd1_nmem.txt")

mem_prompts = load_prompts(mem_prompts_path)
nmem_prompts = load_prompts(nmem_prompts_path)

print(f"Loaded {len(mem_prompts)} memorized prompts.")
print(f"Loaded {len(nmem_prompts)} non-memorized prompts.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_mem_prompts", type=int, default=500, help="Number of memorized prompts to use")
    parser.add_argument("--num_unmem_prompts", type=int, default=500, help="Number of unmemorized prompts to use")
    parser.add_argument("--model_id", type=str, default="v1-4", choices=["v1-4", "v2-1-base"], help="Stable Diffusion model version")
    args = parser.parse_args()

    NUM_MEM_PROMPTS = args.num_mem_prompts
    NUM_UNMEM_PROMPTS = args.num_unmem_prompts
    MODEL_ID = args.model_id
    
    model_map = {
        "v1-4": "CompVis/stable-diffusion-v1-4",
        "v2-1-base": "stabilityai/stable-diffusion-2-1-base"
    }
    model_id = model_map[MODEL_ID]

    results_path = os.path.join(base_path, f"results/clip_model_analysis/{MODEL_ID}")
    os.makedirs(results_path, exist_ok=True)

    print(f"Loading model {model_id} via StableDiffusionPipeline...")
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder.to(device)
    text_encoder.eval()

    torch.cuda.empty_cache()

    print("Model loaded successfully via Pipeline.")

    def compute_jacobian_norm(prompt, tokenizer, text_encoder, num_projections=10):
        # Tokenize
        text_input = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = text_input.input_ids.to(device)                     # discrete한 값
        # Note: Pipeline ignores attention_mask, so we do too.

        # Get Input Embeddings (Token Embeddings)
        # This layer returns Raw Token Embeddings (without position encoding)
        input_embeddings_layer = text_encoder.get_input_embeddings()
        ref_embedding = pipeline._encode_prompt(prompt, device, 1, False, None)
        x = input_embeddings_layer(input_ids).detach()
        x.requires_grad_(True)                                          # continuous한 값

        # Pre-compute Masks for Manual Forward
        bsz, seq_len = x.shape[:2]
        
        # 1. Causal Mask
        # Manually create causal mask since _build_causal_attention_mask might be missing
        # Lower triangle is 0 (attend), Upper is -inf (mask)
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=x.dtype)
        mask = torch.triu(mask, diagonal=1)
        causal_attention_mask = mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seq_len, seq_len)

        # 2. Extended Attention Mask (Padding)
        # We explicitly set this to None to match Stable Diffusion Pipeline behavior
        extended_attention_mask = None
        
        sq_norms = []
        
        for _ in range(num_projections):
            # Manual Forward Pass to utilize inputs_embeds
            
            # A. Embeddings (Add Position Embeddings to x)
            # CLIPTextEmbeddings usually accepts inputs_embeds
            hidden_states = text_encoder.text_model.embeddings(inputs_embeds=x)
            
            # B. Encoder
            encoder_outputs = text_encoder.text_model.encoder(
                inputs_embeds=hidden_states,
                attention_mask=extended_attention_mask,
                causal_attention_mask=causal_attention_mask
            )
            
            last_hidden_state = encoder_outputs[0]
            
            # C. Final Layer Norm
            y = text_encoder.text_model.final_layer_norm(last_hidden_state)
            assert torch.equal(y, ref_embedding), "코드 병신됨"
            
            # Sample random vector u in Output space
            u = torch.randn_like(y)
            
            # Compute VJP
            v = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=u,
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )[0]
            
            norm_sq = torch.sum(v ** 2).item()
            sq_norms.append(norm_sq)
            
        avg_sq_norm = np.mean(sq_norms)
        return np.sqrt(avg_sq_norm)

    mem_norms = []
    nmem_norms = []
    n = 10

    print("Calculating for Memorized Prompts...")
    for prompt in tqdm(mem_prompts[:NUM_MEM_PROMPTS]):
        try:
            norm = compute_jacobian_norm(prompt, tokenizer, text_encoder, num_projections=50)
            mem_norms.append(norm)
        except Exception as e:
            print(f"Error processing prompts: {e}")
            break

    if len(mem_norms) > 0:
        print(f"Completed {len(mem_norms)} memorized prompts.")
        print("Calculating for Non-memorized Prompts...")
        for prompt in tqdm(nmem_prompts[:NUM_UNMEM_PROMPTS]):
            try:
                norm = compute_jacobian_norm(prompt, tokenizer, text_encoder, num_projections=50)
                nmem_norms.append(norm)
            except Exception as e:
                print(f"Error processing prompts: {e}")
                break

        # Save results
        np.save(os.path.join(results_path, "mem_clip_jnorms.npy"), mem_norms)
        np.save(os.path.join(results_path, "nmem_clip_jnorms.npy"), nmem_norms)
    else:
        print("Aborted due to error.")

    if len(mem_norms) > 0 and len(nmem_norms) > 0:
        plt.figure(figsize=(10, 6))

        # Use histplot instead of kdeplot
        sns.histplot(mem_norms, label='Memorized', color='red', alpha=0.3, element="step", stat="density", kde=False, bins=30)
        sns.histplot(nmem_norms, label='Non-memorized', color='blue', alpha=0.3, element="step", stat="density", kde=False, bins=30)

        plt.title('Distribution of CLIP Jacobian Frobenius Norms ($||J_c||_F$)', fontsize=16)
        plt.xlabel('$||J_c||_F$', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, "clip_jacobian_norm_dist.png"), dpi=300)
        plt.show()
    else:
        print("Not enough data to plot.")

if __name__ == "__main__":
    main()
