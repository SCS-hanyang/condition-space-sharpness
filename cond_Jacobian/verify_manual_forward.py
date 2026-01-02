
import torch
import numpy as np
from diffusers import StableDiffusionPipeline

def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

seed_everything()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_id = "CompVis/stable-diffusion-v1-4"
print(f"Loading model {model_id}...")
pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
tokenizer = pipeline.tokenizer
text_encoder = pipeline.text_encoder.to(device)
text_encoder.eval()

test_prompt = "The No Limits Business Woman Podcast"
print(f"Test Prompt: '{test_prompt}'")

text_input = tokenizer(
    test_prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt"
)
input_ids = text_input.input_ids.to(device)

with torch.no_grad():
    # 1. Target Output (No Mask)
    ref_outputs = text_encoder(input_ids, attention_mask=None)
    ref_last_hidden_state = ref_outputs.last_hidden_state

    # 2. Manual Forward Pass (No Mask)
    input_embeddings_layer = text_encoder.get_input_embeddings()
    x = input_embeddings_layer(input_ids).detach()
    
    bsz, seq_len = x.shape[:2]
    
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=x.dtype)
    mask = torch.triu(mask, diagonal=1)
    causal_attention_mask = mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seq_len, seq_len)

    extended_attention_mask = None # As per update

    hidden_states = text_encoder.text_model.embeddings(inputs_embeds=x)
    
    encoder_outputs = text_encoder.text_model.encoder(
        inputs_embeds=hidden_states,
        attention_mask=extended_attention_mask,
        causal_attention_mask=causal_attention_mask
    )
    
    manual_last_hidden_state = encoder_outputs[0]
    manual_output = text_encoder.text_model.final_layer_norm(manual_last_hidden_state)
    
    diff = torch.abs(manual_output - ref_last_hidden_state).max().item()
    print(f"Max Difference: {diff}")

    if diff < 1e-5:
        print("SUCCESS: Manual forward pass matches target (No Mask).")
    else:
        print("FAILURE: Manual forward pass mismatch.")
