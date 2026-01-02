
import torch
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
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

# Check one prompt
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
attention_mask = text_input.attention_mask.to(device)

with torch.no_grad():
    # 1. With Mask
    out_with_mask = text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state
    
    # 2. No Mask
    out_no_mask = text_encoder(input_ids, attention_mask=None).last_hidden_state
    
    # 3. Pipeline
    # pipeline._encode_prompt signature varies by version, let's try standard args
    # _encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt)
    try:
        prompt_embeds = pipeline._encode_prompt(test_prompt, device, 1, False, None)
    except:
        # Fallback for newer diffusers
        prompt_embeds = pipeline.encode_prompt(test_prompt, device, 1, False, None)[0]

    print(f"Out With Mask shape: {out_with_mask.shape}")
    print(f"Out No Mask shape:   {out_no_mask.shape}")
    print(f"Prompt Embeds shape: {prompt_embeds.shape}")

    diff_mask = torch.abs(prompt_embeds - out_with_mask).max().item()
    diff_no_mask = torch.abs(prompt_embeds - out_no_mask).max().item()

    print(f"Diff (Pipeline vs With Mask): {diff_mask}")
    print(f"Diff (Pipeline vs No Mask):   {diff_no_mask}")

    if diff_no_mask < 1e-5:
        print("CONCLUSION: Pipeline ignores attention_mask (matches 'No Mask').")
    elif diff_mask < 1e-5:
        print("CONCLUSION: Pipeline uses attention_mask (matches 'With Mask').")
    else:
        print("CONCLUSION: Pipeline does something else.")
