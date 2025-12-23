
import torch
from local_sd_pipeline import LocalStableDiffusionPipeline
import inspect

def check_text_encoder_type():
    model_id = "CompVis/stable-diffusion-v1-4"
    print(f"Loading pipeline from: {model_id}")
    
    # Load pipeline
    # We use torch_dtype=torch.float16 or bfloat16 to be faster/lighter if available, 
    # but cpu default is safer for type checking without GPU errors if user env is mixed.
    # However, user seems to have GPU env. Let's use default (float32) for safety on CPU if needed, 
    # or just trust env.
    
    try:
        pipe = LocalStableDiffusionPipeline.from_pretrained(
            model_id,
            safety_checker=None,
            requires_safety_checker=False 
        )
        
        text_encoder = pipe.text_encoder
        print("\n=== Text Encoder Information ===")
        print(f"Type: {type(text_encoder)}")
        print(f"Class Name: {text_encoder.__class__.__name__}")
        print(f"Module: {text_encoder.__module__}")
        
        # Check if it is a transformers model
        try:
            import transformers
            if isinstance(text_encoder, transformers.PreTrainedModel):
                print(f"Is Transformers PreTrainedModel: Yes")
                print(f"Model Configuration: {text_encoder.config._name_or_path}")
        except ImportError:
            pass

    except Exception as e:
        print(f"Error loading pipeline: {e}")

if __name__ == "__main__":
    check_text_encoder_type()
