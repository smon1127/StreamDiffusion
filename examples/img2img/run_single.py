import os
import sys
from typing import Literal, Dict, Optional
import fire
import torch

# Add the StreamDiffusionTD directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.wrapper import StreamDiffusionWrapper

def main():
    # Use absolute paths for everything
    input_path = "/Users/simon/Repos/StreamDiffusionTD/StreamDiffusionTD/images/inputs/input.png"
    output_path = "/Users/simon/Repos/StreamDiffusionTD/StreamDiffusionTD/images/outputs/output.png"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Set device to MPS if available, otherwise CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize StreamDiffusion with SDXL from Hugging Face
    stream = StreamDiffusionWrapper(
        model_id_or_path="stabilityai/stable-diffusion-xl-base-1.0",
        t_index_list=[22, 32, 45],
        frame_buffer_size=1,
        width=1024,
        height=1024,
        warmup=10,
        acceleration="none",  # No acceleration on Mac
        mode="img2img",
        use_denoising_batch=True,
        cfg_type="self",
        seed=2,
        device=device,
        use_lcm_lora=False  # Disable LCM LoRA
    )

    # Prepare the model
    stream.prepare(
        prompt="a beautiful landscape with mountains and a lake, masterpiece, highly detailed",
        negative_prompt="low quality, bad quality, blurry",
        num_inference_steps=50,
        guidance_scale=7.5,
        delta=0.5,
    )

    # Load and process image
    image_tensor = stream.preprocess_image(input_path)

    # Generate image
    for _ in range(stream.batch_size - 1):
        stream(image=image_tensor)

    output_image = stream(image=image_tensor)
    output_image.save(output_path)
    print(f"Generated image saved to: {output_path}")

if __name__ == "__main__":
    fire.Fire(main) 