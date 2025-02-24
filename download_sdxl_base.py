import os
from huggingface_hub import hf_hub_download
import shutil

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Get the absolute path to the StreamDiffusionTD directory
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)

# Base paths
base_path = os.path.join(base_dir, "models")
model_path = os.path.join(base_path, "Model")

# Ensure directories exist
ensure_dir(model_path)

print(f"Downloading SDXL base model to: {base_path}")

# Download SDXL base model
print("Downloading Stable Diffusion XL base 1.0...")
model_file = hf_hub_download(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    filename="sd_xl_base_1.0.safetensors",
    repo_type="model"
)
target_model_path = os.path.join(model_path, "sd_xl_base_1.0.safetensors")
shutil.copy(model_file, target_model_path)
print(f"Saved SDXL base model to: {target_model_path}")

print("Download complete!") 