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
controlnet_path = os.path.join(base_path, "ControlNet")
lcm_path = os.path.join(base_path, "LCM_LoRA")

# Ensure directories exist
for path in [model_path, controlnet_path, lcm_path]:
    ensure_dir(path)

print(f"Downloading models to: {base_path}")

# Download base model (SDXL-Turbo)
print("Downloading SDXL-Turbo...")
model_file = hf_hub_download(
    repo_id="stabilityai/sdxl-turbo",
    filename="sd_xl_turbo_1.0_fp16.safetensors",
    repo_type="model"
)
target_model_path = os.path.join(model_path, "sd_xl_turbo_1.0_fp16.safetensors")
shutil.copy(model_file, target_model_path)
print(f"Saved base model to: {target_model_path}")

# Download LCM LoRA
print("Downloading LCM LoRA...")
lcm_file = hf_hub_download(
    repo_id="latent-consistency/lcm-lora-sdxl",
    filename="pytorch_lora_weights.safetensors",
    repo_type="model"
)
target_lcm_path = os.path.join(lcm_path, "lcm-lora-sdxl.safetensors")
shutil.copy(lcm_file, target_lcm_path)
print(f"Saved LCM LoRA to: {target_lcm_path}")

print("Downloads complete!") 