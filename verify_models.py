import os
import json
import shutil

def verify_and_fix_models():
    # Read the config file
    config_path = os.path.join(os.path.dirname(__file__), 'stream_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get the model paths
    model_paths = {
        'Base Model': config['model_id_or_path'],
        'ControlNet': config.get('controlnet_model'),
        'LCM LoRA': config.get('lcm_lora_id')
    }
    
    # Verify each path
    for name, path in model_paths.items():
        if not path:
            continue
            
        # Normalize the path
        normalized_path = os.path.abspath(os.path.expanduser(path))
        
        print(f"\nChecking {name}:")
        print(f"Original path: {path}")
        print(f"Normalized path: {normalized_path}")
        
        if os.path.exists(normalized_path):
            print(f"✅ File exists")
            file_size = os.path.getsize(normalized_path) / (1024 * 1024 * 1024)  # Size in GB
            print(f"File size: {file_size:.2f} GB")
        else:
            print(f"❌ File not found")

if __name__ == '__main__':
    verify_and_fix_models() 