#main_sdtd
import os
import sys
import json
import argparse
import signal
from multiprocessing import Process, Queue, Manager, shared_memory
from typing import Dict, Literal, Optional, List, Union  # Required for type hints
import time
import threading
import platform
import subprocess
import re
from collections import deque
from contextlib import nullcontext
# print('TESTING NEW MAC COMPATIBLITY !!! ')
def read_initial_config(config_path):
    """Minimal config reading just to get GPU ID"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config.get("gpu_id", 0)

def init_gpu_process(gpu_id):
    """Set GPU before any torch-related imports"""
    if platform.system() == "Darwin":  # Mac
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    else:  # Windows/Linux
        # Import torch here just for GPU detection
        import torch
        
        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        
        if num_gpus == 0:
            print("\033[33mWarning: No CUDA devices available. Defaulting to CPU.\033[0m")
            return
            
        # Validate requested GPU ID
        if gpu_id >= num_gpus:
            print(f"\033[33mWarning: Requested GPU {gpu_id} not available. Found {num_gpus} GPU(s).")
            print(f"Available devices:\033[0m")
            for i in range(num_gpus):
                print(f"\033[33m  GPU {i}: {torch.cuda.get_device_name(i)}\033[0m")
            print(f"\033[33mDefaulting to GPU 0.\033[0m")
            gpu_id = 0
            
        # Set CUDA device and environment variable
        torch.cuda.set_device(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Print selected device info
        print(f"\033[32mUsing GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}\033[0m")

def print_gpu_info():
    if platform.system() == "Darwin":  # Mac
        print("\033[33mDevice Info: \033[37mMPS (Apple Silicon)\033[0m\n")
    else:
        import torch
        try:
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                print(f"\033[33mDevice Info: \033[37m{torch.cuda.get_device_name(current_device)}")
                print(f"CUDA Version: {torch.version.cuda}")
                print(f"PyTorch Version: {torch.__version__}\033[0m\n")
                
                # Still try nvidia-smi for additional info
                try:
                    subprocess.check_output(['nvidia-smi'])
                except:
                    pass
            else:
                print("\033[33mWarning: CUDA not available. Using CPU.\033[0m\n")
        except Exception as e:
            print(f"\033[33mWarning: Error getting GPU info: {e}\033[0m\n")

# 3. Set up paths and GPU before any CUDA/torch imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
default_config_path = os.path.join(current_script_dir, 'stream_config.json')
gpu_id = read_initial_config(default_config_path)
init_gpu_process(gpu_id)

# Additional imports after GPU setup
import numpy as np
import cv2
import fire
from queue import Empty
import torch
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F
import uuid
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import array
from itertools import repeat
from pythonosc import udp_client


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from wrapper_td import StreamDiffusionWrapper
from pipeline_td import StreamDiffusion

# from streamdiffusion.image_utils import postprocess_image
from attention_processor import CachedSTXFormersAttnProcessor, CachedSTAttnProcessor2_0  #V2V


class OSCClientFactory:
    _clients = {}
    @staticmethod
    def get_client(osc_out_port):
        if osc_out_port not in OSCClientFactory._clients:
            OSCClientFactory._clients[osc_out_port] = udp_client.SimpleUDPClient("127.0.0.1", osc_out_port)
        return OSCClientFactory._clients[osc_out_port]

def send_osc_message(address, value, osc_out_port):
    client = OSCClientFactory.get_client(osc_out_port)
    client.send_message(address, value)

def calculate_fps_and_send_osc(start_time, transmit_count, osc_out_port, sender_name, frame_created, use_controlnet):
    if frame_created:
        transmit_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:
        if frame_created:
            fps = round(transmit_count / elapsed_time, 3)
            send_osc_message('/stream-info/fps', fps, osc_out_port)
            controlnet_status = "ControlNet is Active | " if use_controlnet else ""
            print(f"Streaming... Active | {controlnet_status}Sender: {sender_name} | FPS: {fps}\r", end='', flush=True)
        send_osc_message('/stream-info/output-name', sender_name, osc_out_port)
        start_time = time.time()
        if frame_created:
            transmit_count = 0
    return start_time, transmit_count

def osc_server(shared_data, ip='127.0.0.1', port=8247, osc_transmit_port=8248):
    def set_negative_prompt_handler(address, *args):
        shared_data["negative_prompt"] = args[0]
    def set_guidance_scale_handler(address, *args):
        shared_data["guidance_scale"] = args[0]
    def set_delta_handler(address, *args):
        shared_data["delta"] = args[0]
    def set_seed_handler(address, *args):
        shared_data["seed"] = args[0]
    def set_t_list_handler(address, *args):
        shared_data['t_list'] = list(args)
    def set_prompt_list_handler(address, *args):
        prompt_list_str = args[0]
        prompt_list = json.loads(prompt_list_str)
        shared_data["prompt_list"] = prompt_list
    def set_seed_list_handler(address, *args):
        seed_list_str = args[0]
        seed_list = json.loads(seed_list_str)
        seed_list = [[int(seed_val), float(weight)] for seed_val, weight in seed_list]
        shared_data["seed_list"] = seed_list
    def set_sdmode_handler(address, *args):
        shared_data["sdmode"] = args[0]
    def stop_stream_handler(address, *args):
        shared_data["stop_stream"] = True
    def pause_stream_handler(address, *args):
        shared_data["pause_stream"] = True
    def play_stream_handler(address, *args):
        shared_data["play_stream"] = True
    def unload_stream_handler(address, *args):
        shared_data["unload_stream"] = True
    def set_gaussian_prompt_handler(address, *args):
        shared_data["gaussian_prompt"] = args[0]
    def set_td_buffer_name_handler(address, *args):
        shared_data["input_mem_name"] = args[0]
    def set_controlnet_weight_handler(address, *args):
        shared_data["controlnet_conditioning_scale"] = args[0]
    def set_use_controlnet_handler(address, *args):
        shared_data["use_controlnet"] = args[0]
    def set_feedback_safe_handler(address, *args):
        shared_data["feedback_safe"] = args[0]
    def set_use_cached_attn_settings_handler(address, *args): #V2V
        shared_data["use_cached_attn_settings"] = json.loads(args[0]) #V2V
    def set_use_cached_attn_handler(address, *args): #V2V
        shared_data["use_cached_attn"] = args[0] #V2V
    def set_disable_cached_attn_handler(address, *args): #V2V
        shared_data["disable_cached_attn"] = args[0] #V2V
    def set_lora_weights_handler(address, *args):
        lora_weights_str = args[0]
        lora_weights = json.loads(lora_weights_str)
        shared_data["lora_weights"] = lora_weights
    def set_max_fps_handler(address, *args):
        shared_data["max_fps"] = args[0]
    def set_slerp_handler(address, *args):
        shared_data["interpolation_method"] = args[0]
    def set_Interpval1_handler(address, *args):
        shared_data["Interpval1"] = args[0]
    def process_frame_handler(address, *args):
        shared_data["process_frame"] = True
    def set_textual_inversion_dict_handler(address, *args):
        ti_dict_str = args[0]
        ti_dict = json.loads(ti_dict_str)
        shared_data["textual_inversion_dict"] = ti_dict

    dispatcher = Dispatcher()
    dispatcher.map("/negative_prompt", set_negative_prompt_handler)
    dispatcher.map("/guidance_scale", set_guidance_scale_handler)
    dispatcher.map("/delta", set_delta_handler)
    dispatcher.map("/seed", set_seed_handler)
    dispatcher.map("/t_list", set_t_list_handler)
    dispatcher.map("/prompt_list", set_prompt_list_handler)
    dispatcher.map("/seed_list", set_seed_list_handler)
    dispatcher.map("/sdmode", set_sdmode_handler)
    dispatcher.map("/stop", stop_stream_handler)
    dispatcher.map("/pause", pause_stream_handler)
    dispatcher.map("/play", play_stream_handler)
    dispatcher.map("/unload", unload_stream_handler)
    dispatcher.map("/gaussian_prompt", set_gaussian_prompt_handler)
    dispatcher.map("/td_buffer_name", set_td_buffer_name_handler)
    dispatcher.map("/controlnet_weight", set_controlnet_weight_handler)
    dispatcher.map("/use_controlnet", set_use_controlnet_handler)
    dispatcher.map("/feedback_safe", set_feedback_safe_handler)
    dispatcher.map("/use_cached_attn_settings", set_use_cached_attn_settings_handler) #V2V
    dispatcher.map("/use_cached_attn", set_use_cached_attn_handler) #V2V
    dispatcher.map("/disable_cached_attn", set_disable_cached_attn_handler) #V2V
    dispatcher.map("/lora_weights", set_lora_weights_handler)
    dispatcher.map("/max_fps", set_max_fps_handler)
    dispatcher.map("/slerp", set_slerp_handler)
    dispatcher.map("/Interpval1", set_Interpval1_handler)
    dispatcher.map("/process_frame", process_frame_handler)
    dispatcher.map("/textual_inversion_dict", set_textual_inversion_dict_handler)
    server = BlockingOSCUDPServer((ip, port), dispatcher)
    print(f"\033[35m=======================================\nOSC ready on {ip}:{port}\n=======================================\033[0m\n")
    def send_heartbeat():
        while True:
            send_osc_message("/server_active", 1, osc_transmit_port)
            time.sleep(.75)
    heartbeat_thread = threading.Thread(target=send_heartbeat)
    heartbeat_thread.daemon = True
    heartbeat_thread.start()
    server.serve_forever()

def print_sdtd_title():
    version = "0.2.5"
    sdtd_title = f"""
\033[33m=======================================\033[0m
\033[33mStreamDiffusionTD v{version}\033[0m
\033[33m=======================================\033[0m
    """
    print(sdtd_title)

def print_gpu_info(gpu_id=0):
    query_attributes = ['name', 'memory.total', 'memory.used']
    command = f'nvidia-smi --id={gpu_id} --query-gpu={",".join(query_attributes)} --format=csv,nounits,noheader'
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True, check=True)
        nvidia_smi_info = dict(zip(query_attributes, [value.strip() for value in result.stdout.strip().split(',')]))
        gpu_id_str = f" {gpu_id}" if gpu_id != 0 else ""
        print(f"\033[33mGPU{gpu_id_str} Info: \033[37m{nvidia_smi_info['name']}\033[33m [ {round(int(nvidia_smi_info['memory.total']) / 1024, 1)} GiB total, {round(int(nvidia_smi_info['memory.used']) / 1024, 1)} GiB used ]\033[0m\n")
    except subprocess.CalledProcessError:
        print(f"Failed to execute nvidia-smi for GPU {gpu_id}")

def terminate_processes(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join()

def image_generation_process(
    model_id_or_path: str,
    lora_dict: Optional[Dict[str, float]],
    prompt: str,
    negative_prompt: str,
    frame_buffer_size: int,
    width: int,
    height: int,
    acceleration: Literal["none", "xformers", "tensorrt"],
    use_denoising_batch: bool,
    seed: int,
    cfg_type: Literal["none", "full", "self", "initialize"],
    guidance_scale: float,
    delta: float,
    do_add_noise: bool,
    enable_similar_image_filter: bool,
    similar_image_filter_threshold: float,
    similar_image_filter_max_skip_frame: float,
    shared_data,
    t_index_list: List[int] ,
    mode:str,
    lcm_lora_id: Optional[str] = None,
    vae_id: Optional[str] = None,
    input_mem_name: str = "input_mem_name",
    osc_transmit_port: Optional[int] = None,
    scheduler_name: str = "EulerAncestral",
    use_karras_sigmas: bool = False,
    device: Literal["cpu","cuda", "mps"] = "cuda",
    gpu_id: int = 0,  # New parameter for GPU ID
    use_controlnet: bool = False,
    controlnet_model: Optional[str] = None,
    controlnet_weight: float = 0.5,
    use_cached_attn: bool = True,
    sd_model_type: Optional[Literal["sd15", "sd21", "sdxl"]] = None,
    max_fps: int = 60,
    textual_inversion_dict: Optional[Dict[str, Optional[str]]] = None
) -> None:
    """
    Process for generating images based on a prompt using a specified model.
    Parameters
    ----------
    queue : Queue
        The queue to put the generated images in.
    fps_queue : Queue
        The queue to put the calculated fps.
    model_id_or_path : str
        The name of the model to use for image generation.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {"LoRA_1" : 0.5 , "LoRA_2" : 0.7 ,...}
    prompt : str
        The prompt to generate images from.
    negative_prompt : str, optional
        The negative prompt to use.
    frame_buffer_size : int, optional
        The frame buffer size for denoising batch, by default 1.
    width : int, optional
        The width of the image, by default 512.
    height : int, optional
        The height of the image, by default 512.
    acceleration : Literal["none", "xformers", "tensorrt"], optional
        The acceleration method, by default "tensorrt".
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default True.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    cfg_type : Literal["none", "full", "self", "initialize"],
    optional
        The cfg_type for img2img mode, by default "self".
        You cannot use anything other than "none" for txt2img mode.
    guidance_scale : float, optional
        The CFG scale, by default 1.2.
    delta : float, optional
        The delta multiplier of virtual residual noise,
        by default 1.0.
    do_add_noise : bool, optional
        Whether to add noise for following denoising steps or not,
        by default True.
    enable_similar_image_filter : bool, optional
        Whether to enable similar image filter or not,
        by default False.
    similar_image_filter_threshold : float, optional
        The threshold for similar image filter, by default 0.98.
    similar_image_filter_max_skip_frame : int, optional
        The max skip frame for similar image filter, by default 10.
    """
    shared_data["loading_status"] = True    
    use_lcm_lora = True if lcm_lora_id != 'skip' else False
    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=t_index_list,
        frame_buffer_size=frame_buffer_size,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        do_add_noise=do_add_noise,
        use_lcm_lora = use_lcm_lora,
        enable_similar_image_filter=enable_similar_image_filter,
        similar_image_filter_threshold=similar_image_filter_threshold,
        similar_image_filter_max_skip_frame=similar_image_filter_max_skip_frame,
        use_denoising_batch=use_denoising_batch,
        cfg_type=cfg_type,
        seed=seed,
        lcm_lora_id=lcm_lora_id,
        vae_id=vae_id,
        gpu_id=gpu_id,
        scheduler_name=scheduler_name,
        use_karras_sigmas=use_karras_sigmas,
        device=device,
        use_controlnet=use_controlnet,
        controlnet_model=controlnet_model,
        use_cached_attn=use_cached_attn,
        sd_model_type=sd_model_type,
        textual_inversion_dict=textual_inversion_dict
    )
    if lora_dict is not None:
        stream.update_lora_weights(lora_dict, print_info=True)
    current_prompt = prompt
    current_prompt_list = shared_data.get("prompt_list", [[prompt, 1.0]])
    current_seed_list = shared_data.get("seed_list", [[seed, 1.0]])
    shared_data['t_list'] = t_index_list 
    o_index = t_index_list
    last_list = t_index_list
    noise_bank = {}
    prompt_cache = {}
    input_memory = None
    output_memory = None
    control_memory = None
    start_time = time.time()
    prompt_changed = False
    frame_count = 0
    transmit_count = 0
    output_mem_name = f"sd_output_{int(time.time())}"
    loaded_lora_models = {}
    v2v_state = False
    process_frame = False
    interpolation_method = "average"
    print('Preparing Stream...')
    stream.prepare(
        prompt=current_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=guidance_scale,
        delta=delta,
    )
    time.sleep(.5)
    if device == "cuda":
        print_gpu_info(gpu_id=gpu_id)

    TIME_DEBUG = False
    if TIME_DEBUG:
        def timer(func):
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                end = time.time()
                print(f"{func.__name__} took {end - start:.4f} seconds")
                return result
            return wrapper
        timing_stats = {
            'parameter_update': [],
            'input_processing': [],
            'image_generation': [],
            'postprocessing': [],
            'shared_memory': [],
            'osc_messages': []
        }
        loop_times = []
    previous_frame = None
    frame_changed = False
    input_tensor = torch.empty((3, height, width), dtype=torch.float32, device=device)
    control_input_tensor = torch.empty((3, height, width), dtype=torch.float32, device=device)
    
    # Check platform once and set stream method
    is_macos = platform.system() == 'Darwin'
    if is_macos:
        stream_method = "syphon"
        # print("\033[36mMacOS detected - using Syphon for frame transfer\033[0m")
    else:
        stream_method = "shared_mem"
        # print("\033[36mUsing shared memory for frame transfer\033[0m")

    # Initialize stream handlers
    ndi_handler = None
    syphon_handler = None
    
    if stream_method == "ndi":
        from ndi_utils import NDIUtils
        print("\033[36mUsing NDI for frame transfer\033[0m")
        ndi_handler = NDIUtils(
            sender_name=output_mem_name,
            input_name=input_mem_name,
            control_name=input_mem_name + "-cn"
        )
        ndi_handler.start()
    elif stream_method == "syphon":
        from syphon_utils import SyphonUtils
        syphon_handler = SyphonUtils(
            sender_name=output_mem_name,
            input_name=input_mem_name,
            control_name=input_mem_name + "-cn",
            width=width,  # Pass the width from main
            height=height  # Pass the height from main
        )
        syphon_handler.start()
    else:
        # print("\033[36mUsing shared memory for frame transfer\033[0m")  
        # Initialize memory buffers (no pinning on macOS)
        if is_macos:
            pinned_input_buffer = torch.empty((height, width, 3), dtype=torch.uint8)
            pinned_control_buffer = torch.empty((height, width, 3), dtype=torch.uint8)
            # print("\033[36mMacOS detected - using standard memory buffers\033[0m")
        else:
            pinned_input_buffer = torch.empty((height, width, 3), dtype=torch.uint8, pin_memory=True)
            pinned_control_buffer = torch.empty((height, width, 3), dtype=torch.uint8, pin_memory=True)
        
    shared_data["loading_status"] = False
    while True:
        try:
            if TIME_DEBUG:
                param_update_start = time.time()#timer stats
            loop_start_time = time.time()#timer stats
            feedback_safe = shared_data.get("feedback_safe", False)
            new_sdmode = shared_data.get("sdmode", mode)
            if new_sdmode != mode:
                mode = new_sdmode
            new_lora_weights = shared_data.get("lora_weights", lora_dict)
            if new_lora_weights != lora_dict:
                lora_dict = new_lora_weights
                stream.update_lora_weights(lora_dict, print_info=True)
            # PROMPT DICT + GUIDANCE SCALE + DELTA
            new_guidance_scale = float(shared_data.get("guidance_scale", guidance_scale))
            new_delta = float(shared_data.get("delta", delta))
            new_prompt_list = shared_data.get("prompt_list", {})
            new_negative_prompt = shared_data.get("negative_prompt", negative_prompt)
            new_interpolation_method = shared_data.get("interpolation_method", "average")
            # Add debug prints for parameter changes
            # if new_guidance_scale != guidance_scale:
            #     print(f"\033[33mUpdating guidance_scale: {guidance_scale} -> {new_guidance_scale}\033[0m")
            # if new_prompt_list != current_prompt_list:
            #     print(f"\033[33mUpdating prompt_list: {current_prompt_list} -> {new_prompt_list}\033[0m")

            if (new_prompt_list != current_prompt_list or 
                new_guidance_scale != guidance_scale or 
                new_delta != delta or 
                new_negative_prompt != negative_prompt or
                new_interpolation_method != interpolation_method
                or frame_count < 10):
                current_prompt_list = new_prompt_list
                guidance_scale = new_guidance_scale
                delta = new_delta
                negative_prompt = new_negative_prompt
                interpolation_method = shared_data.get("interpolation_method", "average")
                update_combined_prompts_and_parameters(
                    stream.stream, 
                    current_prompt_list, 
                    guidance_scale, 
                    delta, 
                    negative_prompt,
                    prompt_cache,
                    interpolation_method,
                    shared_data
                )
            # TEXTUAL INVERSIONS DICT - ti_dict
            new_ti_dict = shared_data.get("textual_inversion_dict", None)
            if new_ti_dict is not None:
                # Convert loaded_textual_inversions to same format as new_ti_dict for comparison
                current_ti_dict = {
                    info["path"]: info["token"]
                    for info in getattr(stream, 'loaded_textual_inversions', {}).values()
                }
                
                if new_ti_dict != current_ti_dict:
                    # print(f"\n\033[36m...Updating Textual Inversions\033[0m")
                    stream.update_textual_inversions(new_ti_dict, print_info=True)
            ##SEED DICT
            new_seed_list = shared_data.get("seed_list", current_seed_list)
            # if new_seed_list != current_seed_list:
                # print(f"\033[33mUpdating seed_list: {current_seed_list} -> {new_seed_list}\033[0m")
            if new_seed_list != current_seed_list:
                current_seed_list = new_seed_list
                if any(weight > 0 for _, weight in current_seed_list):
                    blended_noise = blend_noise_tensors(current_seed_list, noise_bank, stream.stream)
                    stream.stream.init_noise = blended_noise
            ##T_LIST
            new_t_list = shared_data.get('t_list', t_index_list)
            list_update = False
            if len(new_t_list) != len(o_index):
                list_update = False
            else:
                list_update = True
            if list_update:
                if new_t_list != last_list:
                    last_list = new_t_list
                    update_t_list_attributes(stream.stream, new_t_list)
            #V2V
            if not is_macos:
                disable_cached_attn = shared_data.get("disable_cached_attn", True)
                if disable_cached_attn and not v2v_state:
                    update_attn_processors(stream, {}, disable_cached_attn=True, device=device)
                    stream.stream.use_cached_attn = False
                    stream.use_cached_attn_settings = {}
                    v2v_state = True
                elif not disable_cached_attn and v2v_state:
                    v2v_state = False
                    new_use_cached_attn = shared_data.get("use_cached_attn", False)
                    new_use_cached_attn_settings = shared_data.get("use_cached_attn_settings", {})
                    if new_use_cached_attn:
                        stream.stream.use_cached_attn = new_use_cached_attn
                        stream.use_cached_attn_settings = new_use_cached_attn_settings.copy()
                        update_attn_processors(stream.stream, new_use_cached_attn_settings, disable_cached_attn=False, device=device)
                elif not disable_cached_attn:
                    if "use_cached_attn" in shared_data:
                        new_use_cached_attn = shared_data.get("use_cached_attn", False)
                        if new_use_cached_attn != stream.stream.use_cached_attn:
                            stream.stream.use_cached_attn = new_use_cached_attn
                    new_use_cached_attn_settings = shared_data.get("use_cached_attn_settings", {})
                    if new_use_cached_attn_settings != stream.use_cached_attn_settings:
                        stream.use_cached_attn_settings = new_use_cached_attn_settings.copy()
                        update_attn_processors(stream.stream, new_use_cached_attn_settings, disable_cached_attn=False, device=device)
            new_controlnet_conditioning_scale = shared_data.get("controlnet_conditioning_scale", stream.stream.controlnet_conditioning_scale)
            if new_controlnet_conditioning_scale != stream.stream.controlnet_conditioning_scale:
                update_controlnet_conditioning_scale(stream.stream, shared_data)
            new_use_controlnet = shared_data.get("use_controlnet", stream.stream.use_controlnet)
            if new_use_controlnet != stream.stream.use_controlnet:
                stream.stream.use_controlnet = new_use_controlnet
            if TIME_DEBUG:
                timing_stats['parameter_update'].append(time.time() - param_update_start) #timer stats
                input_processing_start = time.time() #timer stats
            process_frame = shared_data.get("process_frame", False)
            paused = shared_data.get("paused", False)
            if paused:
                if not process_frame:
                    time.sleep(.001)
                    continue
            shared_data["process_frame"] = False
            # Controlnet frame
            if use_controlnet:
                if stream_method == "ndi":
                    control_frame = ndi_handler.capture_control_frame()
                    if control_frame is not None:
                        control_input_tensor.copy_(torch.from_numpy(control_frame).permute(2, 0, 1).float().div(255))
                elif stream_method == "syphon":
                    control_frame = syphon_handler.capture_control_frame()
                    if control_frame is not None:
                        control_input_tensor.copy_(torch.from_numpy(control_frame).permute(2, 0, 1).float().div(255))
                else:
                    control_mem_name = input_mem_name + '-cn'
                    if 'control_mem_name' in shared_data and shared_data['control_mem_name'] != control_mem_name:
                        if control_memory is not None:
                            control_memory.close()
                            control_memory = None
                        control_mem_name = shared_data['control_mem_name']
                    if control_memory is None:
                        try:
                            control_memory = shared_memory.SharedMemory(name=control_mem_name)
                        except FileNotFoundError:
                            print(f"\Controlnet Stream '{input_mem_name}' not found. Try changing the Stream Out Name (to SD) parameter in Stream Settings 2 of TD operator.")
                            continue
                    control_total_size_bytes = control_memory.size
                    control_buffer = np.ndarray(shape=(control_total_size_bytes,), dtype=np.uint8, buffer=control_memory.buf)
                    control_image_data_size = width * height * 3
                    control_frame_np = control_buffer[:control_image_data_size].reshape((height, width, 3))
                    if is_macos:
                        control_input_tensor.copy_(torch.from_numpy(control_frame_np).permute(2, 0, 1).float().div(255))
                    else:
                        pinned_control_buffer.copy_(torch.from_numpy(control_frame_np))
                        control_input_tensor.copy_(pinned_control_buffer.permute(2, 0, 1).float().div(255), non_blocking=True)
            # img2img mode
            if mode == "img2img":
                if stream_method == "ndi":
                    frame = ndi_handler.capture_input_frame()
                    if frame is not None:
                        try:
                            input_tensor.copy_(torch.from_numpy(frame).permute(2, 0, 1).float().div(255))
                        except Exception as e:
                            print(f"\033[91mError processing NDI input frame: {e}\033[0m")
                elif stream_method == "syphon":
                    frame = syphon_handler.capture_input_frame()
                    if frame is not None:
                        try:
                            input_tensor.copy_(torch.from_numpy(frame).permute(2, 0, 1).float().div(255))
                        except Exception as e:
                            print(f"\033[91mError processing Syphon input frame: {e}\033[0m")
                else:
                    if 'input_mem_name' in shared_data and shared_data['input_mem_name'] != input_mem_name:
                        if input_memory is not None:
                            input_memory.close()
                            input_memory = None
                        input_mem_name = shared_data['input_mem_name']
                    if input_memory is None:
                        try:
                            input_memory = shared_memory.SharedMemory(name=input_mem_name)
                            print(f"\033[32m=======================================\033[0m")
                            print(f"\033[32mInput Stream '{input_mem_name}' found.\033[0m")
                            print(f"\033[32m=======================================\033[0m")
                        except FileNotFoundError:
                            print(f"\nInput Stream '{input_mem_name}' not found. Try changing the Stream Out Name (to SD) parameter in Stream Settings 2 of TD operator.")
                            continue
                    total_size_bytes = input_memory.size
                    buffer = np.ndarray(shape=(total_size_bytes,), dtype=np.uint8, buffer=input_memory.buf)
                    image_data_size = width * height * 3
                    frame_np = buffer[:image_data_size].reshape((height, width, 3))
                    try:    
                        if is_macos:
                            input_tensor.copy_(torch.from_numpy(frame_np).permute(2, 0, 1).float().div(255))
                        else:
                            pinned_input_buffer.copy_(torch.from_numpy(frame_np))
                            input_tensor.copy_(pinned_input_buffer.permute(2, 0, 1).float().div(255), non_blocking=True)
                    except ValueError as e:
                        if "cannot reshape" in str(e):
                            print(f"\033[91mError: {e}\033[0m")
                            print("\033[91mThere might be a different StreamDiffusionTD operator open that is sending feed with the same name.")
                            print("Change the Streamoutname parameter on the one you are sending from and try relaunching.\033[0m")
                        else:
                            print(f"ValueError: {e}")
                    except Exception as e:
                        print(f"Error copying input buffer: {e}")
                if TIME_DEBUG:
                    timing_stats['input_processing'].append(time.time() - input_processing_start) #timer stats
                    image_generation_start = time.time() #timer stats
                if use_controlnet:
                    processed_tensor = stream.stream(input_tensor, control_input_tensor)
                    if feedback_safe:
                        processed_tensor = stream.stream(input_tensor, control_input_tensor)
                else:
                    processed_tensor = stream.stream(input_tensor)
                    if feedback_safe:
                        processed_tensor = stream.stream(input_tensor)
            # txt2img mode
            elif mode == "txt2img":
                if TIME_DEBUG:
                    timing_stats['input_processing'].append(time.time() - input_processing_start) #timer stats
                    image_generation_start = time.time() #timer stats
                processed_np = custom_txt2img_using_prepared_noise(stream_diffusion=stream.stream, expected_batch_size=1, output_type='np', control_image=control_input_tensor if use_controlnet else None)

            if TIME_DEBUG:
                timing_stats['image_generation'].append(time.time() - image_generation_start)
                postprocessing_start = time.time()

            if mode == "img2img":
                processed_np = optimized_postprocess_image(processed_tensor, output_type="np")
            elif mode == "txt2img":
                if processed_np.max() <= 1.0:
                    processed_np = (processed_np * 255).astype(np.uint8)

            # # Add frame comparison here
            # if previous_frame is not None:
            #     # Compare current frame with previous frame
            #     frame_diff = np.mean(np.abs(processed_np - previous_frame))
            #     frame_changed = frame_diff > 0.01  # Threshold for considering frames different
            #     print(f"\033[36mFrame {frame_count} - Changed: {frame_changed} (diff: {frame_diff:.4f})\033[0m")
            # else:
            #     print("\033[36mFirst frame generated\033[0m")
            #     frame_changed = True

            # # Store current frame for next comparison
            # previous_frame = processed_np.copy()

            # if frame_changed:
            #     send_osc_message('/frame_updated', 1, osc_transmit_port)
            if TIME_DEBUG:
                timing_stats['postprocessing'].append(time.time() - postprocessing_start)
                shared_memory_start = time.time()

            # Output frame based on stream method
            if stream_method == "ndi":
                if processed_np.shape[2] == 3:
                    # Add alpha channel for NDI
                    alpha = np.full((processed_np.shape[0], processed_np.shape[1], 1), 255, dtype=np.uint8)
                    processed_np = np.concatenate([processed_np, alpha], axis=2)
                ndi_handler.transmit_frame(processed_np)
                # print(f"\033[32mTransmitted frame {frame_count} via NDI\033[0m")
            elif stream_method == "syphon":
                syphon_handler.transmit_frame(processed_np)
                # print(f"\033[32mTransmitted frame {frame_count} via Syphon\033[0m")
            else:
                if output_memory is None:
                    print(f"\033[35mCreating shared memory segment: {output_mem_name}\033[0m")
                    output_memory = shared_memory.SharedMemory(name=output_mem_name, create=True, size=processed_np.nbytes)

                try:
                    output_array = np.ndarray(processed_np.shape, dtype=processed_np.dtype, buffer=output_memory.buf)
                    output_array[:] = processed_np[:]
                    # print(f"\033[32mWrote frame {frame_count} to shared memory (size: {processed_np.nbytes})\033[0m")
                except Exception as e:
                    print(f"\033[91mError writing to shared memory: {e}\033[0m")

            if TIME_DEBUG:
                timing_stats['shared_memory'].append(time.time() - shared_memory_start) #timer stats
                osc_messages_start = time.time() #timer stats
            send_osc_message('/framecount', frame_count, osc_transmit_port)
            start_time, transmit_count = calculate_fps_and_send_osc(start_time, transmit_count, osc_transmit_port, output_mem_name, True, use_controlnet)
            if TIME_DEBUG:
                timing_stats['osc_messages'].append(time.time() - osc_messages_start) #timer stats
            frame_count += 1
            ##STOP STREAM
            if shared_data.get("stop_stream", False):
                print("\r\n")
                print("Stopping image generation process.")
                break
            # print(f"frame_count: {frame_count}")
            try:
                if TIME_DEBUG:
                    loop_end_time = time.time()
                    loop_time = loop_end_time - loop_start_time
                    loop_times.append(loop_time)
                    if frame_count % 100 == 0:
                        print("\nDetailed timing (average over last 100 frames):")
                        avg_loop_time = sum(loop_times) / len(loop_times)
                        actual_fps = 1 / avg_loop_time if avg_loop_time > 0 else 0
                        print(f"Actual FPS (based on total loop time): {actual_fps:.2f}")
                        
                        total_time = sum(timing_stats['image_generation'])
                        approx_fps = 100 / total_time if total_time > 0 else 0
                        print(f"Approximate FPS (based on image generation): {approx_fps:.2f}")
                        
                        for key, times in timing_stats.items():
                            avg_time = sum(times) / len(times)
                            print(f"{key}: {avg_time:.4f} seconds")
                        
                        unaccounted_time = avg_loop_time - sum(sum(times) / len(times) for times in timing_stats.values())
                        print(f"Unaccounted time: {unaccounted_time:.4f} seconds")
                        
                        timing_stats = {key: [] for key in timing_stats}  # Reset timing stats
                        loop_times = []  # Reset loop times
            except Exception as e:
                pass
            else:
                max_fps = shared_data.get("max_fps", 60)
                if max_fps <= 0.5:
                    max_fps = 0.5
                frame_time = time.time() - loop_start_time
                sleep_time = max(1.0 / max_fps - frame_time, 0.0001)
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\r", end='')
            break



def optimized_postprocess_image(
    image: torch.Tensor,
    output_type: str = "np",
    do_denormalize: bool = True
) -> Union[torch.Tensor, np.ndarray]:
    if not isinstance(image, torch.Tensor):
        raise ValueError(f"Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor")

    if output_type == "latent":
        return image

    # Check device type and use appropriate autocast
    if image.device.type == 'cuda':
        autocast_context = torch.cuda.amp.autocast(enabled=True)
    elif image.device.type == 'mps':
        autocast_context = torch.amp.autocast(device_type='cpu', enabled=True)
    else:
        autocast_context = nullcontext()

    with autocast_context:
        if do_denormalize:
            image = (image / 2 + 0.5).clamp(0, 1)

        if output_type == "pt":
            return image

        if output_type == "np":
            image = image.mul(255).byte()
            # Move to CPU if on MPS or CUDA
            if image.device.type in ['mps', 'cuda']:
                image = image.cpu()
            return image.permute(0, 2, 3, 1).numpy()

    raise ValueError(f"Unsupported output type: {output_type}")

def update_lora_weights(stream_diffusion, lora_dict, use_lcm_lora, lcm_lora_id, wrapper):
    if lora_dict is not None:
        try:
            wrapper.update_lora_weights(lora_dict)
            if use_lcm_lora:
                if lcm_lora_id:
                    stream_diffusion.load_lcm_lora(lcm_lora_id)
                else:
                    stream_diffusion.load_lcm_lora()
        except Exception as e:
            error_str = str(e)
            if "size mismatch" in error_str:
                # Extract model architecture info from error
                current_size = None
                target_size = None
                if "shape in current model is" in error_str:
                    try:
                        current_size = re.search(r'shape in current model is torch.Size\(\[(.*?)\]\)', error_str).group(1)
                    except:
                        pass
                
                # Determine model types based on size patterns
                model_type = "unknown"
                if current_size and "1024" in current_size:
                    model_type = "SDXL"
                elif current_size and "768" in current_size:
                    model_type = "SD1.5"
                
                print("\n\033[91m=============== LoRA Loading Error ===============\033[0m")
                print(f"\033[91mArchitecture mismatch detected!\033[0m")
                print("\n\033[93mDetails:\033[0m")
                print(f"• Base Model: {model_type}")
                for lora_name in lora_dict.keys():
                    print(f"• Failed LoRA: {os.path.basename(lora_name)}")
                print("\n\033[93mProblem:\033[0m")
                print("The LoRA model architecture doesn't match your base model.")
                print("\n\033[93mSolution:\033[0m")
                print("Please ensure you're using:")
                print("• SD1.5 LoRAs with SD1.5 models")
                print("• SD2.1 LoRAs with SD2.1 models")
                print("• SDXL LoRAs with SDXL models")
                print("\033[91m===============================================\033[0m\n")
            elif "already in use in the Unet" in error_str:
                # Handle the duplicate adapter name issue
                print("\n\033[93mNote: Attempting to clean up existing LoRA adapters...\033[0m")
                try:
                    # Try to unload existing adapters before loading new ones
                    stream_diffusion.unload_all_loras()
                    # Retry loading after cleanup
                    wrapper.update_lora_weights(lora_dict)
                    print("\033[92mSuccessfully reloaded LoRA after cleanup.\033[0m\n")
                except Exception as retry_error:
                    print(f"\033[91mFailed to recover from LoRA loading error: {retry_error}\033[0m\n")
            else:
                print(f'\033[91mCaution: Error in update_lora_weights: {e}\033[0m')
def update_attn_processors(stream_diffusion, new_use_cached_attn_settings, disable_cached_attn, device):
    if disable_cached_attn:
        if stream_diffusion.original_processors is not None:
            for key, processor in stream_diffusion.stream.pipe.unet.attn_processors.items():
                if isinstance(processor, (CachedSTXFormersAttnProcessor, CachedSTAttnProcessor2_0)):
                    processor.clear_cache()
            stream_diffusion.stream.pipe.unet.set_attn_processor(stream_diffusion.original_processors.copy())
            # Only run CUDA-specific commands if we're on CUDA
            if device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            return
    attn_processors = stream_diffusion.pipe.unet.attn_processors
    new_attn_processors = {}
    try:
        for key, attn_processor in attn_processors.items():
            if isinstance(attn_processor, CachedSTXFormersAttnProcessor):
                attn_processor.use_feature_injection = new_use_cached_attn_settings.get("use_feature_injection", True)
                attn_processor.fi_strength = new_use_cached_attn_settings.get("feature_injection_strength", 0.5)
                attn_processor.threshold = new_use_cached_attn_settings.get("feature_similarity_threshold", 0.9)
                attn_processor.interval = new_use_cached_attn_settings.get("cache_interval", 2)
                new_max_frames = new_use_cached_attn_settings.get("cache_maxframes", 1)
                if new_max_frames != attn_processor.cached_key.maxlen:
                    attn_processor.cached_key = deque(attn_processor.cached_key, maxlen=new_max_frames)
                    attn_processor.cached_value = deque(attn_processor.cached_value, maxlen=new_max_frames)
                    attn_processor.cached_output = deque(attn_processor.cached_output, maxlen=new_max_frames)
                attn_processor.use_tome_cache = new_use_cached_attn_settings.get("use_tome_cache", True)
                attn_processor.tome_metric = new_use_cached_attn_settings.get("tome_metric", "keys")
                attn_processor.use_grid = new_use_cached_attn_settings.get("use_grid", False)
                attn_processor.tome_ratio = 0.5 if attn_processor.use_grid else new_use_cached_attn_settings.get("tome_ratio", 0.5)
                new_attn_processors[key] = attn_processor
            else:
                new_attn_processors[key] = CachedSTXFormersAttnProcessor(
                    name=key,
                    use_feature_injection=new_use_cached_attn_settings.get("use_feature_injection", True),
                    feature_injection_strength=new_use_cached_attn_settings.get("feature_injection_strength", 0.5),
                    feature_similarity_threshold=new_use_cached_attn_settings.get("feature_similarity_threshold", 0.9),
                    interval=new_use_cached_attn_settings.get("cache_interval", 2),
                    max_frames=new_use_cached_attn_settings.get("cache_maxframes", 1),
                    use_tome_cache=new_use_cached_attn_settings.get("use_tome_cache", True),
                    tome_metric=new_use_cached_attn_settings.get("tome_metric", "keys"),
                    tome_ratio=0.5 if new_use_cached_attn_settings.get("use_grid", False) else new_use_cached_attn_settings.get("tome_ratio", 0.5),
                    use_grid=new_use_cached_attn_settings.get("use_grid", False),
                )
        stream_diffusion.pipe.unet.set_attn_processor(new_attn_processors)
    except Exception as e:
        return


def update_controlnet_conditioning_scale(stream_diffusion, shared_data):
    new_controlnet_conditioning_scale = shared_data.get("controlnet_conditioning_scale", stream_diffusion.controlnet_conditioning_scale)
    if new_controlnet_conditioning_scale != stream_diffusion.controlnet_conditioning_scale:
        stream_diffusion.controlnet_conditioning_scale = new_controlnet_conditioning_scale

def update_t_list_attributes(stream_diffusion_instance, new_t_list):
    stream_diffusion_instance.t_list = new_t_list
    stream_diffusion_instance.sub_timesteps = [stream_diffusion_instance.timesteps[t] for t in new_t_list]
    sub_timesteps_tensor = torch.tensor(
        stream_diffusion_instance.sub_timesteps, dtype=torch.long, device=stream_diffusion_instance.device
    )
    stream_diffusion_instance.sub_timesteps_tensor = torch.repeat_interleave(
        sub_timesteps_tensor, 
        repeats=stream_diffusion_instance.frame_bff_size if stream_diffusion_instance.use_denoising_batch else 1, 
        dim=0
    )
    c_skip_list = []
    c_out_list = []
    for timestep in stream_diffusion_instance.sub_timesteps:
        c_skip, c_out = stream_diffusion_instance.scheduler.get_scalings_for_boundary_condition_discrete(timestep)
        c_skip_list.append(c_skip)
        c_out_list.append(c_out)
    stream_diffusion_instance.c_skip = torch.stack(c_skip_list).view(len(new_t_list), 1, 1, 1).to(dtype=stream_diffusion_instance.dtype, device=stream_diffusion_instance.device)
    stream_diffusion_instance.c_out = torch.stack(c_out_list).view(len(new_t_list), 1, 1, 1).to(dtype=stream_diffusion_instance.dtype, device=stream_diffusion_instance.device)
    # Calculate alpha_prod_t_sqrt and beta_prod_t_sqrt
    alpha_prod_t_sqrt_list = []
    beta_prod_t_sqrt_list = []
    for timestep in stream_diffusion_instance.sub_timesteps:
        alpha_prod_t_sqrt = stream_diffusion_instance.scheduler.alphas_cumprod[timestep].sqrt()
        beta_prod_t_sqrt = (1 - stream_diffusion_instance.scheduler.alphas_cumprod[timestep]).sqrt()
        alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
        beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
    alpha_prod_t_sqrt = torch.stack(alpha_prod_t_sqrt_list).view(len(new_t_list), 1, 1, 1).to(dtype=stream_diffusion_instance.dtype, device=stream_diffusion_instance.device)
    beta_prod_t_sqrt = torch.stack(beta_prod_t_sqrt_list).view(len(new_t_list), 1, 1, 1).to(dtype=stream_diffusion_instance.dtype, device=stream_diffusion_instance.device)
    stream_diffusion_instance.alpha_prod_t_sqrt = torch.repeat_interleave(alpha_prod_t_sqrt, repeats=stream_diffusion_instance.frame_bff_size if stream_diffusion_instance.use_denoising_batch else 1, dim=0)
    stream_diffusion_instance.beta_prod_t_sqrt = torch.repeat_interleave(beta_prod_t_sqrt, repeats=stream_diffusion_instance.frame_bff_size if stream_diffusion_instance.use_denoising_batch else 1, dim=0)

def slerp(t, v0, v1, dot_threshold=0.9995):
    dot = torch.sum(v0 * v1 / (torch.norm(v0) * torch.norm(v1)))
    if abs(dot) > dot_threshold:
        result = v0 + t * (v1 - v0)
        return result / torch.norm(result)
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    return s0 * v0 + s1 * v1

def multi_slerp(embeddings, weights):

    total_weight = sum(weights)
    scale_factor = max(1.0, total_weight)
    if len(embeddings) == 1:
        return embeddings[0] * scale_factor
    scaled_weights = [w / scale_factor for w in weights]
    sorted_pairs = sorted(zip(embeddings, scaled_weights), key=lambda x: x[1], reverse=True)
    sorted_embeddings, sorted_weights = zip(*sorted_pairs)
    result = sorted_embeddings[0]
    accumulated_weight = sorted_weights[0]
    for i in range(1, len(sorted_embeddings)):
        if sorted_weights[i] == 0:
            continue
        t = sorted_weights[i] / (accumulated_weight + sorted_weights[i])
        result = slerp(t, result, sorted_embeddings[i])
        accumulated_weight += sorted_weights[i]
    return result * scale_factor

def cosine_similarity_weighted_average(embeddings, weights):
    embeddings_tensor = torch.stack(embeddings)
    weights_tensor = torch.tensor(weights, device=embeddings_tensor.device, dtype=embeddings_tensor.dtype)
    result = multi_slerp(embeddings, weights)
    return result

def interpolate_embeddings(embeddings, weights, method='slerp', sdxl=False, **kwargs):
    if sdxl:
        return sum(embed * weight for embed, weight in zip(embeddings, weights))
    if method == 'slerp':
        return multi_slerp(embeddings, weights)
    elif method == 'cosine_weighted_interpolation':
        return cosine_similarity_weighted_average(embeddings, weights)
    else:
        return sum(embed * weight for embed, weight in zip(embeddings, weights))

@torch.no_grad()
def update_combined_prompts_and_parameters(stream_diffusion, prompt_list, new_guidance_scale, new_delta, new_negative_prompt, prompt_cache, interpolation_method, shared_data):
    stream_diffusion.guidance_scale = new_guidance_scale
    stream_diffusion.delta = new_delta
    stream_diffusion.stock_noise *= stream_diffusion.delta
    sdxl = stream_diffusion.sdxl
    current_prompts = set()
    embeddings = []
    weights = []
    for idx, (prompt_text, weight) in enumerate(prompt_list):
        if weight == 0:
            continue
        current_prompts.add(idx)
        if idx not in prompt_cache or prompt_cache[idx]['text'] != prompt_text:
            encoder_output = stream_diffusion.pipe.encode_prompt(
                prompt=prompt_text,
                device=stream_diffusion.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=new_negative_prompt,
            )
            prompt_cache[idx] = {'embed': encoder_output[0], 'text': prompt_text}
        embeddings.append(prompt_cache[idx]['embed'])
        weights.append(weight)
    Interpval1 = shared_data.get("Interpval1", 0.5)
    if embeddings and weights:
        embeddings, weights = zip(*sorted(zip(embeddings, weights), key=lambda x: x[1], reverse=True))
        if embeddings:
            combined_embeds = interpolate_embeddings(embeddings, weights, method=interpolation_method, val1=Interpval1, sdxl=sdxl, shared_data=shared_data)
            stream_diffusion.prompt_embeds = combined_embeds.repeat(stream_diffusion.batch_size, 1, 1)
    unused_prompts = set(prompt_cache.keys()) - current_prompts
    for prompt in unused_prompts:
        del prompt_cache[prompt]

def blend_noise_tensors(seed_list, noise_bank, stream_diffusion):
    blended_noise = None
    total_weight = 0
    for seed_val, weight in seed_list:
        if weight == 0:
            continue
        noise_tensor = noise_bank.get(seed_val)
        if noise_tensor is None:
            generator = torch.Generator().manual_seed(seed_val)
            noise_tensor = torch.randn(
                (stream_diffusion.batch_size, 4, stream_diffusion.latent_height, stream_diffusion.latent_width),
                generator=generator
            ).to(device=stream_diffusion.device, dtype=stream_diffusion.dtype)
            noise_bank[seed_val] = noise_tensor
        if blended_noise is None:
            blended_noise = noise_tensor * weight
        else:
            blended_noise += noise_tensor * weight
        total_weight += weight
    return blended_noise

def custom_txt2img_using_prepared_noise(stream_diffusion, expected_batch_size, output_type='np', control_image=None):
    if stream_diffusion.init_noise.size(0) > expected_batch_size:
        adjusted_noise = stream_diffusion.init_noise[:expected_batch_size]
    elif stream_diffusion.init_noise.size(0) < expected_batch_size:
        repeats = [expected_batch_size // stream_diffusion.init_noise.size(0)] + [-1] * (stream_diffusion.init_noise.dim() - 1)
        adjusted_noise = stream_diffusion.init_noise.repeat(*repeats)[:expected_batch_size]
    else:
        adjusted_noise = stream_diffusion.init_noise
    x_0_pred_out = stream_diffusion.predict_x0_batch(adjusted_noise, control_image=control_image)
    x_output = stream_diffusion.decode_image(x_0_pred_out).detach().clone()
    if output_type == 'np':
        x_output = optimized_postprocess_image(x_output, output_type=output_type)
    return x_output

def safe_basename(path):
    return os.path.basename(path) if path else "Not specified"

def print_info(label, value=None, weight=None, start_phrase=None, color="green"):
    color_codes = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "default": "\033[39m"
    }
    color_code = color_codes.get(color, "\033[32m")  # Default to green if color not found
    if start_phrase:
        print(start_phrase)
    if value and value != "None":
        base_name = safe_basename(value)
        weight_info = f", Weight: {weight}" if weight is not None else ""
        print(f"{label} {base_name}{weight_info}")
        print(f"{color_code}{value}\033[0m\n")

def read_config(config_file_path):
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)
    
    # Determine device and acceleration based on platform
    if platform.system() == "Darwin":  # Mac
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        acceleration = "none"  # Force no acceleration on Mac
        use_cached_attn = False
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        acceleration = config["acceleration"]
        use_cached_attn = config.get("use_cached_attn", True)

    # Normalize file paths to always use forward slashes
    def normalize_path(path):
        if not path:
            return path
        # Convert path to absolute path with forward slashes
        abs_path = os.path.abspath(os.path.expanduser(path)).replace('\\', '/')
        # print(f"Normalizing path: {path} -> {abs_path}")  # Debug print
        return abs_path

    model_path = normalize_path(config['model_id_or_path'])
    controlnet_path = normalize_path(config.get('controlnet_model'))
    lcm_lora_path = normalize_path(config.get('lcm_lora_id'))

    # Normalize LoRA paths
    lora_dict = {}
    for lora_path, weight in config.get('lora_dict', {}).items():
        normalized_path = normalize_path(lora_path)
        lora_dict[normalized_path] = weight

    # Normalize textual inversion paths
    ti_dict = {}
    if config.get('textual_inversion_dict'):
        for embed_path, token in config['textual_inversion_dict'].items():
            normalized_path = normalize_path(embed_path)
            ti_dict[normalized_path] = token

    config_dict = {
        "osc_receive_port": config.get("osc_out_port", 8247),
        "osc_transmit_port": config.get("osc_in_port", 8248),
        "model_id_or_path": model_path,
        "lora_dict": lora_dict,
        "prompt": config["prompt"],
        "negative_prompt": config["negative_prompt"],
        "frame_buffer_size": config["frame_buffer_size"],
        "width": config["width"],
        "height": config["height"],
        "acceleration": acceleration,
        "use_denoising_batch": config["use_denoising_batch"],
        "seed": config["seed"],
        "cfg_type": config["cfg_type"],
        "guidance_scale": config["guidance_scale"],
        "delta": config["delta"],
        "do_add_noise": config["do_add_noise"],
        "enable_similar_image_filter": config["enable_similar_image_filter"],
        "similar_image_filter_threshold": config["similar_image_filter_threshold"],
        "similar_image_filter_max_skip_frame": config["similar_image_filter_max_skip_frame"],
        "t_index_list": config.get("t_index_list", [25, 40]),
        "mode": config.get("sdmode", "img2img"),
        "lcm_lora_id": lcm_lora_path,
        "vae_id": config.get("vae_id"),
        "scheduler_name": config.get("scheduler_name", "EulerAncestral"),
        "use_karras_sigmas": config.get("use_karras_sigmas", False),
        "input_mem_name": config["input_mem_name"],
        "device": device,
        "gpu_id": config.get("gpu_id", 0),
        "use_controlnet": config.get("use_controlnet", False),
        "controlnet_model": controlnet_path,
        "controlnet_weight": config.get("controlnet_weight", 0.5),
        "use_cached_attn": use_cached_attn,
        "sd_model_type": config.get("sd_model_type", None),
        "max_fps": config.get("max_fps", 60),
        "hf_cache": config.get("hf_cache", None),
        "textual_inversion_dict": ti_dict
    }
    return config_dict

def print_config_info(config, config_file_path):
    time.sleep(0.04)
    print(f"\033[36mLoading {config_file_path}\033[0m")
    model_info = f"{config['model_id_or_path']}" if config['sd_model_type'] is None else f"{config['model_id_or_path']} ({config['sd_model_type']})"
    time.sleep(0.04)
    print_info("Main SD Model:", model_info, start_phrase="\033[36m=======================================\033[0m")
    if config['lcm_lora_id'] != "skip":
        time.sleep(0.04)
        print_info("LCM LoRA ID:", config['lcm_lora_id'])
    time.sleep(0.04)
    print_info("VAE ID:", config['vae_id'])
    if config['use_controlnet']:
        time.sleep(0.04)
        print_info("ControlNet Model:", config['controlnet_model'], weight=config['controlnet_weight'])
    if config['lora_dict']:
        for model_path, weight in config['lora_dict'].items():
            time.sleep(0.04)
            print_info("LoRA Model:", model_path, weight)
    # Add this section for textual inversions
    if config['textual_inversion_dict']:
        for embed_path, token in config['textual_inversion_dict'].items():
            time.sleep(0.04)
            print_info("Textual Inversion:", embed_path, token if token else "(default token)")
    time.sleep(0.04)
    print("\033[36mStream name:\033[0m", config['input_mem_name'])
    if config['hf_cache']:
        time.sleep(0.04)
        print("\033[36mHuggingFace Model Folder:\033[0m", config['hf_cache'])
    time.sleep(0.04)
    print("\033[36m=======================================\033[0m\n")
    if config['acceleration'] == "tensorrt":
        time.sleep(0.04)
        print_info("Using TensorRT...", "V2V + ControlNet are not supported and will be disabled.", color="red")

def main():
    def signal_handler(sig, frame):
        print('Exiting...')
        terminate_processes([osc_process, generation_process])
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description="StreamDiffusionTD")
    parser.add_argument('-c', '--config', type=str, default='stream_config.json', help='Path to the configuration file')
    args = parser.parse_args()
    
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(current_script_dir, args.config)
    
    with Manager() as manager:
        print_sdtd_title()
        shared_data = manager.dict()
        
        config = read_config(config_file_path)
        shared_data.update(config)
        device = config['device']
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        os_name = "\033[36mWindows\033[0m" if platform.system() == "Windows" else "\033[35mmacOS\033[0m"
        print(f"\033[33mOperating System:\033[0m {os_name}")
        print(f"\033[33mUsing device:\033[0m {device}")
        if device == "cuda":
            print_gpu_info(gpu_id=config["gpu_id"])
        print()
        print_config_info(config, config_file_path)
        
        osc_process = Process(target=osc_server, args=(shared_data, '127.0.0.1', config['osc_receive_port'], config['osc_transmit_port']))
        osc_process.start()
        time.sleep(.3)
        
        generation_process = None

        def start_generation_process():
            nonlocal generation_process, config
            if generation_process is None or not generation_process.is_alive():
                new_config = read_config(config_file_path)
                if config != new_config:
                    config = new_config
                    print_config_info(config, config_file_path)
                    shared_data.update(config)
                generation_process = Process(
                    target=image_generation_process,
                    args=(
                        config['model_id_or_path'],
                        config['lora_dict'],
                        config['prompt'],
                        config['negative_prompt'],
                        config['frame_buffer_size'],
                        config['width'],
                        config['height'],
                        config['acceleration'],
                        config['use_denoising_batch'],
                        config['seed'],
                        config['cfg_type'],
                        config['guidance_scale'],
                        config['delta'],
                        config['do_add_noise'],
                        config['enable_similar_image_filter'],
                        config['similar_image_filter_threshold'],
                        config['similar_image_filter_max_skip_frame'],
                        shared_data,
                        config['t_index_list'],
                        config['mode'],
                        config['lcm_lora_id'],
                        config['vae_id'],
                        config['input_mem_name'],
                        config['osc_transmit_port'],
                        config['scheduler_name'],
                        config['use_karras_sigmas'],
                        config['device'],
                        config['gpu_id'],
                        config['use_controlnet'],
                        config['controlnet_model'],
                        config['controlnet_weight'],
                        config['use_cached_attn'],
                        config['sd_model_type'],
                        config['max_fps'],
                        config['textual_inversion_dict']
                    ),
                )
                generation_process.start()
        
        try:
            start_generation_process()
            while True:
                if shared_data.get("stop_stream", False):
                    print("\n\033[31mStop command received. Initiating shutdown...\033[0m")
                    break
                elif shared_data.get("pause_stream", False):
                    print("\n\033[33;1mPause command received. Pausing generation...\033[0m")
                    shared_data["paused"] = True
                    shared_data["pause_stream"] = False
                elif shared_data.get("play_stream", False):
                    if generation_process is None or not generation_process.is_alive():
                        print("\n\033[32mPlay command received. Starting generation process...\033[0m\n")
                        start_generation_process()
                    else:
                        print("\n\033[92mPlay command received. Resuming generation...\033[0m\n")
                    shared_data["paused"] = False
                    shared_data["play_stream"] = False
                elif shared_data.get("unload_stream", False):
                    if generation_process:
                        print("\n\033[35mUnload command received. Terminating generation process...\033[0m\n")
                        generation_process.terminate()
                        generation_process.join()
                        generation_process = None
                        print_gpu_info(gpu_id=config['gpu_id'])
                    else:
                        print("\n\033[35mUnload command received. No generation process to terminate.\033[0m\n")
                        print_gpu_info(gpu_id=config['gpu_id'])
                    shared_data["unload_stream"] = False
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n\033[31mKeyboardInterrupt received, signalling to stop subprocesses...\033[0m")
        finally:
            if generation_process:
                generation_process.terminate()
                generation_process.join()
            osc_process.terminate()
            osc_process.join()
            print("\n\033[36mAll subprocesses terminated. Exiting main process...\033[0m")
            sys.exit(0)

if __name__ == "__main__":
    fire.Fire(main)