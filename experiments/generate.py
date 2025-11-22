import torch
import time
import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
from initno.pipelines.pipeline_sd_initno import StableDiffusionInitNOPipeline

MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda"
OUTPUT_DIR = "results" 
SAFE_ITERATIONS = 10 

PROMPTS_DATA = [
    {"text": "a cat and a rabbit", "indices": [2, 5], "short": "Cat & Rabbit"},
    {"text": "a red suitcase and a yellow clock", "indices": [3, 7], "short": "Suitcase & Clock"},
    {"text": "A frog and a purple balloon", "indices": [2, 6], "short": "Frog & Balloon"},
    {"text": "A cat and a sunflower, Van Gogh style", "indices": [2, 5], "short": "Cat (Van Gogh)"}
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()

def load_or_generate(pipe, prompt, func_name, file_prefix, **kwargs):
    img_path = f"{OUTPUT_DIR}/{file_prefix}.png"
    time_path = f"{OUTPUT_DIR}/{file_prefix}_time.txt"
    
    if os.path.exists(img_path) and os.path.exists(time_path):
        with open(time_path, 'r') as f:
            duration = float(f.read().strip())
        print(f"   [SKIP] '{file_prefix}' already exists. Time: {duration:.2f}s")
        return duration

    print(f"   [RUN] Generating '{file_prefix}'...")
    flush_memory()
    torch.cuda.manual_seed(0) 
    
    start = time.time()
    image = pipe(prompt, **kwargs).images[0]
    duration = time.time() - start
    
    image.save(img_path)
    with open(time_path, 'w') as f:
        f.write(str(duration))
        
    print(f"   -> Done in {duration:.2f}s")
    return duration

def run_generation():
   

    baseline_times = []
    initno_times = []
    

    needed = any(not os.path.exists(f"{OUTPUT_DIR}/base_{i}.png") for i in range(len(PROMPTS_DATA)))
    
    if needed:
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, local_files_only=True).to(DEVICE)
        pipe.set_progress_bar_config(disable=True)
        pipe("warmup", num_inference_steps=1)
    else:
        print("   All Baseline images exist. Skipping model load.")
        pipe = None 

    for i, item in enumerate(PROMPTS_DATA):
        if pipe: 
            dur = load_or_generate(pipe, item["text"], "Base", f"base_{i}", num_inference_steps=50, guidance_scale=7.5)
        else: 
            with open(f"{OUTPUT_DIR}/base_{i}_time.txt", 'r') as f: dur = float(f.read())
            print(f"   [Loaded] base_{i}: {dur:.2f}s")
        baseline_times.append(dur)
    
    if pipe: del pipe
    flush_memory()

    print("\n>>> Checking/Running InitNO...")
    needed = any(not os.path.exists(f"{OUTPUT_DIR}/initno_{i}.png") for i in range(len(PROMPTS_DATA)))
    
    if needed:
        pipe = StableDiffusionInitNOPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, local_files_only=True).to(DEVICE)
        pipe.set_progress_bar_config(disable=True)
        gen = torch.Generator(DEVICE).manual_seed(0)
        pipe("warmup", token_indices=[1], num_inference_steps=1, max_iter_to_alter=1, guidance_scale=7.5, generator=gen, result_root=OUTPUT_DIR, run_sd=False)
    else:
        print("   All InitNO images exist. Skipping model load.")
        pipe = None

    for i, item in enumerate(PROMPTS_DATA):
        kwargs = {
            "token_indices": item["indices"],
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "max_iter_to_alter": SAFE_ITERATIONS,
            "result_root": OUTPUT_DIR,
            "generator": torch.Generator(DEVICE).manual_seed(100), 
            "seed": 100,
            "run_sd": False
        }
        
        if pipe:
            dur = load_or_generate(pipe, item["text"], "InitNO", f"initno_{i}", **kwargs)
        else:
            with open(f"{OUTPUT_DIR}/initno_{i}_time.txt", 'r') as f: dur = float(f.read())
            print(f"   [Loaded] initno_{i}: {dur:.2f}s")
        initno_times.append(dur)

    if pipe: del pipe
    flush_memory()

    print("\n>>> Generating Chart...")
    labels = [item['short'] for item in PROMPTS_DATA]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, baseline_times, width, label='Baseline', color='#A0A0A0')
    rects2 = ax.bar(x + width/2, initno_times, width, label='InitNO', color='#FFA500')

    ax.set_ylabel('Time (s)')
    ax.set_title('Inference Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}s', (rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center')

    autolabel(rects1)
    autolabel(rects2)
    
    plt.savefig(f"{OUTPUT_DIR}/time_chart_smart.png")
    
if __name__ == "__main__":
    run_generation()
