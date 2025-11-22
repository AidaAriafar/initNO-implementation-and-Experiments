import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

DEVICE = "cuda" 
OUTPUT_DIR = "final_correct_indices" 
CLIP_ID = "openai/clip-vit-base-patch32"

PROMPTS_DATA = [
    {"text": "a cat and a rabbit", "short": "Cat & Rabbit"},
    {"text": "a red suitcase and a yellow clock", "short": "Suitcase & Clock"},
    {"text": "A frog and a purple balloon", "short": "Frog & Balloon"},
    {"text": "A cat and a sunflower, Van Gogh style", "short": "Cat (Van Gogh)"}
]

def get_score(image_path, text, model, processor):
    if not os.path.exists(image_path):
        return 0.0
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(DEVICE)
        outputs = model(**inputs)
        return outputs.logits_per_image.item() / 100.0
    except:
        return 0.0

def run_analysis():

    print("Loading CLIP")
    model = CLIPModel.from_pretrained(CLIP_ID).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_ID)

    baseline_scores = []
    initno_scores = []
    labels = []

    for i, item in enumerate(PROMPTS_DATA):
        prompt = item["text"]
        label = item["short"]
        print(f"Analyzing: {label}")

        base_path = f"{OUTPUT_DIR}/base_{i}.png"
        b_score = get_score(base_path, prompt, model, processor)
        baseline_scores.append(b_score)

        seeds = [0, 100, 2024]
        i_scores = []
        for s in seeds:
            path = f"{OUTPUT_DIR}/initno_{i}_seed{s}.png"
            val = get_score(path, prompt, model, processor)
            if val > 0: i_scores.append(val)
        
        if i_scores:
            avg_initno = np.mean(i_scores)
        else:
            avg_initno = 0
        initno_scores.append(avg_initno)
        
        labels.append(label)
        print(f"   Base: {b_score:.4f} | InitNO (Avg): {avg_initno:.4f}")

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline', color='#A0A0A0')
    rects2 = ax.bar(x + width/2, initno_scores, width, label='InitNO (Ours)', color='#FFA500')

    ax.set_ylabel('CLIP Score')
    ax.set_title('Quantitative Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    all_scores = baseline_scores + initno_scores
    if all_scores:
        ax.set_ylim(min(all_scores) - 0.02, max(all_scores) + 0.02)
    
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.3f}', (rect.get_x() + rect.get_width()/2, h), 
                        xytext=(0, 3), textcoords="offset points", ha='center')

    autolabel(rects1)
    autolabel(rects2)

    plt.savefig(f"{OUTPUT_DIR}/final_chart.png", dpi=300)

if __name__ == "__main__":
    run_analysis()
