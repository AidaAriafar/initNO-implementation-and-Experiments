# initNO-implementation-and-Experiments


### PROJECT OVERVIEW

implementation, testing, and benchmarking of the **Initial Noise Optimization (InitNO)** method (arXiv:2404.04650).



---

### CLIP SCORE EVALUATION

The plot below compares average **CLIP Image–Text Similarity** between Baseline SD and InitNO.

Higher means better.

<p align="center">
  <img src="output/clip_score_chart.png" width="500">
</p>

**InitNO shows consistently higher CLIP scores across all tested prompts.**

---

### TIME OVERHEAD

InitNO introduces an expected computational cost:

- Approximately **10× slower** than standard Stable Diffusion inference.
- The additional time comes from the gradient-based optimization on the noise.

as described in the original paper.

---

###  QUALITATIVE RESULTS

<table style="width:100%; table-layout: fixed; text-align: center;"> <thead> <tr> <th width="30%">Prompt</th> <th width="35%">Standard SD (Baseline)</th> <th width="35%">InitNO (Optimized)</th> </tr> </thead> <tbody> <tr> <td>A cat and a rabbit</td> <td><img src="output/base_0.png" width="250"></td> <td><img src="output/prompt_0_seed2024.png" width="250"></td> </tr> <tr> <td>A frog and a purple balloon</td> <td><img src="output/base_2.png" width="250"></td> <td><img src="output/prompt_1_seed2024.png" width="250"></td> </tr> <tr> <td>A red suitcase and a yellow clock</td> <td><img src="output/base_1.png" width="250"></td> <td><img src="output/prompt_2_seed0.png" width="250"></td> </tr> <tr> <td>A cat and a sunflower, Van Gogh style</td> <td><img src="output/base_3.png" width="250"></td> <td><img src="output/prompt_3_seed0.png" width="250"></td> </tr> </tbody> </table>

---

##  PROJECT FILES

- **generate.py** : Runs InitNO, generates images, and logs inference time.  
- **clip.py**:Computes CLIP similarity scores and plots the comparison chart.  
- **initno/pipeline&utils&run**: Source code for the InitNO method (imported from the official implementation).  

---
