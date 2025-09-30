# DreamDiffusion: Generating High-Quality Images from Brain EEG Signals
<p align="center">
<img src=assets/eeg_teaser.png />
</p>



**[2025.9.30]** Some paths were set incorrectly. Since they were only tested on the previous server, they were not noticed and have been corrected.
The pre-trained encoder does not need to be loaded separately during inference

## DreamDiffusion
**DreamDiffusion** is a framework for generating high-quality images from brain EEG signals.
This document introduces the precesedures required for replicating the results in *DreamDiffusion: Generating High-Quality Images from Brain EEG Signals*

## Abstract
This paper introduces DreamDiffusion, a novel method for generating high-quality images directly from brain electroencephalogram (EEG) signals, without the need to translate thoughts into text. DreamDiffusion leverages pre-trained text-to-image models and employs temporal masked signal modeling to pre-train the EEG encoder for effective and robust EEG representations. Additionally, the method further leverages the CLIP image encoder to provide extra supervision to better align EEG, text, and image embeddings with limited EEG-image pairs. Overall, the proposed method overcomes the challenges of using EEG signals for image generation, such as noise, limited information, and individual differences, and achieves promising results. Quantitative and qualitative results demonstrate the effectiveness of the proposed method as a significant step towards portable and low-cost "thoughts-to-image", with potential applications in neuroscience and computer vision. 


## Overview
![pipeline](assets/eeg_pipeline.png)


The **datasets** folder and **pretrains** folder are not included in this repository. 
Please download them from [eeg](https://github.com/perceivelab/eeg_visual_classification) and put them in the root directory of this repository as shown below. We also provide a copy of the Imagenet subset [imagenet](https://drive.google.com/file/d/1y7I9bG1zKYqBM94odcox_eQjnP9HGo9-/view?usp=drive_link).

For Stable Diffusion, we just use standard SD1.5. You can download it from the [official page of Stability](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main). You want the file ["v1-5-pruned.ckpt"](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main).

File path | Description
```

/pretrains
┣ 📂 models
┃   ┗ 📜 config.yaml
┃   ┗ 📜 v1-5-pruned.ckpt

┣ 📂 generation  
┃   ┗ 📜 checkpoint_best.pth 

┣ 📂 eeg_pretain
┃   ┗ 📜 checkpoint.pth  (pre-trained EEG encoder)

/datasets
┣ 📂 imageNet_images (subset of Imagenet)

┗  📜 block_splits_by_image_all.pth
┗  📜 block_splits_by_image_single.pth 
┗  📜 eeg_5_95_std.pth  

/code
┣ 📂 sc_mbm
┃   ┗ 📜 mae_for_eeg.py
┃   ┗ 📜 trainer.py
┃   ┗ 📜 utils.py

┣ 📂 dc_ldm
┃   ┗ 📜 ldm_for_eeg.py
┃   ┗ 📜 utils.py
┃   ┣ 📂 models
┃   ┃   ┗ (adopted from LDM)
┃   ┣ 📂 modules
┃   ┃   ┗ (adopted from LDM)

┗  📜 stageA1_eeg_pretrain.py   (main script for EEG pre-training)
┗  📜 eeg_ldm.py    (main script for fine-tuning stable diffusion)
┗  📜 gen_eval_eeg.py               (main script for generating images)

┗  📜 dataset.py                (functions for loading datasets)
┗  📜 eval_metrics.py           (functions for evaluation metrics)
┗  📜 config.py                 (configurations for the main scripts)

```


## Environment setup

Create and activate conda environment named ```dreamdiffusion``` from the ```env.yaml```
```sh
conda env create -f env.yaml
conda activate dreamdiffusion
```


## Generating Images with Trained Checkpoints
Run this stage with our provided checkpoints: Here we provide a checkpoint [ckpt](https://drive.google.com/file/d/1Ygplxe1TB68-aYu082bjc89nD8Ngklnc/view?usp=drive_link), which you may want to try.
```sh
python3 code/gen_eval_eeg.py --dataset EEG --model_path  pretrains/generation/checkpoint.pth --splits_path "./block_splits_by_image_single.pth" --eeg_signals_path "./eeg_5_95_std.pth"
```


![results](assets/results.png)

## Acknowledgement

This code is built upon the publicly available code [Mind-vis](https://github.com/zjc062/mind-vis) and [StableDiffusion](https://github.com/CompVis/stable-diffusion). Thanks these authors for making their excellent work and codes publicly available.


## Citation ##
Please cite the following paper if you use this repository in your reseach.

```
@article{bai2023dreamdiffusion,
  title={DreamDiffusion: Generating High-Quality Images from Brain EEG Signals},
  author={Bai, Yunpeng and Wang, Xintao and Cao, Yanpei and Ge, Yixiao and Yuan, Chun and Shan, Ying},
  journal={arXiv preprint arXiv:2306.16934},
  year={2023}
}
