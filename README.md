# DreamDiffusion: Generating High-Quality Images from Brain EEG Signals
<p align="center">
<img src=assets/eeg_teaser.png />
</p>

## DreamDiffusion
**DreamDiffusion** is a framework for generating high-quality images from brain EEG signals.
This document introduces the precesedures required for replicating the results in *DreamDiffusion: Generating High-Quality Images from Brain EEG Signals*

## Abstract
This paper introduces DreamDiffusion, a novel method for generating high-quality images directly from brain electroencephalogram (EEG) signals, without the need to translate thoughts into text. DreamDiffusion leverages pre-trained text-to-image models and employs temporal masked signal modeling to pre-train the EEG encoder for effective and robust EEG representations. Additionally, the method further leverages the CLIP image encoder to provide extra supervision to better align EEG, text, and image embeddings with limited EEG-image pairs. Overall, the proposed method overcomes the challenges of using EEG signals for image generation, such as noise, limited information, and individual differences, and achieves promising results. Quantitative and qualitative results demonstrate the effectiveness of the proposed method as a significant step towards portable and low-cost "thoughts-to-image", with potential applications in neuroscience and computer vision. 

## Overview
![pipeline](assets/eeg_pipeline.png)


The **datasets** folder is not included in this repository. Neither are the pretrained models' checkpoints. 
Please download them from [eeg](https://github.com/perceivelab/eeg_visual_classification) and put them in the root directory of this repository as shown below. We also provide a copy of the Imagenet subset [imagenet](https://drive.google.com/file/d/1y7I9bG1zKYqBM94odcox_eQjnP9HGo9-/view?usp=drive_link).

For Stable Diffusion, we just use standard SD1.5. You can download it from the [official page of Stability](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5). You want the file ["v1-5-pruned.ckpt"](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).

File path | Description
```

/pretrains
┣ 📂 models
┃   ┗ 📜 config.yaml
┃   ┗ 📜 v1-5-pruned.ckpt

┣ 📂 generation  
┃   ┗ 📜 checkpoint.pth

┣ 📂 eeg_pretain
┃   ┗ 📜 checkpoint-eeg-500.pth  (pre-trained EEG encoder)

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

## Download checkpoint after pre-training on EEG data

No checkpoint provided

To make the code work (poorly), please find a checkpoint after pretraining the EEG Autoencoder in [this repository](https://github.com/alinvdu/reproduce-dream-diffusion/). They propose to use only the EEGs from the (EEG, Image) pairs but it makes poor results as this is few data.

And put it in ```pretrains/eeg-pretrain/checkpoint-eeg-500.pth``` 

## Pre-training on EEG data

remember to move generated checkpoints from the ```results``` directory to the
appropriate location in the ```pretrains``` directory

You can download the dataset for pretraining from here [MOABB](https://github.com/NeuroTechX/moabb).

The datasets used to get the papers's results aren't specified, but you can put
any EEG dataset, put them in ```datasets/mne_data/``` in a `.npy` format. 

To perform the pre-training from scratch with defaults parameters, run 
```sh
python3 code/stageA1_eeg_pretrain.py
``` 

Hyper-parameters can be changed with command line arguments,
```sh
python3 code/stageA1_eeg_pretrain.py --mask_ratio 0.75 --num_epoch 800 --batch_size 2
```

Or the parameters can also be changed in ```code/config.py```

Multiple-GPU (DDP) training is supported, run with 
```sh
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS code/stageA1_eeg_pretrain.py
```

## Finetune the Stable Diffusion with Pre-trained EEG Encoder

remember to move generated checkpoints from the ```results``` directory to the
appropriate location in the ```pretrains``` directory

In this stage, the cross-attention heads and pre-trained EEG encoder will be jointly optimized with EEG-image pairs. 

```sh
python3 code/eeg_ldm.py --dataset EEG  --num_epoch 300 --batch_size 4 --pretrain_mbm_path ./pretrains/eeg_pretrain/checkpoint-eeg-500.pth
```

## Generating Images with Trained Checkpoints
Run this stage with our provided checkpoints: Here we provide a checkpoint [ckpt](https://drive.google.com/file/d/1Ygplxe1TB68-aYu082bjc89nD8Ngklnc/view?usp=drive_link), which you may want to try.

```sh
python3 code/gen_eval_eeg.py --dataset EEG --model_path  pretrains/generation/checkpoint.pth
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
