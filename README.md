# Stable Diffusion LoRA Fine-Tuning Framework

## Overview

This repository provides a complete pipeline for fine-tuning Stable Diffusion models using Low-Rank Adaptation (LoRA). The framework is built on top of Hugging Face Diffusers and enables efficient training on custom image datasets while significantly reducing the number of trainable parameters compared to full model fine-tuning.

The project automates:

* Dataset metadata generation
* LoRA adapter training
* Checkpoint management
* Validation image generation
* Final sample generation

This implementation is designed for researchers, students, and practitioners who want to train custom concepts, objects, styles, or characters using Stable Diffusion.

---

# Features

* Stable Diffusion LoRA Fine-Tuning
* Configuration-Based Training
* Automatic Metadata Generation
* Checkpoint Saving and Recovery
* Validation Image Monitoring
* Mixed Precision Training (FP16/BF16)
* Custom Dataset Support
* GPU Accelerated Training
* Sample Image Generation after Training

---

# Project Structure

```text
.
├── train.py
├── train_text_to_image_lora.py
├── config.json
├── requirements.txt
│
├── dataset/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   └── ...
│
└── outputs/
    └── rabbit_lora/
```

---

# Installation

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Requirements

```text
Python >= 3.10
PyTorch
Diffusers 0.38.0
Transformers 4.55.4
Accelerate
Datasets
PEFT
Pillow
```

Install manually if required:

```bash
pip install \
diffusers==0.38.0 \
transformers==4.55.4 \
accelerate \
datasets \
peft==0.17.1 \
torch \
Pillow
```

---

# Dataset Preparation

Create a dataset folder containing your training images.

```text
dataset/
├── image1.jpg
├── image2.jpg
├── image3.jpg
└── ...
```

Supported image formats:

```text
.jpg
.jpeg
.png
```

The training script automatically generates the required metadata file:

```text
metadata.jsonl
```

Example:

```json
{"file_name":"image1.jpg","text":"rabbitx rabbit"}
{"file_name":"image2.jpg","text":"rabbitx rabbit"}
```

No manual metadata creation is required.

---

# Configuration

All training settings are stored inside `config.json`.

Example:

```json
{
  "dataset_folder": "./dataset",
  "token": "rabbitx",
  "class_name": "rabbit",
  "output_dir": "./outputs/rabbit_lora",
  "validation_prompt": "rabbitx rabbit",
  "test_prompt": "rabbitx rabbit on a beach",
  "model_name": "runwayml/stable-diffusion-v1-5",
  "resolution": 512,
  "train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "learning_rate": 0.0001,
  "max_train_steps": 1000,
  "checkpointing_steps": 500,
  "mixed_precision": "fp16"
}
```

---

# Running Training

Run training using:

```bash
python train.py --config config.json
```

Training workflow:

1. Load configuration
2. Create metadata.jsonl
3. Load Stable Diffusion model
4. Attach LoRA adapters
5. Start training
6. Save checkpoints
7. Generate validation images
8. Save final LoRA weights
9. Generate sample image

---

# Commonly Modified Parameters

These are the parameters most users will modify when training a new LoRA model.

## Dataset Path

```json
"dataset_folder": "./dataset"
```

Specifies the folder containing training images.

---

## Token

```json
"token": "rabbitx"
```

Unique identifier learned by the model.

Example prompt:

```text
rabbitx rabbit
```

---

## Output Directory

```json
"output_dir": "./outputs/rabbit_lora"
```

Directory where LoRA weights, checkpoints, logs, and sample images are saved.

---

## Validation Prompt

```json
"validation_prompt": "rabbitx rabbit"
```

Used during training to generate validation images.

---

## Test Prompt

```json
"test_prompt": "rabbitx rabbit on a beach"
```

Used after training to generate the final sample image.

---

## Base Model

```json
"model_name": "runwayml/stable-diffusion-v1-5"
```

Specifies the Stable Diffusion model used for training.

Popular choices:

```text
runwayml/stable-diffusion-v1-5
stabilityai/stable-diffusion-2-1
```

---

## Resolution

```json
"resolution": 512
```

Controls training image size.

| Resolution | Description                            |
| ---------- | -------------------------------------- |
| 512        | Fastest training                       |
| 768        | Better detail                          |
| 1024       | Highest quality but requires more VRAM |

---

## Batch Size

```json
"train_batch_size": 1
```

Number of images processed simultaneously.

Higher values require more GPU memory.

---

## Gradient Accumulation

```json
"gradient_accumulation_steps": 4
```

Allows simulation of larger batch sizes.

Effective Batch Size:

```text
train_batch_size × gradient_accumulation_steps × number_of_GPUs
```

Example:

```text
1 × 4 × 1 = 4
```

---

## Learning Rate

```json
"learning_rate": 0.0001
```

Controls how quickly the model learns.

Typical values:

```text
1e-4
5e-5
1e-5
```

---

## Training Steps

```json
"max_train_steps": 1000
```

Total number of optimization steps.

Higher values generally improve learning but may increase overfitting.

Typical values:

```text
500
1000
1500
2000
```

---

## Checkpoint Frequency

```json
"checkpointing_steps": 500
```

Save training checkpoints every N steps.

Example:

```text
checkpoint-500
checkpoint-1000
```

---

## Mixed Precision

```json
"mixed_precision": "fp16"
```

Available options:

```text
fp16
bf16
no
```

Recommended:

```text
fp16
```

for most GPUs.

---

# Changing LoRA Rank

The LoRA rank controls the capacity of the LoRA adapter and determines how much information the adapter can learn.

Current default value:

```python
parser.add_argument(
    "--rank",
    type=int,
    default=4,
    help=("The dimension of the LoRA update matrices."),
)
```

To change the rank, open:

```text
train_text_to_image_lora.py
```

Locate:

```python
parser.add_argument(
    "--rank",
    type=int,
    default=4
)
```

and modify:

```python
default=4
```

to:

```python
default=8
```

or

```python
default=16
```

or

```python
default=32
```

Typical values:

| Rank | Description                                     |
| ---- | ----------------------------------------------- |
| 4    | Lowest memory usage                             |
| 8    | Balanced                                        |
| 16   | Better detail learning                          |
| 32   | Higher capacity and quality                     |
| 64   | Maximum capacity but higher memory requirements |

Higher ranks increase:

* Learning capacity
* Fine detail representation
* LoRA file size
* Training memory requirements

---

# Output Files

After training:

```text
outputs/
└── rabbit_lora/
    ├── pytorch_lora_weights.safetensors
    ├── sample.png
    ├── logs/
    ├── checkpoint-500/
    ├── checkpoint-1000/
    └── ...
```

### Output Description

| File                             | Description                           |
| -------------------------------- | ------------------------------------- |
| pytorch_lora_weights.safetensors | Final trained LoRA weights            |
| sample.png                       | Sample image generated after training |
| checkpoint-*                     | Intermediate checkpoints              |
| logs                             | Training logs                         |

---

# Resume Training

Resume from a checkpoint:

```bash
accelerate launch train_text_to_image_lora.py \
--resume_from_checkpoint checkpoint-1000
```

Resume from latest checkpoint:

```bash
accelerate launch train_text_to_image_lora.py \
--resume_from_checkpoint latest
```

---

# GPU Requirements

Approximate recommendations:

| Resolution | Minimum VRAM |
| ---------- | ------------ |
| 512        | 8 GB         |
| 768        | 12 GB        |
| 1024       | 16+ GB       |

Actual memory requirements depend on batch size, gradient accumulation, LoRA rank, optimizer settings, and mixed precision configuration.

---

# Example Inference

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

pipe.load_lora_weights("./outputs/rabbit_lora")

image = pipe(
    "rabbitx rabbit on a beach",
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

image.save("result.png")
```

---

# Applications

This framework can be used for:

* Custom Character Generation
* Personalized Product Generation
* Artistic Style Learning
* Object-Specific Fine-Tuning
* Educational Projects
* Research in Parameter-Efficient Fine-Tuning
* Diffusion Model Experimentation

---

# Acknowledgements

This project builds upon the following open-source libraries:

* Hugging Face Diffusers
* Hugging Face Accelerate
* Hugging Face Transformers
* PEFT (Parameter-Efficient Fine-Tuning)
* Stable Diffusion

---

# License

This repository follows the licenses of the underlying libraries and models used during training. Users are responsible for complying with the licensing terms of Stable Diffusion models, datasets, and third-party dependencies before deployment or redistribution.
