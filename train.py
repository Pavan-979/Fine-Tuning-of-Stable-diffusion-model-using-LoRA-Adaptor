import argparse
import json
import os
import subprocess
import torch

from diffusers import StableDiffusionPipeline


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def create_metadata(dataset_folder, prompt):

    metadata_path = os.path.join(
        dataset_folder,
        "metadata.jsonl"
    )

    with open(metadata_path, "w") as f:

        for image_name in os.listdir(dataset_folder):

            if image_name.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):

                json.dump(
                    {
                        "file_name": image_name,
                        "text": prompt
                    },
                    f
                )

                f.write("\n")

    print("metadata.jsonl created")


def train_lora(cfg):

    command = [
        "accelerate",
        "launch",
        "train_text_to_image_lora.py",

        "--pretrained_model_name_or_path",
        cfg["model_name"],

        "--train_data_dir",
        cfg["dataset_folder"],

        "--resolution",
        str(cfg["resolution"]),

        "--center_crop",

        "--train_batch_size",
        str(cfg["train_batch_size"]),

        "--gradient_accumulation_steps",
        str(cfg["gradient_accumulation_steps"]),

        "--learning_rate",
        str(cfg["learning_rate"]),
        
        "--rank",
        str(cfg["rank"]),

        "--max_train_steps",
        str(cfg["max_train_steps"]),

        "--checkpointing_steps",
        str(cfg["checkpointing_steps"]),

        "--output_dir",
        cfg["output_dir"],

        "--validation_prompt",
        cfg["validation_prompt"],

        "--mixed_precision",
        cfg["mixed_precision"]
    ]

    subprocess.run(command, check=True)


def generate_sample(cfg):

    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )

    pipe = pipe.to("cuda")

    pipe.load_lora_weights(
        cfg["output_dir"]
    )

    image = pipe(
        cfg["test_prompt"],
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    os.makedirs(
        cfg["output_dir"],
        exist_ok=True
    )

    sample_path = os.path.join(
        cfg["output_dir"],
        "sample.png"
    )

    image.save(sample_path)

    print("\nTraining Complete")
    print("LoRA saved at:", cfg["output_dir"])
    print("Sample saved at:", sample_path)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.json"
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    print("=" * 50)
    print("CUDA Available:",
          torch.cuda.is_available())

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print("=" * 50)

    create_metadata(
        cfg["dataset_folder"],
        cfg["validation_prompt"]
    )

    train_lora(cfg)

    generate_sample(cfg)


if __name__ == "__main__":
    main()





#python train.py --config config.json