---
layout: post
comments: true
title: Fine-tuning Stable Diffusion with Dreambooth
author: Rory Hemmings, Brody Jones, Tomasz Jezak, Hank Lin
date: 2024-03-20
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].

## Introduction

Our project is about DreamBooth.

## Traning Script

Now that you have an idea of how the DreamBooth traning method works, let's dive into an example of how it is utilized in practice.

The de-facto standard for anything related to diffusions is the python library [diffusers](https://huggingface.co/docs/diffusers/en/index) created by Hugging Face. It provides primatives and pretrained diffusion models for use accross several domains like image and audio generation.

In this case we used their `train_dreambooth` script in combination with their pretrained Stable Diffusion model to run our experiments.

### Script overview

To finetune raw Stable Diffusion, we ran the script with the following command which reveals our chosen hyper parameters:
```
accelerate launch ./diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir="./train_data" \
  --output_dir="finetuned_model" \
  --instance_prompt="A photo of rory" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --train_text_encode
```
As you can see, we load the pretrained stable diffusion, provide our training data and a label for each image. Additionally, we train for 400 steps with a constant learning rate of 5e-6. Another thing to note is that we instruct the script to retrain the text encoder with the flag `--train_text_encode`. We found this to be incredibly important for image quality, particularly in generating images of faces. This makes sense as it helps the model distinguish between text encodings for the provided training faces and its preconcieved notion of a face.

Using this script, 6 training images and an A100 GPU, we were able to train the model in around 4 minutes.

Taking a closer look at the base script, we can shead light on the `diffusers` implementation of DreamBooth. Considering that the actual training loops is around 400 lines of code, we have annotated the code and omitted many of the details for the sake of brevity. While not identical, the following code closely resembles the training loop in the script.
```py
for epoch in range(first_epoch, args.num_train_epochs):
    unet.train()
    text_encoder.train()
    for step, batch in enumerate(train_dataloader):
        # Encode images into the latent space using provided vae
        pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
        model_input = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()

        # Sample noise to add to the image
        noise = torch.randn_like(model_input) + 0.1 * torch.randn(
            model_input.shape[0], model_input.shape[1], 1, 1, device=model_input.device
        )

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
        ).long()

        # Add noise to each image (forward diffusion)
        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

        # Create text embedding for conditioning using provided training label
        encoder_hidden_states = encode_prompt(
            text_encoder,
            batch["input_ids"],
            batch["attention_mask"],
            text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
        )

        # Predict noise residual
        model_pred = unet(
            noisy_model_input, timesteps, encoder_hidden_states, class_labels=class_labels, return_dict=False
        )[0]


```


For Stable Diffusion XL we used this command:
```
!accelerate launch diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --instance_data_dir="./train_data" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --output_dir="finetuned_model_xl" \
  --mixed_precision="fp16" \
  --instance_prompt="A photo of rory's face" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --seed="0" \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --use_8bit_ada
```
We ended up using pretty much the same hyperparameters except with a more aggressive learning rate of `1e-4` and `500` traning steps. Something to note here is that we ended up using **lora** which allows us to save ram. This is particularly important as it significantly reduced training times and RAM usage. This is imoprtant for SDXL because it is such a large model.

## Results

Overall we attempted to generate custom images using DreamBooth to fine-tune both Stable Diffusion and Stable Diffusion XL. These experiments included generating images of our faces and scenes from movies given the following traning datasets:

* A face from same perspective in same environment (6 images)
* Full body images from different angles and environments (6 images)
* Frames from given movie scene (5 images)

The purpose of having these different datasets was evaluate whether the model learned better given images from different contexts, as well as it's capability to replicate the surrounding environment as opposed to specific objects.

#### Training images

![Limited Context Training Images]({{ '/assets/images/team39/face1_train/1.jpg' | relative_url }})

#### Generated Images

![Rory in the desert]({{ '/assets/images/team39/face1_generated/1.png' | relative_url }})

SD trained on prompt (A photo of rory)
SDXL trained on prompt (A photo of rory's face)

### Stable Diffusion


### Stable Diffusion XL


### Hyperparams
### Different Training Datasets

## Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.



---
