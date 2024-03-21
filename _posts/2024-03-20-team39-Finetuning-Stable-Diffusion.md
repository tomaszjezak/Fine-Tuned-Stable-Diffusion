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

### Table
Here is an example for creating tables, including alignment syntax.

|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          |



### Formula
Please use latex to generate formulas, such as:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

or you can write in-text formula $$y = wx + b$$.

## Introduction

Our project is about DreamBooth.

## Traning Script

Now that you have an idea of how the DreamBooth traning method works, let's dive into an example of how it is utilized in practice.

The de-facto standard for anything related to diffusions is the python library [diffusers](https://huggingface.co/docs/diffusers/en/index) created by Hugging Face. It provides primatives and pretrained diffusion models for use accross several domains like image and audio generation.

In this case we used their `train_dreambooth` script in combination with their pretrained Stable Diffusion model to run our experiments.

### Script overview

This is how the script works.

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

### Stable Diffusion


### Stable Diffusion XL


### Hyperparams
### Different Training Datasets

## Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.



---
