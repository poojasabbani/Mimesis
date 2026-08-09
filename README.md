# Mimesis — Neural Style Transfer

> A deep learning system that transforms the visual style of an artwork onto a content image using **Adaptive Instance Normalization (AdaIN)** and a pretrained **VGG-19 encoder**.

## Overview

**Mimesis** is a Neural Style Transfer system designed to separate the **content** of one image from the **style** of another and combine them into a new stylized image.

Given:

* 🖼️ **Content Image** — the image whose structure and objects should be preserved
* 🎨 **Style Image** — the image whose visual characteristics should be transferred

Mimesis generates:

* ✨ **Stylized Image** — preserving the content structure while adopting the visual style of the reference image

The project focuses not only on implementing Neural Style Transfer, but also on understanding and building the complete deep learning pipeline from **data preprocessing → feature extraction → style transfer → image reconstruction → training and checkpointing**.

---

## Demo

### Content

<img width="512" height="512" alt="brad_pitt" src="https://github.com/user-attachments/assets/c37201bd-276e-4dff-89e9-131cec2478c3" />


### Style

<img width="300" height="300" alt="sketch" src="https://github.com/user-attachments/assets/33dbbf4e-7b8b-4f38-9e28-3fd8f1f2f2f8" />


### Result

<img width="512" height="512" alt="brad_pitt_sketch" src="https://github.com/user-attachments/assets/b8e4a0c7-75e4-49c3-85b2-3316babd8903" />



## How It Works

Mimesis uses **Adaptive Instance Normalization (AdaIN)** to align the statistical properties of content features with those of style features.

The overall pipeline is:

```text
                ┌─────────────────┐
                │  Content Image  │
                └────────┬────────┘
                         │
                         ▼
                  ┌─────────────┐
                  │ VGG-19      │
                  │ Encoder     │
                  └──────┬──────┘
                         │
                  Content Features
                         │
                         │
                         ▼
                 ┌──────────────┐
                 │    AdaIN     │◄──────── Style Features
                 └──────┬───────┘
                        │
                  Stylized Features
                        │
                        ▼
                 ┌──────────────┐
                 │   Decoder    │
                 └──────┬───────┘
                        │
                        ▼
                ┌─────────────────┐
                │ Stylized Image  │
                └─────────────────┘

Style Image
     │
     ▼
 VGG-19 Encoder
```

---

## Core Idea — Adaptive Instance Normalization

For a content feature map (x) and style feature map (y), AdaIN aligns the channel-wise mean and variance of the content features with those of the style features.

The transformation can be expressed as:

[
AdaIN(x,y) =
\sigma(y)
\left(
\frac{x-\mu(x)}
{\sigma(x)}
\right)
+\mu(y)
]

Where:

* (\mu(x)) = channel-wise mean of content features
* (\sigma(x)) = channel-wise standard deviation of content features
* (\mu(y)) = channel-wise mean of style features
* (\sigma(y)) = channel-wise standard deviation of style features

This allows the network to transfer important style statistics while retaining the underlying spatial structure of the content image.

---

## Architecture

Mimesis consists primarily of two components:

### 1. VGG-19 Encoder

A pretrained **VGG-19** network is used as the feature extractor.

The encoder:

* Receives RGB images
* Extracts hierarchical visual features
* Provides feature representations used for content and style
* Remains frozen during decoder training

The deeper feature representations contain increasingly abstract visual information, making them useful for separating content structure from low-level image appearance.

### 2. Decoder

The decoder is trained to reconstruct an RGB image from the transformed feature representation produced by AdaIN.

```text
Image
  │
  ▼
VGG-19 Encoder
  │
  ▼
Feature Representation
  │
  ▼
AdaIN
  │
  ▼
Stylized Feature Representation
  │
  ▼
Trainable Decoder
  │
  ▼
Generated Image
```

---

## Training Strategy

The decoder is trained using a combination of **content reconstruction** and **style reconstruction** objectives.

### Content Loss

The generated image is passed through the encoder again.

The resulting feature representation is compared with the target AdaIN feature representation.

This encourages the generated image to preserve the content structure.

### Style Loss

Style statistics are extracted from multiple feature levels and compared between the generated image and the target style image.

The objective is to encourage the generated image to reproduce the style characteristics of the reference image.

Conceptually:

[
L = L_{content} + \lambda L_{style}
]

where:

* (L_{content}) = content reconstruction loss
* (L_{style}) = style reconstruction loss
* (\lambda) = style-loss weighting factor

---

## Tech Stack

| Component           | Technology                            |
| ------------------- | ------------------------------------- |
| Language            | Python                                |
| Deep Learning       | PyTorch                               |
| Computer Vision     | Torchvision                           |
| Neural Network      | VGG-19 + Custom Decoder               |
| Style Transfer      | Adaptive Instance Normalization       |
| Image Processing    | PIL                                   |
| Data Loading        | PyTorch DataLoader                    |
| Experiment Tracking | Saved checkpoints & generated samples |

---

## Project Structure

```text
Mimesis/
│
├── train.py
│
├── inference.py
│
├── requirements.txt
│
├── README.md
│
├── utils/
│   ├── models.py
│   └── utils.py
│
├── data/
│   ├── content/
│   └── style/
│
├── experiments/
│   └── <experiment_name>/
│       ├── decoder_final.pth
│       ├── checkpoint_latest.pth
│       └── output_*.png
│
└── pretrained/
    └── vgg19.pth
```

---

## Dataset

Mimesis uses two image collections:

### Content Dataset

Contains images whose structure and semantic information are preserved in the generated output.

### Style Dataset

Contains artwork and other images providing the visual characteristics to be transferred.

The training pipeline uses separate data loaders for content and style images.

Images are:

1. Loaded using PIL
2. Converted to RGB
3. Resized/cropped
4. Converted to tensors
5. Batched using PyTorch DataLoader

---

## Training Pipeline

```text
Content Dataset ────────┐
                        │
                        ▼
                   DataLoader
                        │
                        ▼
                 Preprocessing
                        │
                        ▼
                VGG-19 Encoder
                        │
                        ▼
              Content Features
                        │
                        │
                        ├──────────────┐
                        │              │
                        ▼              ▼
                     AdaIN       Style Features
                        │              ▲
                        │              │
                        │        VGG-19 Encoder
                        │              ▲
                        │              │
                        │        Style Dataset
                        ▼
               Target Features
                        │
                        ▼
                    Decoder
                        │
                        ▼
                 Generated Image
                        │
                        ▼
                 Loss Calculation
                        │
                        ▼
                 Backpropagation
                        │
                        ▼
                Decoder Update
```

---

## Training Features

The training pipeline includes:

* Custom PyTorch dataset loading
* Batched content/style training
* Adaptive Instance Normalization
* Frozen pretrained VGG encoder
* Trainable image decoder
* Content and style losses
* Periodic generated-image outputs
* Decoder checkpointing
* Optimizer state checkpointing
* Training resume support
* Configurable image resolution
* Configurable batch size and number of epochs

---

## Example Training Command

```bash
python train.py \
    --content_dir ./data/content \
    --style_dir ./data/style \
    --batch_size 8 \
    --epoch 5 \
    --experiment mimesis
```

Modify the arguments according to your dataset and training configuration.

---

## Inference

After training, the decoder can be used to generate stylized images from previously unseen content/style combinations.

```text
Content Image + Style Image
            │
            ▼
       VGG-19 Encoder
            │
            ▼
           AdaIN
            │
            ▼
      Trained Decoder
            │
            ▼
      Stylized Output
```

---

## Engineering Challenges

Building the system involved several practical deep learning challenges beyond implementing the core algorithm.

### 1. Tensor Shape Management

AdaIN requires the content and style feature statistics to have compatible dimensions.

Incorrect broadcasting can result in errors involving dimensions such as:

```text
[Batch, Channels, Height, Width]
```

Understanding these dimensions was essential for correctly implementing feature normalization.

### 2. Variable Image Dimensions

Images in real-world datasets do not necessarily share the same dimensions.

Batching such images directly can produce errors because PyTorch expects tensors in a batch to have compatible shapes.

The preprocessing pipeline therefore standardizes image dimensions before batching.

### 3. DataLoader Stability

Large image datasets introduce challenges involving:

* Worker processes
* Tensor memory
* Batch construction
* Image dimensions
* CPU/GPU memory

The data pipeline was adjusted to ensure reliable batch generation during training.

### 4. GPU Memory Management

Neural Style Transfer can be computationally expensive, particularly when increasing:

* Image resolution
* Batch size
* Dataset size
* Number of feature computations

Training therefore requires balancing image quality, batch size, GPU memory, and training time.

---

## Checkpointing

Training checkpoints store the state required to resume experiments.

A checkpoint can contain:

```python
{
    "epoch": epoch,
    "decoder": decoder.state_dict(),
    "optimizer": optimizer.state_dict(),
}
```

This allows training to continue without restarting from the beginning.

---

## Results

Add your final results here after completing the larger training run.

Recommended comparison:

| Content | Style | Output |
| ------- | ----- | ------ |
| Image   | Image | Image  |
| Image   | Image | Image  |
| Image   | Image | Image  |

### What to Evaluate

The final model should be evaluated based on:

* Preservation of content structure
* Strength of style transfer
* Visual coherence
* Artifacts
* Generalization to unseen style images
* Performance at different resolutions

---

## What I Learned

This project helped me understand the complete lifecycle of a deep learning system rather than only the model implementation.

Key concepts explored:

* CNN feature representations
* VGG-19 architecture
* Transfer learning
* Adaptive Instance Normalization
* Feature statistics
* Content vs. style representations
* PyTorch Dataset and DataLoader
* GPU memory management
* Training loops
* Loss functions
* Checkpointing and experiment management
* Debugging tensor shape issues
* Scaling training to larger datasets

---

## Future Improvements

Potential improvements include:

* [ ] Higher-resolution generation
* [ ] Improved decoder architecture
* [ ] Better style-loss weighting
* [ ] Multi-scale style representations
* [ ] Faster inference
* [ ] Web-based inference interface
* [ ] Batch style transfer
* [ ] Quantitative evaluation metrics
* [ ] Experiment tracking and visualization
* [ ] Deployment as an inference API

---

## References

The project is based on the ideas introduced in:

**Huang, X., & Belongie, S. — "Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization."**

The project also builds upon the use of pretrained convolutional feature representations for image generation and style transfer.

---

## Author

**Pooja Sabbani**

B.Tech — Robotics & Automation Engineering

Interested in:

* Machine Learning
* Computer Vision
* Deep Learning
* AI Systems
* Applied Research

---

## License

This project is intended for educational and research purposes.
