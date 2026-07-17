# CrowdGuard: Real-Time Crowd Crush Risk Detection with CSRNet

A crowd counting and safety alert system built with PyTorch, implementing the **CSRNet Single-Column CNN** architecture to estimate crowd density from images and trigger notifications when density crosses safety thresholds grounded in established pedestrian safety research.

## Overview

This project estimates the number of people in an image by generating a **density map** using a deep convolutional neural network (CSRNet), then sums that map to produce a total headcount. If the estimated count (or local density) exceeds a configurable safety threshold, the system flags a potential crowd crush risk.

## Motivation

Crowd crushes are a recurring and devastating public safety hazard at large gatherings, concerts, and festivals; tragedies like the 2022 Itaewon Halloween crowd crush in Seoul, which claimed over 150 lives, underscore how quickly a dense crowd can turn fatal and how critical early warning can be. This project was motivated by those events, and explores whether a lightweight, single-column CNN model can provide fast, reasonably accurate crowd density estimates suitable for early-risk alerts.

## Features

- CSRNet (Single-Column CNN) architecture implemented from scratch in PyTorch
- Custom dataset loader with adaptive density map generation/transformation
- Trained and validated on the **ShanghaiTech Part A** benchmark dataset
- Threshold-based alert logic for flagging high-risk crowd density
- Trained on Google Colab (GPU-accelerated)

## Evaluation Methodology

To translate model output into a meaningful safety signal, this project grounds its threshold logic in two established pedestrian dynamics models:

- **Fruin's Level of Service (LOS):** A classification system (LOS A–F) that rates crowd safety and comfort based on the space available per pedestrian (m² per person). Lower space per person means more restricted movement and higher risk. The model's estimated crowd count is converted to a density value and mapped against Fruin's LOS thresholds to classify the current safety level.
- **Helbing's Social Force Model:** A model of pedestrian crowd dynamics that identifies a **critical density of 6–7 people/m²** as the threshold beyond which crowd crush risk sharply increases due to compressive forces between individuals. This value is used as the upper-bound trigger for high-risk alerts.

## Dataset

**[ShanghaiTech Part A](https://www.kaggle.com/datasets/tthien/shanghaitech)** was chosen for its variation in crowd scale, density, and scene composition, making it well-suited for benchmarking crowd counting models.

## Architecture

CSRNet uses a VGG-16 front-end for feature extraction paired with a dilated convolutional back-end, allowing it to preserve spatial resolution while expanding the receptive field. It is well suited for dense crowd scenes without needing multi-column complexity.

Reference: [CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes](https://arxiv.org/abs/1802.10062)

## Setup

```bash
git clone https://github.com/annieyp/cv-crowdcrush-detection.git
cd cv-crowdcrush-detection
pip install -r requirements.txt
```

## Usage

**Training:**
```bash
python train.py --data_path ./data/ShanghaiTech/part_A --epochs 100
```

**Inference / crowd counting on an image:**
```bash
python test.py --image_path ./samples/crowd.jpg
```

**Crowd-crush alert threshold:**
Density thresholds in `alert.py` are based on Fruin's LOS scale and Helbing's critical density findings:
```python
CRITICAL_DENSITY = 6.0  # people/m^2, per Helbing's social force model
# Estimated count is divided by scene area (m^2) to get density,
# then classified against Fruin's LOS bands (A-F) for a safety rating
```

## Results

| Dataset | MAE | MSE |
|---|---|---|
| ShanghaiTech Part A | _TBD_ | _TBD_ |

## Use Cases

This system is designed with deployment by **police departments, event safety teams, and public safety organizations** in mind, particularly for high-density gatherings where crowd crush risk is elevated:

- **Festivals and concerts** — monitoring entry points, stages, and choke points where crowds compress unexpectedly
- **Parades/holiday street gatherings** — the kind of unpermitted, high-density crowd events (like Itaewon) where no official crowd management infrastructure exists
- **Sporting events and stadium entrances/exits** — flagging dangerous bottlenecks before or after games
- **Religious pilgrimages and large public gatherings** — supporting safety teams monitoring crowd flow over wide areas
- **Transit hubs during peak events** — subway/train station platforms during major public events

The goal is to give safety personnel a real-time, camera-fed density signal — grounded in Fruin's LOS and Helbing's critical density research — so they can intervene (e.g., redirect crowd flow, close entry points, deploy additional personnel) *before* density reaches crush-risk levels, rather than reacting after an incident begins.

> **Note:** This is a personal/academic project and has not been validated for real-world safety-critical deployment. Any use in an actual public safety context would require rigorous testing, calibration to specific camera/venue setups, and review by domain experts before relying on it operationally.

## Future Work

- Real-time video stream support via OpenCV
- Deployment as a live monitoring demo
- Expand training to additional benchmark datasets (e.g., ShanghaiTech Part B, UCF-CC-50)

## Tech Stack

- Python
- PyTorch
- Google Colab (training environment)
- OpenCV (planned/exploratory, for image/video processing)

## Acknowledgements

- [CSRNet paper](https://arxiv.org/abs/1802.10062) by Li, Zhang, and Chen
- ShanghaiTech crowd counting dataset
