
# 💣 Gradient-Mine Units: Scorched-Earth Strategy for Model Protection against Unauthorized Fine-Tuning

Welcome to the official repository for **Gradient-Mine Units (GMU)**. 

As open-source models become increasingly valuable, protecting their intellectual property (IP) from unauthorized downstream adaptation is a critical challenge. GMU introduces a novel **"scorched-earth" strategy** in parameter space. 

Instead of relying on passive post-hoc watermarking, GMUs act as hidden traps planted directly within the pretrained weights. They remain completely silent during normal inference, but actively react to gradient-based updates, driving the unauthorized fine-tuning process into a destructive collapse.

### ✨ Key Features
* **Zero-Shot Inference Preservation:** Maintains the original model's performance exactly as intended through a symmetry-locked initialization.
* **Post-hoc & Data-free Injection:** Placed directly into pretrained weights without requiring access to original training data or expensive retraining.
* **Proactive Deterrence:** Degrades model utility *only* when fine-tuning is attempted, making unauthorized adaptation practically unrewarding.
---
