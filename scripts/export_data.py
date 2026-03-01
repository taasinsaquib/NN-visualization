#!/usr/bin/env python3
"""
Export AlexNet weights and activations for the first two conv blocks.
Saves binary Float32 files and metadata for visualization.
"""

import io
import json
import os
import urllib.request

import numpy as np
import torch
from PIL import Image
from torchvision.models import alexnet, AlexNet_Weights
from torchvision import transforms


# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Input size per spec (227x227); 224 produces 55x55 spatial dims
INPUT_SIZE = 227

CAT_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"


def ensure_dirs(base_path: str):
    """Create all necessary output directories."""
    dirs = [
        os.path.join(base_path, "public", "data", "weights"),
        os.path.join(base_path, "public", "data", "activations"),
        os.path.join(base_path, "public", "data", "images"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  Created/verified: {d}")


def download_cat_image(save_path: str) -> Image.Image:
    """Download cat image and save raw copy for display."""
    print("Downloading cat image...")
    req = urllib.request.Request(
        CAT_IMAGE_URL,
        headers={"User-Agent": "AlexNet-Viz/1.0 (Python; educational project)"}
    )
    with urllib.request.urlopen(req) as resp:
        img = Image.open(io.BytesIO(resp.read())).convert("RGB")
    img.save(save_path)
    print(f"  Saved raw image to {save_path}")
    return img


def preprocess_image(img: Image.Image) -> torch.Tensor:
    """Preprocess image for AlexNet: resize, to tensor, ImageNet normalize."""
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(img).unsqueeze(0)  # add batch dim


def save_tensor_bin(tensor: torch.Tensor, path: str):
    """Save tensor as binary Float32, little-endian, no batch dim."""
    arr = tensor.squeeze(0).detach().numpy().astype(np.float32)
    with open(path, "wb") as f:
        f.write(arr.tobytes())
    print(f"  Saved {path} shape {tuple(arr.shape)}")


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base, "public", "data")

    print("=== AlexNet Data Export ===\n")

    # 1. Create directories
    print("1. Creating directories...")
    ensure_dirs(base)

    # 2. Download and save raw cat image
    cat_path = os.path.join(data_dir, "images", "cat.jpg")
    img = download_cat_image(cat_path)

    # 3. Load AlexNet
    print("\n2. Loading pretrained AlexNet...")
    model = alexnet(weights=AlexNet_Weights.DEFAULT)
    model.eval()

    # 4. Preprocess image
    print("\n3. Preprocessing image...")
    x = preprocess_image(img)

    # 5. Register hooks to capture activations
    activations = {}

    def make_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach()
        return hook

    hooks = [
        model.features[0].register_forward_hook(make_hook("conv1")),
        model.features[1].register_forward_hook(make_hook("relu1")),
        model.features[2].register_forward_hook(make_hook("pool1")),
        model.features[3].register_forward_hook(make_hook("conv2")),
        model.features[4].register_forward_hook(make_hook("relu2")),
        model.features[5].register_forward_hook(make_hook("pool2")),
    ]

    # 6. Forward pass through first two conv blocks only
    print("\n4. Running forward pass (features 0-5)...")
    with torch.no_grad():
        _ = model.features[:6](x)

    # Remove hooks
    for h in hooks:
        h.remove()

    # 7. Extract weights (torchvision uses simplified AlexNet: 64, 192 channels)
    print("\n5. Extracting weights...")
    conv1_weight = model.features[0].weight
    conv1_bias = model.features[0].bias
    conv2_weight = model.features[3].weight
    conv2_bias = model.features[3].bias

    weights_dir = os.path.join(data_dir, "weights")
    save_tensor_bin(conv1_weight, os.path.join(weights_dir, "conv1_weight.bin"))
    save_tensor_bin(conv1_bias, os.path.join(weights_dir, "conv1_bias.bin"))
    save_tensor_bin(conv2_weight, os.path.join(weights_dir, "conv2_weight.bin"))
    save_tensor_bin(conv2_bias, os.path.join(weights_dir, "conv2_bias.bin"))

    # 8. Save activations (including preprocessed input)
    print("\n6. Saving activations...")
    activations_dir = os.path.join(data_dir, "activations")
    save_tensor_bin(x, os.path.join(activations_dir, "input.bin"))
    save_tensor_bin(activations["conv1"], os.path.join(activations_dir, "conv1.bin"))
    save_tensor_bin(activations["relu1"], os.path.join(activations_dir, "relu1.bin"))
    save_tensor_bin(activations["pool1"], os.path.join(activations_dir, "pool1.bin"))
    save_tensor_bin(activations["conv2"], os.path.join(activations_dir, "conv2.bin"))
    save_tensor_bin(activations["relu2"], os.path.join(activations_dir, "relu2.bin"))
    save_tensor_bin(activations["pool2"], os.path.join(activations_dir, "pool2.bin"))

    # 9. Write metadata (dynamically from actual layer shapes)
    print("\n7. Writing metadata.json...")
    c1_shape = list(activations["conv1"].shape[1:])
    c2_shape = list(activations["conv2"].shape[1:])
    conv1_out_ch, conv1_in_ch = conv1_weight.shape[0], conv1_weight.shape[1]
    conv2_out_ch, conv2_in_ch = conv2_weight.shape[0], conv2_weight.shape[1]

    metadata = {
        "image": "cat",
        "layers": [
            {
                "id": "input",
                "type": "input",
                "shape": [3, INPUT_SIZE, INPUT_SIZE],
                "activations_file": "activations/input.bin"
            },
            {
                "id": "conv1",
                "type": "conv",
                "shape": c1_shape,
                "groups": 1,
                "kernel_size": 11,
                "stride": 4,
                "padding": 2,
                "in_channels": 3,
                "weights_file": "weights/conv1_weight.bin",
                "weights_shape": list(conv1_weight.shape),
                "bias_file": "weights/conv1_bias.bin",
                "activations_file": "activations/conv1.bin",
                "input_layer": "input"
            },
            {
                "id": "relu1",
                "type": "relu",
                "shape": list(activations["relu1"].shape[1:]),
                "activations_file": "activations/relu1.bin",
                "input_layer": "conv1"
            },
            {
                "id": "pool1",
                "type": "maxpool",
                "shape": list(activations["pool1"].shape[1:]),
                "kernel_size": 3,
                "stride": 2,
                "padding": 0,
                "activations_file": "activations/pool1.bin",
                "input_layer": "relu1"
            },
            {
                "id": "conv2",
                "type": "conv",
                "shape": c2_shape,
                "groups": getattr(model.features[3], "groups", 1),
                "kernel_size": 5,
                "stride": 1,
                "padding": 2,
                "in_channels": conv1_out_ch,
                "weights_file": "weights/conv2_weight.bin",
                "weights_shape": list(conv2_weight.shape),
                "bias_file": "weights/conv2_bias.bin",
                "activations_file": "activations/conv2.bin",
                "input_layer": "pool1"
            },
            {
                "id": "relu2",
                "type": "relu",
                "shape": list(activations["relu2"].shape[1:]),
                "activations_file": "activations/relu2.bin",
                "input_layer": "conv2"
            },
            {
                "id": "pool2",
                "type": "maxpool",
                "shape": list(activations["pool2"].shape[1:]),
                "kernel_size": 3,
                "stride": 2,
                "padding": 0,
                "activations_file": "activations/pool2.bin",
                "input_layer": "relu2"
            }
        ]
    }

    with open(os.path.join(data_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved metadata.json")
    print("\n=== Export complete ===")


if __name__ == "__main__":
    main()
