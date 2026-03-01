#!/usr/bin/env python3
"""
Export AlexNet weights and activations for the full architecture.
Saves binary Float32 files and metadata for visualization.
Includes all conv/relu/pool layers, FC layers, and top-5 class predictions.
"""

import io
import json
import os
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import alexnet, AlexNet_Weights
from torchvision import transforms


# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Input size per spec (227x227)
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

    print("=== AlexNet Full Data Export ===\n")

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

    categories = AlexNet_Weights.DEFAULT.meta["categories"]

    # 4. Preprocess image
    print("\n3. Preprocessing image...")
    x = preprocess_image(img)

    # 5. Register hooks for ALL layers
    activations = {}

    def make_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach()
        return hook

    # Features: 0-12
    feature_hooks = [
        model.features[0].register_forward_hook(make_hook("conv1")),
        model.features[1].register_forward_hook(make_hook("relu1")),
        model.features[2].register_forward_hook(make_hook("pool1")),
        model.features[3].register_forward_hook(make_hook("conv2")),
        model.features[4].register_forward_hook(make_hook("relu2")),
        model.features[5].register_forward_hook(make_hook("pool2")),
        model.features[6].register_forward_hook(make_hook("conv3")),
        model.features[7].register_forward_hook(make_hook("relu3")),
        model.features[8].register_forward_hook(make_hook("conv4")),
        model.features[9].register_forward_hook(make_hook("relu4")),
        model.features[10].register_forward_hook(make_hook("conv5")),
        model.features[11].register_forward_hook(make_hook("relu5")),
        model.features[12].register_forward_hook(make_hook("pool5")),
    ]
    # Classifier: avgpool, then classifier layers (skip dropout)
    avgpool_hook = model.avgpool.register_forward_hook(make_hook("avgpool"))
    fc_hooks = [
        model.classifier[1].register_forward_hook(make_hook("fc6")),
        model.classifier[2].register_forward_hook(make_hook("relu6")),
        model.classifier[4].register_forward_hook(make_hook("fc7")),
        model.classifier[5].register_forward_hook(make_hook("relu7")),
        model.classifier[6].register_forward_hook(make_hook("fc8")),
    ]

    # 6. Full forward pass
    print("\n4. Running full forward pass...")
    with torch.no_grad():
        logits = model(x)

    # Remove hooks
    for h in feature_hooks + [avgpool_hook] + fc_hooks:
        h.remove()

    # 7. Compute flatten (avgpool output flattened) and softmax
    avgpool_out = activations["avgpool"]
    flatten_out = avgpool_out.view(1, -1)
    activations["flatten"] = flatten_out

    probs = F.softmax(logits, dim=1)
    activations["output"] = probs

    top5_probs, top5_indices = torch.topk(probs[0], 5)
    predictions = [
        {
            "class_idx": int(idx),
            "class_name": categories[idx],
            "probability": float(probs[0, idx].item()),
        }
        for idx in top5_indices
    ]

    # 8. Extract all weights
    print("\n5. Extracting weights...")
    weights_dir = os.path.join(data_dir, "weights")
    save_tensor_bin(model.features[0].weight, os.path.join(weights_dir, "conv1_weight.bin"))
    save_tensor_bin(model.features[0].bias, os.path.join(weights_dir, "conv1_bias.bin"))
    save_tensor_bin(model.features[3].weight, os.path.join(weights_dir, "conv2_weight.bin"))
    save_tensor_bin(model.features[3].bias, os.path.join(weights_dir, "conv2_bias.bin"))
    save_tensor_bin(model.features[6].weight, os.path.join(weights_dir, "conv3_weight.bin"))
    save_tensor_bin(model.features[6].bias, os.path.join(weights_dir, "conv3_bias.bin"))
    save_tensor_bin(model.features[8].weight, os.path.join(weights_dir, "conv4_weight.bin"))
    save_tensor_bin(model.features[8].bias, os.path.join(weights_dir, "conv4_bias.bin"))
    save_tensor_bin(model.features[10].weight, os.path.join(weights_dir, "conv5_weight.bin"))
    save_tensor_bin(model.features[10].bias, os.path.join(weights_dir, "conv5_bias.bin"))
    save_tensor_bin(model.classifier[1].weight, os.path.join(weights_dir, "fc6_weight.bin"))
    save_tensor_bin(model.classifier[1].bias, os.path.join(weights_dir, "fc6_bias.bin"))
    save_tensor_bin(model.classifier[4].weight, os.path.join(weights_dir, "fc7_weight.bin"))
    save_tensor_bin(model.classifier[4].bias, os.path.join(weights_dir, "fc7_bias.bin"))
    save_tensor_bin(model.classifier[6].weight, os.path.join(weights_dir, "fc8_weight.bin"))
    save_tensor_bin(model.classifier[6].bias, os.path.join(weights_dir, "fc8_bias.bin"))

    # 9. Save all activations
    print("\n6. Saving activations...")
    activations_dir = os.path.join(data_dir, "activations")
    save_tensor_bin(x, os.path.join(activations_dir, "input.bin"))
    for name in [
        "conv1", "relu1", "pool1", "conv2", "relu2", "pool2",
        "conv3", "relu3", "conv4", "relu4", "conv5", "relu5", "pool5",
        "avgpool", "flatten", "fc6", "relu6", "fc7", "relu7", "fc8", "output",
    ]:
        save_tensor_bin(activations[name], os.path.join(activations_dir, f"{name}.bin"))

    # 10. Build metadata
    print("\n7. Writing metadata.json...")
    conv1_weight = model.features[0].weight
    conv2_weight = model.features[3].weight
    conv3_weight = model.features[6].weight
    conv4_weight = model.features[8].weight
    conv5_weight = model.features[10].weight
    fc6 = model.classifier[1]
    fc7 = model.classifier[4]
    fc8 = model.classifier[6]

    def shape1d(t):
        return list(t.shape[1:])

    metadata = {
        "image": "cat",
        "predictions": predictions,
        "categories": list(categories),
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
                "shape": shape1d(activations["conv1"]),
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
                "shape": shape1d(activations["relu1"]),
                "activations_file": "activations/relu1.bin",
                "input_layer": "conv1"
            },
            {
                "id": "pool1",
                "type": "maxpool",
                "shape": shape1d(activations["pool1"]),
                "kernel_size": 3,
                "stride": 2,
                "padding": 0,
                "activations_file": "activations/pool1.bin",
                "input_layer": "relu1"
            },
            {
                "id": "conv2",
                "type": "conv",
                "shape": shape1d(activations["conv2"]),
                "groups": getattr(model.features[3], "groups", 1),
                "kernel_size": 5,
                "stride": 1,
                "padding": 2,
                "in_channels": shape1d(activations["pool1"])[0],
                "weights_file": "weights/conv2_weight.bin",
                "weights_shape": list(conv2_weight.shape),
                "bias_file": "weights/conv2_bias.bin",
                "activations_file": "activations/conv2.bin",
                "input_layer": "pool1"
            },
            {
                "id": "relu2",
                "type": "relu",
                "shape": shape1d(activations["relu2"]),
                "activations_file": "activations/relu2.bin",
                "input_layer": "conv2"
            },
            {
                "id": "pool2",
                "type": "maxpool",
                "shape": shape1d(activations["pool2"]),
                "kernel_size": 3,
                "stride": 2,
                "padding": 0,
                "activations_file": "activations/pool2.bin",
                "input_layer": "relu2"
            },
            {
                "id": "conv3",
                "type": "conv",
                "shape": shape1d(activations["conv3"]),
                "groups": 1,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "in_channels": shape1d(activations["pool2"])[0],
                "weights_file": "weights/conv3_weight.bin",
                "weights_shape": list(conv3_weight.shape),
                "bias_file": "weights/conv3_bias.bin",
                "activations_file": "activations/conv3.bin",
                "input_layer": "pool2"
            },
            {
                "id": "relu3",
                "type": "relu",
                "shape": shape1d(activations["relu3"]),
                "activations_file": "activations/relu3.bin",
                "input_layer": "conv3"
            },
            {
                "id": "conv4",
                "type": "conv",
                "shape": shape1d(activations["conv4"]),
                "groups": 1,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "in_channels": shape1d(activations["relu3"])[0],
                "weights_file": "weights/conv4_weight.bin",
                "weights_shape": list(conv4_weight.shape),
                "bias_file": "weights/conv4_bias.bin",
                "activations_file": "activations/conv4.bin",
                "input_layer": "relu3"
            },
            {
                "id": "relu4",
                "type": "relu",
                "shape": shape1d(activations["relu4"]),
                "activations_file": "activations/relu4.bin",
                "input_layer": "conv4"
            },
            {
                "id": "conv5",
                "type": "conv",
                "shape": shape1d(activations["conv5"]),
                "groups": 1,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "in_channels": shape1d(activations["relu4"])[0],
                "weights_file": "weights/conv5_weight.bin",
                "weights_shape": list(conv5_weight.shape),
                "bias_file": "weights/conv5_bias.bin",
                "activations_file": "activations/conv5.bin",
                "input_layer": "relu4"
            },
            {
                "id": "relu5",
                "type": "relu",
                "shape": shape1d(activations["relu5"]),
                "activations_file": "activations/relu5.bin",
                "input_layer": "conv5"
            },
            {
                "id": "pool5",
                "type": "maxpool",
                "shape": shape1d(activations["pool5"]),
                "kernel_size": 3,
                "stride": 2,
                "padding": 0,
                "activations_file": "activations/pool5.bin",
                "input_layer": "relu5"
            },
            {
                "id": "avgpool",
                "type": "adaptive_avg_pool",
                "shape": shape1d(activations["avgpool"]),
                "activations_file": "activations/avgpool.bin",
                "input_layer": "pool5"
            },
            {
                "id": "flatten",
                "type": "flatten",
                "shape": [int(flatten_out.shape[1])],
                "activations_file": "activations/flatten.bin",
                "input_layer": "avgpool"
            },
            {
                "id": "fc6",
                "type": "linear",
                "shape": [fc6.out_features],
                "out_features": fc6.out_features,
                "in_features": fc6.in_features,
                "weights_file": "weights/fc6_weight.bin",
                "weights_shape": list(fc6.weight.shape),
                "bias_file": "weights/fc6_bias.bin",
                "activations_file": "activations/fc6.bin",
                "input_layer": "flatten"
            },
            {
                "id": "relu6",
                "type": "relu",
                "shape": [int(activations["relu6"].shape[1])],
                "activations_file": "activations/relu6.bin",
                "input_layer": "fc6"
            },
            {
                "id": "fc7",
                "type": "linear",
                "shape": [fc7.out_features],
                "out_features": fc7.out_features,
                "in_features": fc7.in_features,
                "weights_file": "weights/fc7_weight.bin",
                "weights_shape": list(fc7.weight.shape),
                "bias_file": "weights/fc7_bias.bin",
                "activations_file": "activations/fc7.bin",
                "input_layer": "relu6"
            },
            {
                "id": "relu7",
                "type": "relu",
                "shape": [int(activations["relu7"].shape[1])],
                "activations_file": "activations/relu7.bin",
                "input_layer": "fc7"
            },
            {
                "id": "fc8",
                "type": "linear",
                "shape": [fc8.out_features],
                "out_features": fc8.out_features,
                "in_features": fc8.in_features,
                "weights_file": "weights/fc8_weight.bin",
                "weights_shape": list(fc8.weight.shape),
                "bias_file": "weights/fc8_bias.bin",
                "activations_file": "activations/fc8.bin",
                "input_layer": "relu7"
            },
            {
                "id": "output",
                "type": "softmax",
                "shape": [1000],
                "activations_file": "activations/output.bin",
                "input_layer": "fc8"
            },
        ],
    }

    with open(os.path.join(data_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved metadata.json")
    print("\nTop-5 predictions:")
    for p in predictions:
        print(f"  {p['class_idx']}: {p['class_name']} ({p['probability']:.4f})")
    print("\n=== Export complete ===")


if __name__ == "__main__":
    main()
