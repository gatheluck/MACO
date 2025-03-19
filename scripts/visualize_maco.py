import argparse
import pathlib

import timm
import torch
import torchvision.utils as vutils
from torchvision.transforms import Compose, Normalize

from src.maco import fourier_to_image, normalize, normalize_alpha, run_maco

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mapping_to_timm_model = {
    "resnet50": "timm/resnet50.tv_in1k",  # https://huggingface.co/timm/resnet50.tv_in1k
    "resnet50v2": "timm/resnetv2_50.a1h_in1k",  # https://huggingface.co/timm/resnetv2_50.a1h_in1k
    "mobilenet": "timm/mobilenetv3_large_100.ra_in1k",  # https://huggingface.co/timm/mobilenetv3_large_100.ra_in1k
    "inception": "inception_next_small.sail_in1k",  # https://huggingface.co/timm/inception_next_small.sail_in1k
    "vit": "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k",  # https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--unnormalized-average-magnitude-path",
    "-i",
    type=pathlib.Path,
    default=pathlib.Path("outputs/unnormalized_average_magnitude.pth"),
)
parser.add_argument(
    "--output-dir-path",
    "-o",
    type=pathlib.Path,
    default=pathlib.Path("outputs/"),
)
parser.add_argument(
    "--model",
    "-m",
    choices=mapping_to_timm_model.keys(),
    default="resnet50",
)
parser.add_argument(
    "--image-size",
    type=int,
    default=512,
    help="Image size for visualization. This is NOT model input image size.",
)
parser.add_argument(
    "--target-logit-idx",
    "-t",
    type=int,
    default=0,
)
parser.add_argument(
    "--num_steps",
    "-s",
    type=int,
    default=256,
)
parser.add_argument(
    "--num-crops",
    "-c",
    type=int,
    default=32,
)
parser.add_argument(
    "--noise-std",
    "-n",
    type=float,
    default=-1.0,
)
args = parser.parse_args()

# load the unnormalized average magnitude.
if not args.unnormalized_average_magnitude_path.exists():
    raise FileNotFoundError(
        f"Unnormalized average magnitude file not found: {args.unnormalized_average_magnitude_path}"
    )

unnormalized_average_magnitude = torch.load(args.unnormalized_average_magnitude_path)

# upscale the unnormalized_average_magnitude.
unnormalized_average_magnitude = unnormalized_average_magnitude.unsqueeze(0)
unnormalized_average_magnitude = torch.nn.functional.interpolate(
    unnormalized_average_magnitude,
    size=(args.image_size, args.image_size),
    mode="bilinear",
    align_corners=False,
)
unnormalized_average_magnitude = unnormalized_average_magnitude.squeeze(0)

# load the model.
model = timm.create_model(mapping_to_timm_model[args.model], pretrained=True)
model = model.eval()

# get model specific transforms (normalization, resize).
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)
normalize_transform = next(
    (t for t in transform.transforms if isinstance(t, Normalize)), None
)
if normalize_transform is None:
    raise ValueError("Normalize transform not found in model transforms.")

# run MACO algorithm.
phase, alpha = run_maco(
    model,
    device,
    unnormalized_average_magnitude,
    Compose([normalize_transform]),
    target_logit_idx=args.target_logit_idx,
    num_steps=args.num_steps,
    num_crops=args.num_crops,
    noise_std=args.noise_std,
    learning_rate=1.0,
)

# reconstruct the image from the optimized phase and alpha.
x = fourier_to_image(unnormalized_average_magnitude, phase)
alpha_normalized = normalize_alpha(alpha)
x_alpha = normalize(x * alpha_normalized)

vutils.save_image(
    [x, alpha_normalized.repeat(3, 1, 1), x_alpha],
    args.output_dir_path / "maco_result.png",
)
