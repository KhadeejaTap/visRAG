import argparse
import glob
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, SiglipVisionModel
from transformers.image_utils import load_image

PROJECT_HOME = Path(__file__).resolve().parent.parent


def load_model(model_name):
    """
    param is dinov3_size or siglip
    """

    if model_name == "dinov3_small":
        # load it
        name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        processor = AutoImageProcessor.from_pretrained(name)

    else:
        # load it
        name = "google/siglip-base-patch16-224"
        processor = AutoProcessor.from_pretrained(name)

    model = AutoModel.from_pretrained(name).eval()
    return processor, model


def embed(processor, model, datadir):
    data = sorted(glob.glob(f"{datadir}/*_parts/train/images/*.jpg"))


def main():
    parser = argparse.ArgumentParser(description="cool")
    parser.add_argument(
        "--model", type=str, default="dinov3_small", help="dinov3_small or siglip"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default=f"{PROJECT_HOME}/data",
        help="parent of the subfolders",
    )
    parser.add_argument(
        "--embed", action="store_true", help="include this to embed before"
    )

    args = parser.parse_args()
    model_name = args.model
    imgs = args.img_dir
    processor, model = load_model(model_name)
    if args.embed:
        embed(processor, model, imgs)
    search()
