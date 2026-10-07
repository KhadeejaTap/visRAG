import argparse
import base64
import glob
import io
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from google import genai
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, SiglipVisionModel
from transformers.image_utils import load_image

matplotlib.use("QtAgg")

load_dotenv()
client = genai.Client()
PROJECT_HOME = Path(__file__).resolve().parent.parent


def load_model(model_name):
    """
    param is dinov3_small or siglip
    """
    if model_name == "dinov3_small":
        name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        processor = AutoImageProcessor.from_pretrained(name)
        model = AutoModel.from_pretrained(name).eval()

    else:
        # Load the FULL multimodal model for SigLIP
        name = "google/siglip-base-patch16-224"
        processor = AutoProcessor.from_pretrained(name)
        model = AutoModel.from_pretrained(name).eval()

    return processor, model


LABEL_MAP = {"0": "engine", "1": "nozzle",
             "2": "pump", "3": "separator", "4": "valve"}


def get_label_for_image(datadir, img_path):
    """Reads the YOLO txt file and returns the string name of the class."""
    img_name = Path(img_path).stem

    # get matching txt file
    match = str(img_path).replace(
        "/images/", "/labels/").replace(".jpg", ".txt")
    with open(match, "r") as f:
        first_line = f.readline().strip()
        class_id = first_line.split()[0]  # The first number
        return LABEL_MAP.get(class_id, "unknown")


def embed_image(processor, model, img):
    # Support both file paths (for main.py) and PIL Images (for app.py)
    if isinstance(img, (str, Path)):
        img = load_image(img)

    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        if hasattr(model, "get_image_features"):
            # SigLIP multimodal model
            out = model.get_image_features(**inputs)
        else:
            # DINOv3 vision model
            out = model(**inputs)

    if hasattr(out, "pooler_output"):
        emb = out.pooler_output
    else:
        emb = out

    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


def embed(processor, model, input, datadir):
    output = []
    for img_path in input:
        emb = embed_image(processor, model, img_path)

        if hasattr(model, "get_text_features"):
            label_name = get_label_for_image(datadir, img_path)
            if label_name != "unknown":
                text_prompt = f"a photo of a {label_name}"
                text_emb = embed_text(processor, model, text_prompt)

                emb = (emb + text_emb) / 2
                emb = emb / emb.norm(dim=-1, keepdim=True)

        output.append(emb)
    return output


def return_embeddings(processor, model, datadir):
    data = sorted(glob.glob(f"{datadir}/*_parts/train/images/*.jpg"))
    embeddings = embed(processor, model, data, datadir)

    test_data = sorted(glob.glob(f"{datadir}/*_parts/valid/images/*.jpg"))
    test_embeddings = embed(processor, model, test_data, datadir)
    return embeddings, test_embeddings


def embed_text(processor, model, text):
    inputs = processor(
        text=[text], padding="max_length", max_length=64, return_tensors="pt"
    )
    with torch.no_grad():
        out = model.get_text_features(**inputs)

    if hasattr(out, "pooler_output"):
        emb = out.pooler_output
    else:
        emb = out

    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


def get_embeddings(embed_flag, processor, model, model_name, datadir, text=None):
    saved_catalog = Path(f"{datadir}/{model_name}_embeddings.pt")

    if saved_catalog.exists() and not embed_flag:
        print("Loading saved embeddings")
        catalog_embs, test_embs = torch.load(saved_catalog)
    else:
        print("Generating new embeddings")
        catalog_embs, test_embs = return_embeddings(processor, model, datadir)
        torch.save((catalog_embs, test_embs), saved_catalog)

    if text is not None:
        print(f"Embedding text query: '{text}'")
        text_emb = embed_text(processor, model, text)
        return catalog_embs, text_emb
    else:
        return catalog_embs, test_embs


def show_sample(datadir):
    valid = sorted(glob.glob(f"{datadir}/*_parts/valid/images/*.jpg"))
    rng = np.random.default_rng()
    idx = rng.integers(0, len(valid))
    test_img = Image.open(f"{valid[idx]}")

    """
        plt.figure()
        plt.imshow(test_img)
        plt.xlabel("input image")
        plt.show(block=False)
        plt.pause(0.1)"""
    return idx


def search(embedded_catalog, embedded_sample, datadir):
    catalog_paths = sorted(glob.glob(f"{datadir}/*_parts/train/images/*.jpg"))
    print("computing similarity")
    catalog_matrix = torch.cat(embedded_catalog)
    scores = torch.matmul(embedded_sample, catalog_matrix.T)
    top_scores, top_indices = torch.topk(scores, k=1)

    top_indices = top_indices[0].tolist()
    for rank, idx in enumerate(top_indices):
        winning_img_path = catalog_paths[idx]
        """
        i = Image.open(f"{winning_img_path}")
        plt.figure()
        plt.imshow(i)
        plt.xlabel(f"Rank {rank + 1}")
        plt.show(block=False)
        plt.pause(0.1) """
    # plt.show(block=True)
    best_idx = top_indices[0]  # rank 1 index (int)
    return Path(catalog_paths[best_idx])


def prompt_llm(llm_model, prompt, top_image, datadir, input_image):
    label = top_image.parents[1] / "labels" / \
        top_image.with_suffix(".txt").name
    text = label.read_text().split()
    class_id = int(text[0])
    class_name = LABEL_MAP[str(class_id)]  # input 1
    with open(f"{datadir}/class_descriptions.json") as f:
        descriptions = json.load(f)
    desc = descriptions[class_name]  # input 2 consider concatting?
    text_input = f"Determine if the classification of the object is accurate: Class: {class_name}. If so, consider the following information: description: {desc}. Concisely answer this prompt from the user: {prompt}. Flag unconfident parts. The following is the input image followed by the top match of which the class and description was given."
    # top_image is input 3 (the path of the matched img)
    # future: consider passing multiple imgs + descriptions n use llm as tiebreaker
    # input_image is input 4 (its the path to the input img)
    img = Image.open(input_image).convert("RGB")
    img.thumbnail((300, 300))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    input_bytes = buf.getvalue()

    img = Image.open(top_image).convert("RGB")
    img.thumbnail((300, 300))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    top_bytes = buf.getvalue()
    input = [
        {"type": "text", "text": text_input},
        {
            "type": "image",
            "data": base64.b64encode(input_bytes).decode("utf-8"),
            "mime_type": "image/jpeg",
        },
        {
            "type": "image",
            "data": base64.b64encode(top_bytes).decode("utf-8"),
            "mime_type": "image/jpeg",
        },
    ]
    interaction = client.interactions.create(model=llm_model, input=input)
    print(interaction.output_text)


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
        "--embed", action="store_true", help="include this to force re-embed"
    )
    parser.add_argument("--text", type=str, help="optional text description")
    parser.add_argument(
        "--image_path",
        type=str,
        help="optional path to a specific image to search with",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="what does this item do in a ship",
        help="what do u wanna ask the llm abt the pic or item u described",
    )
    args = parser.parse_args()

    model_name = args.model
    imgs = args.img_dir
    text = args.text
    image_path = args.image_path
    prompt = args.prompt
    processor, model = load_model(model_name)

    # Always fetch the pure valid test set so we don't break fallback logic
    catalog_embeddings, test_embeddings = get_embeddings(
        args.embed, processor, model, model_name, imgs, None
    )

    if text and image_path:
        print(f"HYBRID QUERY: Image ({image_path}) + Text ('{text}')")
        img_emb = embed_image(processor, model, image_path)
        txt_emb = embed_text(processor, model, text)
        emb = (img_emb + txt_emb) / 2
        emb = emb / emb.norm(dim=-1, keepdim=True)
        top_idx = search(catalog_embeddings, emb, imgs)
    elif text:
        print(f"TEXT QUERY: '{text}'")
        txt_emb = embed_text(processor, model, text)
        top_idx = search(catalog_embeddings, txt_emb, imgs)
    elif image_path:
        print(f"IMAGE QUERY: '{image_path}'")
        img_emb = embed_image(processor, model, image_path)
        top_idx = search(catalog_embeddings, img_emb, imgs)
    else:
        print("RANDOM QUERY: Picking from valid set")
        idx = show_sample(imgs)
        top_idx = search(catalog_embeddings, test_embeddings[idx], imgs)
        valid = sorted(glob.glob(f"{imgs}/*_parts/valid/images/*.jpg"))
        image_path = valid[idx]

    prompt_llm(
        llm_model="gemini-3.5-flash-lite",
        prompt=prompt,
        input_image=image_path,
        top_image=top_idx,
        datadir=imgs
    )


if __name__ == "__main__":
    main()
