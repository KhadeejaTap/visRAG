import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, SiglipVisionModel
from transformers.image_utils import load_image

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


LABEL_MAP = {
    "0": "engine",
    "1": "nozzle",
    "2": "pump",
    "3": "separator",
    "4": "valve"
}

def get_label_for_image(img_path):
    """Reads the YOLO txt file and returns the string name of the class."""
    img_path = Path(img_path)
    # Swap /images/ for /labels/ and .jpg for .txt
    label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
    
    if label_path.exists():
        with open(label_path, "r") as f:
            first_line = f.readline().strip()
            if first_line:
                class_id = first_line.split()[0] # The first number
                return LABEL_MAP.get(class_id, "unknown")
    return "unknown"


def embed_image_single(processor, model, img):
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
            
    # Same Hugging Face quirk as the text model!
    if hasattr(out, "pooler_output"):
        emb = out.pooler_output
    else:
        emb = out
        
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


def embed(processor, model, input):
    output = []
    for img_path in input:
        # 1. Start with the image embedding
        emb = embed_image_single(processor, model, img_path)
        
        # 2. If we are using SigLIP, grab the text label and average it in!
        if hasattr(model, "get_text_features"):
            label_name = get_label_for_image(img_path)
            if label_name != "unknown":
                text_prompt = f"a photo of a {label_name}"
                text_emb = embed_text(processor, model, text_prompt)
                
                # Average them together to create a "hybrid" concept vector
                emb = (emb + text_emb) / 2
                emb = emb / emb.norm(dim=-1, keepdim=True)
                
        output.append(emb)
    return output


def return_embeddings(processor, model, datadir):
    data = sorted(glob.glob(f"{datadir}/*_parts/train/images/*.jpg"))
    embeddings = embed(processor, model, data)

    test_data = sorted(glob.glob(f"{datadir}/*_parts/valid/images/*.jpg"))
    test_embeddings = embed(processor, model, test_data)
    return embeddings, test_embeddings


def embed_text(processor, model, text):
    inputs = processor(
        text=[text],
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = model.get_text_features(**inputs)

    # Depending on the transformers version, it either returns the raw tensor
    # or an output object. We need to grab the pooler_output if it's an object!
    if hasattr(out, "pooler_output"):
        emb = out.pooler_output
    else:
        emb = out

    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


def get_embeddings(embed_flag, processor, model, model_name, datadir, text=None):
    saved_catalog = Path(f"{datadir}/{model_name}_embeddings.pt")

    # 1. Load or Generate the Catalog
    if saved_catalog.exists() and not embed_flag:
        print("Loading saved embeddings")
        catalog_embs, test_embs = torch.load(saved_catalog)
    else:
        print("Generating new embeddings")
        catalog_embs, test_embs = return_embeddings(processor, model, datadir)
        torch.save((catalog_embs, test_embs), saved_catalog)

    # 2. Handle Text vs Image queries
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
    plt.figure()  # Create a specific window for the input
    plt.imshow(test_img)
    plt.xlabel("input image")
    plt.show(block=False)  # Don't pause the script!
    plt.pause(0.1)  # Give the OS time to draw it
    return idx


def search(embedded_catalog, embedded_sample, datadir):
    catalog_paths = sorted(glob.glob(f"{datadir}/*_parts/train/images/*.jpg"))
    print("computing similarity")
    catalog_matrix = torch.cat(embedded_catalog)
    scores = torch.matmul(embedded_sample, catalog_matrix.T)
    top_scores, top_indices = torch.topk(scores, k=5)

    top_indices = top_indices[0].tolist()
    for rank, idx in enumerate(top_indices):
        winning_img_path = catalog_paths[idx]
        i = Image.open(f"{winning_img_path}")
        plt.figure()
        plt.imshow(i)
        plt.xlabel(f"Rank {rank + 1} match (Index {idx})")
        plt.show(block=False)
        plt.pause(0.1)

    print("Search complete! Close the image windows to exit.")
    plt.show(block=True)


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
    args = parser.parse_args()

    model_name = args.model
    imgs = args.img_dir
    text = args.text
    processor, model = load_model(model_name)

    catalog_embeddings, query_embedding = get_embeddings(
        args.embed, processor, model, model_name, imgs, text
    )

    if text:
        # If we passed text, query_embedding is a single tensor!
        search(catalog_embeddings, query_embedding, imgs)
    else:
        # If we didn't pass text, query_embedding is the list of valid images
        idx = show_sample(imgs)
        search(catalog_embeddings, query_embedding[idx], imgs)


if __name__ == "__main__":
    main()
