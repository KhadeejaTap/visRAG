import argparse
import glob
from pathlib import Path

import gradio as gr
import torch

# Import the helper functions you already wrote!
from main import (
    PROJECT_HOME,
    embed_image,
    embed_text,
    get_embeddings,
    get_label_for_image,
    load_model,
)
from PIL import Image

parser = argparse.ArgumentParser(description="Gradio Search Engine")
parser.add_argument(
    "--model", type=str, default="dinov3_small", help="dinov3_small or siglip"
)
parser.add_argument("--embed", action="store_true", help="force re-embed")
args = parser.parse_args()

print(f"Booting up the search engine using {args.model}...")

# 1. One-time setup: Load the model and catalog into memory when the server starts
model_name = args.model
datadir = f"{PROJECT_HOME}/data"

processor, model = load_model(model_name)
catalog_embeddings, _ = get_embeddings(
    args.embed, processor, model, model_name, datadir
)

# Stack the catalog and get the paths once so it's lightning fast
catalog_matrix = torch.cat(catalog_embeddings)
catalog_paths = sorted(glob.glob(f"{datadir}/*_parts/train/images/*.jpg"))

print("Ready! Starting web server...")


def predict(uploaded_image, search_text):
    """This runs every time a user clicks search in the UI"""

    has_text = bool(search_text and search_text.strip())
    has_image = bool(uploaded_image is not None)

    if has_text and model_name != "siglip":
        raise gr.Error("Text search only works if you booted with --model siglip!")

    if has_text and has_image:
        # --- HYBRID SEARCH (BOTH) ---
        # 50/50 average of the text and image embeddings
        img_emb = embed_image(processor, model, uploaded_image)
        txt_emb = embed_text(processor, model, search_text.strip())
        emb = (img_emb + txt_emb) / 2
        emb = emb / emb.norm(dim=-1, keepdim=True)

    elif has_text:
        # --- TEXT ONLY ---
        emb = embed_text(processor, model, search_text.strip())

    elif has_image:
        # --- IMAGE ONLY ---
        emb = embed_image(processor, model, uploaded_image)

    else:
        # User didn't provide anything!
        return []

    # 2. Get similarity scores against the entire catalog
    scores = torch.matmul(emb, catalog_matrix.T)

    # 3. Sort the ENTIRE catalog (k=len(catalog_paths))
    top_scores, top_indices = torch.topk(scores, k=len(catalog_paths))
    top_indices = top_indices[0].tolist()

    # 4. Format the output for the Gradio Gallery
    # Gradio Gallery expects a list of tuples: (image_path, "caption string")
    gallery_items = []
    for rank, idx in enumerate(top_indices):
        path = catalog_paths[idx]
        label = get_label_for_image(datadir, path)
        caption = f"Rank {rank + 1} ({label})"
        gallery_items.append((path, caption))

    return gallery_items


# --- Define the User Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("Visual Search Engine")
    gr.Markdown(
        "Upload an image, type a description, or do BOTH to search the catalog!"
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Search by Text (SigLIP only)",
                placeholder="e.g. 'a rusted red valve'",
            )
            input_image = gr.Image(type="pil", label="...and/or Upload Query Image")
            search_button = gr.Button("Search", variant="primary")

        with gr.Column(scale=3):
            # A gallery component to show the grid of results
            output_gallery = gr.Gallery(
                label="Ranked Catalog Results",
                columns=5,  # Show 5 images per row
                height="auto",
            )

    # Wire the button to the function
    search_button.click(
        fn=predict, inputs=[input_image, input_text], outputs=output_gallery
    )

if __name__ == "__main__":
    # This launches the local web server!
    demo.launch()
