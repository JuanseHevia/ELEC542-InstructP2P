import utils
import pandas as pd
import argparse
from diffusers import StableDiffusionInstructPix2PixPipeline
import torch
import json
from tqdm import tqdm
import os
from PIL import Image

import logging
logging.getLogger("diffusers").setLevel(logging.ERROR) 
# or
logging.getLogger("diffusers").setLevel(logging.WARNING)


def evaluate(args):
    
    # create results directory if not exists
    os.makedirs(args.savedir, exist_ok=True)

    # load dataset
    dataset = utils.MBDataset(data_dir=args.data_dir)
    evaluation_dataset = dataset.build_evaluation_dataset()

    # sample dataset
    # evaluation_dataset = evaluation_dataset.sample(n=args.sample_size)
    color_ds = evaluation_dataset[evaluation_dataset.prompt_type == "color"].head(args.sample_size//2)
    concp_ds = evaluation_dataset[evaluation_dataset.prompt_type == "conceptual"].head(args.sample_size//2)
    
    data = pd.concat([color_ds, concp_ds], axis=0)
    print(data.info())
    print(data.sample(5))
    print("Start evaluation...\n\n\n\n")
    # load pipeline
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16,

    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to(args.device)


    # run evaluation
    results_json = []

    for idx, row in tqdm(data.iterrows()):
        img_path = row.path
        prompt = row.prompt

        # get image ID
        image_id = img_path.split("/")[-1].split(".")[0].replace("-input", "")

        # read image
        _input_image = Image.open(img_path)
        _input_image = _input_image.convert("RGB")

        result_image = pipe(prompt=prompt, image=_input_image,
                        num_inference_steps=args.inference_steps).images[0]

        fname = f"EDS{idx}.png"
        result_image.save(os.path.join(args.savedir, fname))
        results_json.append({
            "path": fname,
            "prompt": prompt,
            "image_path": img_path,
            "image_id": image_id,
            "prompt_type": row.prompt_type,
            "evaluation_dataset_idx": idx
        })

        # checkpoint every 250 generations
        if idx % args.dump_every == 0:
            with open(f"{args.savedir}/results-{idx}.json", "w") as f:
                json.dump(results_json, f)

    with open("results-FINAL.json", "w") as f:
        json.dump(results_json, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dump_every", type=int, default=500)
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--savedir", type=str, default="results")
    parser.add_argument("--sample_size", type=int, default=1000)
    args = parser.parse_args()

    evaluate(args)