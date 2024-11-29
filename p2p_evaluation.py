import utils
import argparse
from diffusers import StableDiffusionInstructPix2PixPipeline
import torch
import json
from tqdm import tqdm
import os
from PIL import Image


def evaluate(args):
    
    # create results directory if not exists
    os.makedirs(args.savedir, exist_ok=True)

    # load dataset
    dataset = utils.MBDataset(data_dir=args.data_dir)
    evaluation_dataset = dataset.build_evaluation_dataset()

    # sample dataset
    evaluation_dataset = evaluation_dataset.sample(n=args.sample_size)

    # load pipeline
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
    )
    pipe.to(args.device)


    # run evaluation
    results_json = []

    for idx, row in tqdm(evaluation_dataset.iterrows()):
        img_path = row.path
        prompt = row.prompt

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