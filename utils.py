from tqdm import tqdm
import prompts
import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

# DATA_DIR = "data/test-MagicBrush/test"

class EditTurn:

    def __init__(self, sample, data_dir):
        self.image_id = sample["input"].split("-")[0]
        self._input = sample["input"]
        self._output = sample["output"]
        self._mask = sample["mask"]
        self._instruction = sample["instruction"]
        self._input_description = ""
        self.data_dir = data_dir

    def _get_input_description(self, glb_descriptions):
        """
        Fetch input description from the global description JSON
        """
        self._input_description = glb_descriptions[self.image_id][f"{self.image_id}-input.png"]

    def to_json(self):
        """
        Converts the instance attributes to a JSON serializable dictionary.
        Returns:
            dict: A dictionary containing the instance attributes 'input', 'output', 
                  'mask', and 'instruction'.
        """
        
        return {
            "input": self._input,
            "output": self._output,
            "mask": self._mask,
            "instruction": self._instruction,
            "desdcription": self._input_description
        }

class MBDataset:

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

        # load edit turns
        self.edit_turns = []

        with open(os.path.join(data_dir, "global_descriptions.json"), "r") as f:
            _global_description = json.load(f)

        with open(os.path.join(data_dir, "edit_turns.json"), "r") as f:
            _data = json.load(f)
            print("Loading edit turns...")
            for sample in tqdm(_data):
                edit_turn = EditTurn(sample, data_dir)
                edit_turn._get_input_description(_global_description)
                self.edit_turns.append(edit_turn)

    def __len__(self):
        return len(self.edit_turns)
    
    def __getitem__(self, idx):
        return self.edit_turns[idx]
    
    def sample(self, n=1):
        """
        Generate a sample of edit turns.
        Parameters:
        n (int): The number of samples to generate. Default is 1.
        Returns:
        list: A list containing 'n' randomly selected edit turns.
        """
        
        return list(np.random.choice(self.edit_turns, n, replace=False))
    
    def get_instructions(self):
        return pd.DataFrame([edit_turn._instruction for edit_turn in self.edit_turns],
                            columns=["prompt"])
    
    def get_unique_inputs(self):
        """
        Get a deduplicated list of input images (paths) and their corresponding descriptions.
        """
        inputs = pd.DataFrame([(edit_turn._input, edit_turn._input_description) if "input" in edit_turn._input else None for edit_turn in self.edit_turns],
                              columns=["image", "description"])
        inputs = inputs.dropna().drop_duplicates()
        inputs["image_id"] = inputs["image"].apply(lambda x: x.split("-")[0])
        inputs["image_path"] = inputs["image"].apply(lambda x: os.path.join(self.data_dir, f"images/{x.split('-')[0]}", x))

        return inputs
    
    def build_evaluation_dataset(self):
        """
        Generate a dataset for evaluation of pix2pix. Get unique inputs and create
        (input data, prompt) pairs for all available prompts in the prompts.py file.
        """

        imgs = self.get_unique_inputs()
        _prompts = prompts.PROMPTS

        img_tuples = []
        for path in imgs.image_path.tolist():
            for prompt_type, prompt_list in _prompts.items():
                for prompt in prompt_list:
                    img_tuples.append({
                        "path" : path,
                        "prompt" : prompt,
                        "prompt_type" : prompt_type
                    })

        return pd.DataFrame(img_tuples)
    
    def plot(self, idx):
        """
        Plots the input, output, and mask images side by side for comparison for a given edit turn index.
        This method creates a figure with three subplots:
        - The first subplot displays the input image.
        - The second subplot displays the output image.
        - The third subplot displays the mask image.
        Each subplot is titled accordingly, and the axes are turned off for better visualization.
        The figure also includes a main title that describes the instruction associated with the images.
        Parameters:
            idx (int): The index of the edit turn in the edit_turns list.
        Raises:
            FileNotFoundError: If any of the image files cannot be found in the specified directory.
        """

        edit_turn = self.edit_turns[idx]

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # before
        img = Image.open(os.path.join(self.data_dir, f"images/{edit_turn.image_id}", edit_turn._input))
        ax[0].imshow(img)
        ax[0].set_title("input")

        # after
        img = Image.open(os.path.join(self.data_dir, f"images/{edit_turn.image_id}", edit_turn._output))
        ax[1].imshow(img)
        ax[1].set_title("output")

        # mask
        img = Image.open(os.path.join(self.data_dir, f"images/{edit_turn.image_id}", edit_turn._mask))
        ax[2].imshow(img)
        ax[2].set_title("mask")

        for sax in ax:
            sax.axis("off")

        fig.suptitle(edit_turn._instruction)

        plt.show()
