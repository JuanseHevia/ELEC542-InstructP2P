from mmcq import get_palette, get_dominant_color
from prompts import CONCEPTUAL_PROMPTS, COLOR_PROMPTS, TEMPLATE_PROMPT
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from prompts import CONCEPTUAL_PROMPTS, COLOR_PROMPTS, TEMPLATE_PROMPT


@dataclass
class PaletteAnalysis:

    img_path: str
    results_path: str
    results_summary_path: str
    PALETTE_SIZE: int = 5

    def __post_init__(self):
        # read JSON results
        with open(self.results_summary_path) as f:
            self.results = json.load(f)

        self.results = pd.DataFrame(self.results)
        self.results["edit"] = self.results.prompt.str.replace(
            TEMPLATE_PROMPT, "").str.replace(".", "").str.strip()
        self.image_palettes = []

        # load palettes
        palettes = []
        dom_colors = []
        for i, path in enumerate(tqdm(self.results.path)):
            palette = ImagePalette(os.path.join(
                self.results_path, "results", path))
            self.image_palettes.append(palette)
            palettes.append(palette.get_colors())
            dom_colors.append(palette.dominant_color)

        self.results["palette"] = palettes
        self.results["palette"] = self.results.palette.apply(
            lambda x: np.array(x))
        self.results["dominant_color"] = dom_colors
        self.results["dominant_color"] = self.results.dominant_color.apply(
            lambda x: np.array(x))
        

    def compute_metric(self, idx: int):
        row = self.results.iloc[idx]
        img = Image.open(os.path.join(self.results_path, "results", row.path))
        img = np.array(img)
        original_img = Image.open(os.path.join(self.img_path, row.image_path))
        original_img = np.array(original_img)
        cm = ColorMetrics(img, original_img)
        return cm.compare_metrics()
    
    def add_metrics(self):
        metrics = []
        for idx in range(self.results.shape[0]):
            _m = self.compute_metric(idx)
            metrics.append(_m)

        self.metrics = pd.DataFrame({
            "generated": [m["generated"] for m in metrics],
            "original": [m["original"] for m in metrics],
            "idx": list(range(self.results.shape[0]))
        })
        return metrics

    def plot_observation(self, idx: int, add_edit: bool = False, add_original: bool = False, savedir: str = None, fname: str = None):
        row = self.results.iloc[idx]

        img = Image.open(os.path.join(self.results_path, "results/", row.path))

        if add_original:
            img_original = Image.open(
                os.path.join(self.img_path, row.image_path))
            plt.subplot(1, 2, 1)
            plt.imshow(img_original)
            plt.axis("off")
            plt.title("original")
            plt.subplot(1, 2, 2)
            plt.imshow(img)
            plt.axis("off")
            if add_edit:
                plt.title(row.edit)
        else:
            plt.imshow(img)
            plt.axis("off")
            if add_edit:
                plt.title(row.edit)

        if savedir:
            _fname = fname if fname else f"observation_{idx}.png"
            plt.savefig(os.path.join(savedir, _fname))
        plt.show()

    def get_average_colors(self):

        avg_colors = self.results.groupby("edit")\
            .dominant_color\
            .apply(lambda col_series: col_series.values.reshape(1, col_series.shape[0]).mean(1)[0])\
            .to_dict()

        return avg_colors

    def _compute_avg_palette(self, img_palettes, seed: int = 42):
        """
        Given an series of 5-element arrays with the colors present in an image
        compute the average color palette by doing:

        1. take each color separately and build a (5 * len(img_palettes), 3) vector
        2. create a cluster of 5 colors using kmeans
        3 return the cluster centers
        """
        colors = np.concatenate(img_palettes.values)
        kmeans = KMeans(n_clusters=self.PALETTE_SIZE,
                        random_state=seed).fit(colors)
        return kmeans.cluster_centers_

    def get_avg_palettes(self):
        avg_palettes = self.results.groupby("edit")\
            .palette\
            .apply(lambda col_series: self._compute_avg_palette(col_series))\
            .to_dict()

        # edit the keys to match the prompt
        avg_palettes = {k.replace(TEMPLATE_PROMPT.format(term="").replace(
            ".", ""), "").strip(): v for k, v in avg_palettes.items()}

        return avg_palettes

    def _plot_palette(self, palette, ax=None, title: str = None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        for i, color in enumerate(palette):
            # normalize
            norm_color = [c / 255 for c in color]
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=norm_color))

        ax.set_xlim(0, len(palette))
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        if title:
            ax.set_title(title)
        ax.axis("off")

        return ax

    def plot_avg_palette(self, avg_palettes, title: str = None, savepath: str = None):
        num_colors = len(avg_palettes)
        num_cols = 5
        num_rows = (num_colors + num_cols - 1) // num_cols

        fig, ax = plt.subplots(num_rows, num_cols, figsize=(20, 2 * num_rows))

        for idx, cp in enumerate(CONCEPTUAL_PROMPTS):
            palette = avg_palettes[cp.lower()]
            self._plot_palette(
                palette=palette, ax=ax[idx // num_cols, idx % num_cols], title=cp)

        for idx, cp in enumerate(COLOR_PROMPTS):
            palette = avg_palettes[cp.lower()]
            self._plot_palette(
                palette=palette, ax=ax[idx // num_cols + 1, idx % num_cols], title=cp)

        plt.tight_layout()
        fig.suptitle(title)
        if savepath:
            plt.savefig(savepath)

        plt.show()


class ImagePalette:
    def __init__(self, path: str, color_count: int = 5):
        self.path = path
        self.color_count = color_count
        with get_palette(path, color_count=color_count) as palette:
            self.palette = palette

        self.normalized_colors = np.array(self.palette) / 255
        self.dominant_color = get_dominant_color(path)

    def __repr__(self):
        return f"ImagePalette(path={self.path}, color_count={self.color_count}),colors={self.palette})"

    def __str__(self):
        return " ".join([f"rgb({c[0]}, {c[1]}, {c[2]})" for c in self.palette])

    def get_colors(self):
        return self.palette

    def plot(self, ax=None, title: str = None):
        if ax is None:
            _, ax = plt.subplots(1, 1)

        for i, nc in enumerate(self.normalized_colors):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=nc))

        ax.set_xlim(0, len(self.normalized_colors))
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        if title:
            ax.set_title(title)
        ax.axis("off")

        return ax


from skimage import color
import numpy as np

class ColorMetrics:
    def __init__(self, generated_img, original_img):
        """
        Initialize with generated and original images.
        :param generated_img: The generated image as a NumPy array.
        :param original_img: The original image as a NumPy array.
        """
        self.generated_img = generated_img
        self.original_img = original_img

    def _saturation(self, img):
        """
        Calculate the mean saturation of an image.
        :param img: Image in RGB format as a NumPy array.
        :return: Mean saturation value.
        """
        hsv_image = color.rgb2hsv(img)
        return np.mean(hsv_image[:, :, 1])

    def _brightness(self, img):
        """
        Calculate the mean brightness of an image.
        :param img: Image in RGB format as a NumPy array.
        :return: Mean brightness value.
        """
        gray_image = color.rgb2gray(img)
        return np.mean(gray_image)

    def _colorfulness(self, img):
        """
        Calculate the colorfulness of an image.
        :param img: Image in RGB format as a NumPy array.
        :return: Colorfulness metric.
        """
        # Split image into R, G, B channels
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        rg = np.abs(R - G)
        yb = np.abs(0.5 * (R + G) - B)

        # Calculate the mean and standard deviation of rg and yb
        std_rg, std_yb = np.std(rg), np.std(yb)
        mean_rg, mean_yb = np.mean(rg), np.mean(yb)

        # Compute the colorfulness metric
        colorfulness = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
        return colorfulness

    def _contrast(self, img):
        """
        Calculate the contrast of an image using the standard deviation of intensity values.
        :param img: Image in RGB format as a NumPy array.
        :return: Contrast value.
        """
        gray_image = color.rgb2gray(img)
        return np.std(gray_image)

    def compare_metrics(self):
        """
        Compare metrics between the generated and original images.
        :return: Dictionary of metrics for generated and original images.
        """
        metrics = {}
        metrics["generated"] = {
            "saturation": self._saturation(self.generated_img),
            "brightness": self._brightness(self.generated_img),
            "colorfulness": self._colorfulness(self.generated_img),
            "contrast": self._contrast(self.generated_img),
        }
        metrics["original"] = {
            "saturation": self._saturation(self.original_img),
            "brightness": self._brightness(self.original_img),
            "colorfulness": self._colorfulness(self.original_img),
            "contrast": self._contrast(self.original_img),
        }
        return metrics