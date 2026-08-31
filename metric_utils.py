import argparse
import csv
import glob
import os
from os.path import join as osp
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms


DEFAULT_INPUT_PATH = "./load/data"
DEFAULT_DATASETS = ("realfusion15", "morpheusobj30")
DEFAULT_METRICS = ("clip", "maniqa", "clipiqa", "psnr", "lpips")
PYIQA_METRICS = ("maniqa", "clipiqa")
IQA_IMAGE_SIZE = 512
NUM_VIEWS = 100
METRIC_CHECKPOINT_ROOT = Path(__file__).resolve().parent / "metric_ckpt"


def read_rgb_images(image_paths, size, normalize=False):
    import cv2

    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image.shape[2] == 4:
            alpha = image[:, :, 3]
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            image[np.where(alpha == 0)] = [255, 255, 255]
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        images.append(image)

    images = np.stack(images, axis=0)
    if normalize:
        images = images.astype(np.float32) / 255.0
    return images


class CLIP(nn.Module):
    def __init__(
        self,
        device,
        clip_name="./clip_ckpt/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        size=224,
    ):
        super().__init__()
        from transformers import CLIPModel

        self.size = size
        self.device = f"cuda:{device}"
        self.clip_model = CLIPModel.from_pretrained(clip_name).to(self.device)
        self.to_tensor = transforms.ToTensor()
        self.aug = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    @torch.no_grad()
    def score_gt(self, ref_img_path, novel_views):
        clip_scores = []
        for novel_view in novel_views:
            clip_scores.append(self.score_from_path(ref_img_path, [novel_view]))
        return np.mean(clip_scores)

    def similarity(self, image1_features, image2_features):
        with torch.no_grad(), torch.cuda.amp.autocast():
            image1_features = image1_features.T.view(
                image1_features.T.shape[1], image1_features.T.shape[0]
            )
            similarity = torch.matmul(image1_features, image2_features.T)
            return similarity[0][0].item()

    def get_img_embeds(self, image):
        if image.shape[0] == 4:
            image = image[:3, :, :]
        image = self.aug(image).to(self.device).unsqueeze(0)
        image_features = self.clip_model.get_image_features(image)
        return image_features / image_features.norm(dim=-1, keepdim=True)

    def score_from_feature(self, image1, image2):
        image1_features = self.get_img_embeds(image1)
        image2_features = self.get_img_embeds(image2)
        return self.similarity(image1_features, image2_features)

    def score_from_path(self, image1_paths, image2_paths):
        image1 = np.squeeze(read_rgb_images(image1_paths, self.size))
        image2 = np.squeeze(read_rgb_images(image2_paths, self.size))
        image1 = self.to_tensor(image1)
        image2 = self.to_tensor(image2)
        return self.score_from_feature(image1, image2)


def numpy_to_torch(images):
    images = images * 2.0 - 1.0
    images = torch.from_numpy(images.transpose((0, 3, 1, 2))).float()
    return images.cuda()


class LPIPSMeter:
    def __init__(self, net="alex", device=None, size=224):
        import lpips

        self.size = size
        self.net = net
        self.results = []
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def measure(self):
        return np.mean(self.results)

    @torch.no_grad()
    def score_gt(self, ref_paths, novel_paths):
        self.results = []
        for ref_path, novel_path in zip(ref_paths, novel_paths):
            ref_image = read_rgb_images([ref_path], self.size, normalize=True)
            novel_image = read_rgb_images([novel_path], self.size, normalize=True)
            ref_image = numpy_to_torch(ref_image)
            novel_image = numpy_to_torch(novel_image)
            ref_image = F.interpolate(
                ref_image, size=(self.size, self.size), mode="area"
            )
            novel_image = F.interpolate(
                novel_image, size=(self.size, self.size), mode="area"
            )
            self.results.append(self.fn.forward(ref_image, novel_image).cpu().numpy())
        return self.measure()


class PSNRMeter:
    def __init__(self, size=800):
        self.results = []
        self.size = size

    def update(self, predictions, references):
        from skimage.metrics import peak_signal_noise_ratio as compare_psnr

        self.results = [
            compare_psnr(prediction, reference, data_range=1.0)
            for prediction, reference in zip(predictions, references)
        ]

    def measure(self):
        return np.mean(self.results)

    def score_gt(self, ref_paths, novel_paths):
        references = read_rgb_images(ref_paths, self.size, normalize=True)
        novel_views = read_rgb_images(novel_paths, self.size, normalize=True)
        self.update(references, novel_views)
        return self.measure()


class PyIQAMeter:
    def __init__(self, metric_name, device):
        import pyiqa

        self.device = (
            torch.device(f"cuda:{device}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.metric = self._create_metric(pyiqa, metric_name).to(self.device)

    @staticmethod
    def _checkpoint(relative_path):
        checkpoint = METRIC_CHECKPOINT_ROOT / relative_path
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Metric checkpoint not found: {checkpoint}")
        return checkpoint

    @classmethod
    def _create_metric(cls, pyiqa, metric_name):
        if metric_name == "clipiqa":
            rn50_checkpoint = cls._checkpoint("RN50.pt")
            return pyiqa.create_metric(
                metric_name, backbone=str(rn50_checkpoint)
            )

        if metric_name != "maniqa":
            return pyiqa.create_metric(metric_name)

        import pyiqa.archs.maniqa_arch as maniqa_arch
        import timm

        vit_checkpoint = cls._checkpoint(
            "vit_base_patch8_224.augreg2_in21k_ft_in1k/model.safetensors"
        )
        maniqa_checkpoint = cls._checkpoint("ckpt_koniq10k.pt")
        original_create_model = timm.create_model
        original_maniqa_checkpoint = maniqa_arch.default_model_urls["koniq"]

        def create_model_with_local_vit(*args, **kwargs):
            model_name = kwargs.get("model_name", args[0] if args else None)
            if model_name == "vit_base_patch8_224" and kwargs.get(
                "pretrained", False
            ):
                pretrained_config = dict(
                    kwargs.get("pretrained_cfg_overlay") or {}
                )
                pretrained_config["file"] = str(vit_checkpoint)
                kwargs["pretrained_cfg_overlay"] = pretrained_config
            return original_create_model(*args, **kwargs)

        timm.create_model = create_model_with_local_vit
        maniqa_arch.default_model_urls["koniq"] = str(maniqa_checkpoint)
        try:
            return pyiqa.create_metric(metric_name)
        finally:
            timm.create_model = original_create_model
            maniqa_arch.default_model_urls["koniq"] = original_maniqa_checkpoint

    def score_gt(self, ref_paths, novel_paths):
        if len(novel_paths) != NUM_VIEWS:
            raise ValueError(
                f"Expected {NUM_VIEWS} rendered views, got {len(novel_paths)}"
            )

        score_sum = 0.0
        for image_path in novel_paths:
            image = (
                Image.open(image_path)
                .convert("RGB")
                .resize((IQA_IMAGE_SIZE, IQA_IMAGE_SIZE))
            )
            image = TF.to_tensor(image).unsqueeze(0).to(device=self.device)
            score_sum += self.metric(image)
        return float(score_sum / NUM_VIEWS)


def split_from_output(result_folder, train_iter=5000):
    """Split each RGB triptych into rendered image, normal, and mask files."""
    import cv2

    images_folder = os.path.join(result_folder, "images")
    normals_folder = os.path.join(result_folder, "normals")
    masks_folder = os.path.join(result_folder, "masks")
    outputs_dir = os.path.join(result_folder, f"it{train_iter}-test")

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(normals_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)

    for output_name in sorted(os.listdir(outputs_dir)):
        output_path = os.path.join(outputs_dir, output_name)
        output_image = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
        height, width = output_image.shape[:2]
        scale = int(width // height)
        section_width = int(width // scale)
        image = output_image[:height, :section_width, :]
        normal = output_image[:height, section_width : section_width * 2, :]
        mask = output_image[:height, section_width * 2 : section_width * 3, :]
        cv2.imwrite(os.path.join(images_folder, output_name), image)
        cv2.imwrite(os.path.join(normals_folder, output_name), normal)
        cv2.imwrite(os.path.join(masks_folder, output_name), mask)

    return images_folder, normals_folder, masks_folder


def score_from_method_for_dataset_ours(
    scorer,
    input_path,
    pred_path,
    score_type="clip",
    result_folder="save",
    train_iter=5000,
):
    scores = {}
    final_result = 0
    total = 0
    examples = sorted(os.listdir(input_path))

    for example in examples:
        if example.startswith("."):
            continue

        ref_path = osp(input_path, example, f"{example}_rgba.png")
        example_result_subdir = None
        for result_subdir in os.listdir(pred_path):
            if result_subdir.startswith(example):
                example_result_subdir = result_subdir
                break
        if example_result_subdir is None:
            continue

        example_result_dir = os.path.join(
            pred_path, example_result_subdir, result_folder
        )
        if score_type in PYIQA_METRICS:
            images_folder = os.path.join(example_result_dir, "images")
            if not os.path.isdir(images_folder):
                images_folder, _, _ = split_from_output(
                    result_folder=example_result_dir, train_iter=train_iter
                )
        else:
            images_folder, _, _ = split_from_output(
                result_folder=example_result_dir, train_iter=train_iter
            )

        if score_type == "clip":
            novel_views = glob.glob(osp(images_folder, "*"))
            print(
                f"[INFO] clip loss for example {example} between 1 GT "
                f"and {len(novel_views)} predictions"
            )
        elif score_type in PYIQA_METRICS:
            novel_views = [
                os.path.join(images_folder, f"{view_index}.png")
                for view_index in range(NUM_VIEWS)
            ]
            print(
                f"[INFO] {score_type} score for example {example} over "
                f"{NUM_VIEWS} predictions"
            )
        else:
            novel_views = [os.path.join(images_folder, "0.png")]
            print(
                f"[INFO] {score_type} loss for example {example} between "
                f"{ref_path} and {novel_views}"
            )

        score = scorer.score_gt([ref_path], novel_views)
        scores[example] = score
        final_result += score
        total += 1

    scores["average"] = final_result / total
    return scores


def merge_metric_results(csv_path, results):
    rows = {}
    if os.path.isfile(csv_path):
        with open(csv_path, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames:
                scene_field = reader.fieldnames[0]
                for row in reader:
                    scene = row.pop(scene_field)
                    rows[scene] = {
                        metric_name: row.get(metric_name, "")
                        for metric_name in DEFAULT_METRICS
                    }

    valid_scenes = {
        scene for metric_scores in results.values() for scene in metric_scores
    }
    rows = {scene: row for scene, row in rows.items() if scene in valid_scenes}

    for metric_name, metric_scores in results.items():
        for row in rows.values():
            row[metric_name] = ""
        for scene, score in metric_scores.items():
            rows.setdefault(scene, {})[metric_name] = str(float(score))

    rows = {
        scene: row
        for scene, row in rows.items()
        if any(row.get(metric_name, "") for metric_name in DEFAULT_METRICS)
    }

    scene_names = sorted(scene for scene in rows if scene != "average")
    if "average" in rows:
        scene_names.append("average")

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=("scene", *DEFAULT_METRICS))
        writer.writeheader()
        for scene in scene_names:
            writer.writerow({"scene": scene, **rows[scene]})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate rendered images and merge selected metrics into one CSV.",
    )
    parser.add_argument(
        "--input-path",
        "--input_path",
        dest="input_path",
        default=DEFAULT_INPUT_PATH,
        help="Root directory containing the input datasets.",
    )
    parser.add_argument(
        "--pred-path",
        "--pred_pattern",
        dest="pred_path",
        required=True,
        help="Concrete experiment stage directory containing per-scene results.",
    )
    parser.add_argument(
        "--results-folder",
        "--results_folder",
        dest="results_folder",
        default="save",
        help="Result directory beneath each scene directory.",
    )
    parser.add_argument(
        "--datasets",
        default=DEFAULT_DATASETS,
        nargs="*",
        help="Dataset directory names to evaluate.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=DEFAULT_METRICS,
        required=True,
        help="Metrics to evaluate.",
    )
    parser.add_argument(
        "--device", type=int, default=0, help="CUDA device used for evaluation."
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=5000,
        help="Training iteration used in the test output directory name.",
    )
    parser.add_argument(
        "--save-dir", "--save_dir", dest="save_dir", default="all_metrics/results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    for metric_name in args.metrics:
        if metric_name == "clip":
            scorer = CLIP(args.device)
        elif metric_name == "lpips":
            scorer = LPIPSMeter()
        elif metric_name == "psnr":
            scorer = PSNRMeter()
        elif metric_name in PYIQA_METRICS:
            scorer = PyIQAMeter(metric_name, args.device)

        for dataset in args.datasets:
            input_path = osp(args.input_path, dataset)
            metric_scores = score_from_method_for_dataset_ours(
                scorer,
                input_path,
                args.pred_path,
                metric_name,
                result_folder=args.results_folder,
                train_iter=args.iter,
            )
            normalized_pred_path = os.path.normpath(args.pred_path)
            method = os.path.basename(os.path.dirname(normalized_pred_path))
            if not method:
                method = os.path.basename(normalized_pred_path)
            results_name = "_".join(args.results_folder.split("/"))
            csv_path = f"{args.save_dir}/{method}-{results_name}-{dataset}.csv"
            merge_metric_results(csv_path, {metric_name: metric_scores})
            print(f"updated metric: {metric_name}")
            print(f"results: {csv_path}")

        del scorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
