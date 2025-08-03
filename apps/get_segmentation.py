# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
import cv2
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import warnings
import numpy as np
from tqdm import tqdm
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config

from third_party.detectron2.projects.CropFormer.mask2former import (
    add_maskformer2_config,
)
from third_party.detectron2.projects.CropFormer.demo_cropformer.predictor import (
    VisualizationDemo,
)
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def setup_cfg(cfg):
    # load config from file and command-line arguments
    args = argparse.Namespace()
    args.config_file = cfg.config_file
    args.opts = cfg.opts
    args.root = cfg.root
    args.seq_name_list = cfg.splits.seq_name_list
    args.image_path_pattern = cfg.image_path_pattern
    args.confidences_threshold = cfg.confidence_threshold
    args.dataset = cfg.dataset

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

# python -m apps.get_segmentation 
@hydra.main(
    config_path="../conf", config_name="get_segmentation.yaml", version_base=None
)
def get_segmentations(args: DictConfig):
    mp.set_start_method("spawn", force=True)
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    seq_name_list = args.splits.seq_name_list
    for i, seq_name in tqdm(enumerate(seq_name_list), total=len(seq_name_list)):
        print(f"Processing sequence {i+1}/{len(seq_name_list)}: {seq_name}")
        print(f"Image path pattern: {args.image_path_pattern}")
        seq_dir = os.path.join(args.root, seq_name)
        output_dir = os.path.join(args.output_dir, f'{seq_name}/mask')

        print(f"Sequence directory: {seq_dir}")
        image_list = sorted(glob.glob(os.path.join(seq_dir, args.image_path_pattern)))
        print(f"Number of images: {len(image_list)}")

        print(f"Output directory: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)

        iter_list = tqdm(image_list)
        for path in iter_list:
            img = read_image(path, format="BGR")
            predictions = demo.run_on_image(img)

            ##### color_mask
            pred_masks = predictions["instances"].pred_masks
            pred_scores = predictions["instances"].scores

            # select by confidence threshold
            selected_indexes = pred_scores >= args.confidence_threshold
            selected_scores = pred_scores[selected_indexes]
            selected_masks = pred_masks[selected_indexes]
            _, m_H, m_W = selected_masks.shape
            mask_image = np.zeros((m_H, m_W), dtype=np.uint8)

            # rank
            mask_id = 1
            selected_scores, ranks = torch.sort(selected_scores)
            for index in ranks:
                num_pixels = torch.sum(selected_masks[index])
                if num_pixels < 400:
                    # ignore small masks
                    continue
                mask_image[(selected_masks[index] == 1).cpu().numpy()] = mask_id
                mask_id += 1
            cv2.imwrite(
                os.path.join(output_dir, os.path.basename(path).split(".")[0] + ".png"),
                mask_image,
            )


if __name__ == "__main__":
    get_segmentations()