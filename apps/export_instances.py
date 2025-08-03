import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from utils.ovutils import construct_instances, get_semantic_info
from dataset.matterport import MatterportDataset
from dataset.scannet import ScanNetDataset
import numpy as np
import torch
from dataset.constants import (
    SCANNET_LABELS,
    SCANNET_IDS,
    SCANNET_LABELS_DESCRIPTION,
    MATTERPORT_IDS,
    MATTERPORT_LABELS,
    MATTERPORT_LABEL_DESCRIPTIONS,
)
import os


@hydra.main(
    config_path="../conf", config_name="export_instances.yaml", version_base="1.3"
)
def export_instances(cfg: DictConfig):
    target_labels = globals()[cfg.dataset.target_labels]
    target_ids = globals()[cfg.dataset.target_ids]
    for seq_name in tqdm(cfg.splits.seq_name_list):
        cfg.seq_name = seq_name
        dataset = (
            ScanNetDataset(cfg.dataset.data_root, cfg.seq_name,cfg.dataset.output_dir)
            if cfg.dataset.name == "scannet"
            else MatterportDataset(cfg.dataset.data_root, cfg.seq_name)
        )
        object_dict = np.load(dataset.object_dict_dir, allow_pickle=True).item()
        feature_dict = torch.load(dataset.clip_feature_path)
        label_features = torch.load(cfg.dataset.label_feature_save_path)
        instances = construct_instances(
            object_dict,
            feature_dict,
            dataset.get_scene_points(),
            label_features,
            target_labels,
            target_ids,
        )
        pred_semantic_dict = get_semantic_info(
            object_dict,
            dataset.get_scene_points().shape[0],
            feature_dict,
            label_features,
            target_labels,
            dataset.get_label_id()[0],
            cfg,
        )
        np.savez(f"{cfg.semantic_info_path}/{seq_name}.npz", **pred_semantic_dict)
        torch.save(instances, dataset.instance_path)

if __name__ == "__main__": 
    export_instances()