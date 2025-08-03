import open_clip
from open_clip import tokenizer
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.ovutils import load_clip
from dataset.constants import (
    SCANNET_LABELS,
    SCANNET_IDS,
    SCANNET_LABELS_DESCRIPTION,
    MATTERPORT_IDS,
    MATTERPORT_LABELS,
    MATTERPORT_LABEL_DESCRIPTIONS,
)


@hydra.main(
    config_path="../conf", config_name="get_label_feature.yaml", version_base="1.3"
)
def extract_text_feature(cfg: DictConfig):
    labels = globals()[cfg.dataset.target_labels]
    descriptions = globals()[cfg.dataset.target_label_descriptions]
    model, _ = load_clip()
    text_tokens = tokenizer.tokenize(descriptions).cuda()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features

    text_features_dict = {}
    for i, (label, description) in enumerate(zip(labels, descriptions)):
        text_features_dict[label] = text_features[i]
        assert label in description, f"{label} not in {description}"

    torch.save(text_features_dict, cfg.dataset.label_feature_save_path)


if __name__ == "__main__":
    extract_text_feature()
