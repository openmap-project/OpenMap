import torch
from copy import deepcopy
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from collections import Counter
import sys
import open_clip

import torch

sys.path.append("..")
from dataset.constants import SCANNET_LABELS, SCANNET_IDS


def get_repre_feature(features, cfg):
    if cfg.use_mean:
        repre_feature = np.mean(features, axis=0)
        return repre_feature
    if cfg.use_dbscan:
        if len(features) > 1:
            db = DBSCAN(eps=cfg.dbscan_eps, min_samples=2, metric="cosine").fit(
                features
            )
            labels = db.labels_

            
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            if cfg.debug:
                print(f"Instance has {len(unique_labels)} clusters with sizes {counts}")
            if len(unique_labels) > 0:
                max_label = unique_labels[np.argmax(counts)]
                cluster_indices = np.where(labels == max_label)[0]
                repre_feature = np.mean(features[cluster_indices], axis=0)
                repre_feature = F.normalize(
                    torch.from_numpy(repre_feature).float(), dim=-1
                ).numpy()
                # ind = np.random.choice(len(cluster_indices), 1)[0]
                # instance_feature = instance_features[cluster_indices[ind]]
            else:
                repre_feature = np.mean(features, axis=0)
                repre_feature = F.normalize(
                    torch.from_numpy(repre_feature).float(), dim=-1
                ).numpy()
        else:
            repre_feature = features[0]
        return repre_feature


@torch.no_grad()
def predict_labels(features, target_labels_features, target_labels):
    features = F.normalize(features, dim=-1).cuda()
    target_labels_features_tensor = F.normalize(target_labels_features, dim=-1).cuda()
    pred_scores = torch.mm(features, target_labels_features_tensor.T)
    pred_confidences = torch.max(pred_scores, dim=1)[0]
    pred_labels = [target_labels[i] for i in torch.argmax(pred_scores, dim=1)]
    return pred_labels, pred_confidences


def get_binary_mask(points_num, point_ids):
    binary_mask = np.zeros(points_num, dtype=bool)
    binary_mask[point_ids] = True
    return binary_mask


def construct_instances(
    object_dict,
    mask_features,
    scene_points,
    target_label_features,
    target_labels,
    target_ids,
):
    point_id_list = []
    center_list = []
    feature_list = []

    for object in object_dict.values():
        repre_mask_list = object["repre_mask_list"]
        object_features_list = [
            mask_features[f"{mask_info[0]}_{mask_info[1]}"]
            for mask_info in repre_mask_list
        ]
        feature = torch.stack(object_features_list).cuda()
        object_feature = torch.mean(feature, dim=0)
        feature_list.append(object_feature)
        point_id_list.append(object["point_ids"])
    feature_tensor = torch.stack(feature_list)
    label_features = torch.stack(list(target_label_features.values())).cuda()
    label_list, score_list = predict_labels(
        feature_tensor, label_features, target_labels
    )
    label2id = get_label_id_dict(target_labels, target_ids)[0]
    label_id_list = [label2id[label] for label in label_list]
    for point_ids in point_id_list:
        center = np.mean(scene_points[point_ids], axis=0)
        center_list.append(center)
    instances = {
        "point_ids": point_id_list,
        "center": np.stack(center_list),
        "label": np.array(label_list),
        "label_id": np.array(label_id_list),
        "score": score_list,
        "feature": feature_tensor,
    }
    return instances


def desolve_instances(instances):
    return [
        {
            "point_ids": point_ids,
            "center": center,
            "label": label,
            "label_id": label_id,
            "score": score,
            "feature": feature,
        }
        for point_ids, center, label, label_id, score, feature in zip(
            instances["point_ids"],
            instances["center"],
            instances["label"],
            instances["label_id"],
            instances["score"],
            instances["feature"],
        )
    ]


def get_semantic_info(
    object_dict,
    scene_points_num,
    mask_features,
    target_label_features,
    target_labels,
    label2id,
    cfg,
):
    pred_binary_masks = []
    repre_features = []

    for i, object in object_dict.items():
        binary_mask = np.zeros(scene_points_num, dtype=bool)
        binary_mask[object["point_ids"]] = True
        pred_binary_masks.append(binary_mask)

        repre_mask_list = object["repre_mask_list"]
        object_features_list = [
            mask_features[f"{mask_info[0]}_{mask_info[1]}"]
            for mask_info in repre_mask_list
        ]
        feature = torch.stack(object_features_list)
        object_feature = torch.mean(feature, dim=0)
        repre_features.append(object_feature)
    repre_features = torch.stack(repre_features)
    label_features = torch.stack(list(target_label_features.values())).cuda()
    pred_labels, pred_confidences = predict_labels(
        repre_features, label_features, target_labels
    )
    pred_binary_masks = np.stack(pred_binary_masks, axis=1)
    pred_classes = [label2id[label] for label in pred_labels]
    pred_semantic_dict = {
        "pred_masks": pred_binary_masks,
        "pred_labels": pred_labels,
        "pred_score": pred_confidences.cpu().numpy(),
        "pred_classes": pred_classes,
    }
    return pred_semantic_dict


def get_label_id_dict(target_labels, target_ids):
    label2id = {}
    id2label = {}
    for label, id in zip(target_labels, target_ids):
        label2id[label] = id
        id2label[id] = label
    return label2id, id2label


def load_clip():
    print(f"[INFO] loading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k"
    )
    model = model.cuda()
    model.eval()
    print(f"[INFO]", " finish loading CLIP model...")
    return model, preprocess
