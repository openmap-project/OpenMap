import json
import pandas as pd
import numpy as np
import os
from plyfile import PlyData
import torch
from dataset.constants import (
    HEAD_CATS_SCANNET_200,
    COMMON_CATS_SCANNET_200,
    TAIL_CATS_SCANNET_200,
    SCANNET_LABELS,
    SCANNET_IDS,
    MATTERPORT_LABELS,
    MATTERPORT_IDS,
)
from dataset.preprocess.matterport3d.constants import RAW_TO_NYU, MATTERPORT_VALID_IDS
from utils.ovutils import get_label_id_dict
from utils.viz_utils import VizUtils
import open3d as o3d



def read_csv(file_path):
    df = pd.read_csv(file_path)

    head_data = df[df["class"].isin(HEAD_CATS_SCANNET_200)]
    common_data = df[df["class"].isin(COMMON_CATS_SCANNET_200)]
    tail_data = df[df["class"].isin(TAIL_CATS_SCANNET_200)]
    return head_data, common_data, tail_data


def get_gt_instances(gt_path_txt):
    label2id, id2label = get_label_id_dict(SCANNET_LABELS, SCANNET_IDS)
    gt_data = np.loadtxt(gt_path_txt)
    unique_ids = np.unique(gt_data)
    gt_instances = []

    for id in unique_ids:
        if id == 0:
            continue
        label_id = id // 1000
        if label_id == 0:
            label_id = 1

        label = id2label[label_id] if label_id in label2id.keys() else "void"
        instance_id = id % 1000
        point_ids = np.where(gt_data == id)[0]
        gt_instances.append(
            {
                "label": label,
                "instance_id": instance_id,
                "point_ids": point_ids,
                "label_id": label_id,
            }
        )
    return gt_instances


def load_instance_json(filepath):
    with open(filepath, "r") as f:
        info = json.load(f)
    instances = info["segGroups"]
    instance_segments_id = []
    for instance in instances:
        instance_segments_id.append(np.array(instance["segments"]))
    return instance_segments_id


def load_face_json(filepath):
    with open(filepath, "r") as f:
        info = json.load(f)
    return np.array(info["segIndices"])


def load_ply(filepath):
    with open(filepath, "rb") as f:
        plydata = PlyData.read(f)
    vdata = plydata.elements[0].data  # vertex
    coords = np.array([vdata["x"], vdata["y"], vdata["z"]], dtype=np.float32).T

    fdata = plydata["face"].data
    faces = np.stack(fdata["vertex_indices"])

    face_semantic_id = np.array(fdata["category_id"], dtype=np.int32)
    vert_semantic_id = np.zeros(coords.shape[0], dtype=np.int32)
    vert_semantic_id[faces.reshape(-1)] = (
        face_semantic_id[None].repeat(3, axis=1).reshape(-1)
    )

    return vert_semantic_id, faces


def get_gt_matterport3d(scene_path):
    seq_name = scene_path.split("/")[-1]
    vert_semantic_id, faces = load_ply(
        os.path.join(scene_path, "house_segmentations", f"{seq_name}.ply")
    )

    face_segment_id = load_face_json(
        os.path.join(scene_path, "house_segmentations", f"{seq_name}.fsegs.json")
    )
    vert_segment_id = np.zeros_like(vert_semantic_id)
    vert_segment_id[faces.reshape(-1)] = (
        face_segment_id[None].repeat(3, axis=1).reshape(-1)
    )

    segment_ids = np.unique(vert_segment_id)
    instance_segments_id = load_instance_json(
        os.path.join(scene_path, "house_segmentations", f"{seq_name}.semseg.json")
    )
    segment_instance_id = np.full(segment_ids.max() + 1, -1)
    for instance_id, segments_id in enumerate(instance_segments_id):
        segment_instance_id[segments_id] = instance_id

    vert_instance_id = segment_instance_id[vert_segment_id]

    assert vert_instance_id.shape == vert_segment_id.shape
    assert vert_instance_id.min() >= 0 and vert_instance_id.max() <= len(
        instance_segments_id
    )

    vert_semantic_id[vert_semantic_id < 0] = 0
    vert_semantic_id = RAW_TO_NYU[vert_semantic_id]
    vert_semantic_id[np.isin(vert_semantic_id, MATTERPORT_VALID_IDS, invert=True)] = 0

    unique_ids = np.unique(vert_instance_id)
    id2label = get_label_id_dict(MATTERPORT_LABELS, MATTERPORT_IDS)[1]
    gt_instances = []
    for unique_id in unique_ids:
        if unique_id == -1:
            continue
        point_ids = np.where(vert_instance_id == unique_id)[0]
        label_id = vert_semantic_id[point_ids[0]]
        label = id2label[label_id] if label_id in id2label.keys() else "void"
        if label == "void":
            print(f"Invalid label {label_id} for instance {unique_id}")

        gt_instances.append(
            {
                "point_ids": point_ids,
                "label": label,
                "label_id": label_id,
            }
        )
        torch.save(gt_instances, os.path.join(scene_path, f"gt_instances.pth"))
    return gt_instances


def get_gt_scannet(scene_path, label_map_file):
    scene_id = scene_path.split("/")[-1]
    labels_pd = pd.read_csv(label_map_file, sep="\t", header=0)
    segments_file = os.path.join(
        scene_path, f"{scene_id}_vh_clean_2.0.010000.segs.json"
    )
    aggregations_file = os.path.join(scene_path, f"{scene_id}.aggregation.json")
    if not os.path.exists(segments_file) or not os.path.exists(aggregations_file):
        print(f"Missing files for {scene_id}")
        return

    with open(segments_file) as f:
        segments = json.load(f)
        seg_indices = np.array(segments["segIndices"])

    with open(aggregations_file) as f:
        aggregation = json.load(f)
        seg_groups = np.array(aggregation["segGroups"])

    instances = []
    for group in seg_groups:
        group_segments = np.array(group["segments"])
        label = group["label"]

        # Map the category name to id
        label_ids = labels_pd[labels_pd["raw_category"] == label]["id"]
        label = labels_pd[labels_pd["id"] == label_ids.iloc[0]]["category"].iloc[0]
        label_id = int(label_ids.iloc[0]) if len(label_ids) > 0 else 0

        # get points, where segment indices (points labelled with segment ids) are in the group segment list
        point_IDs = np.where(np.isin(seg_indices, group_segments))[0]
        instances.append({"point_ids": point_IDs, "label_id": label_id, "label": label})

    return instances
