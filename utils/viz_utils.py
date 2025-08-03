import random
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pyviz3d.visualizer as viz
import time
import torch
from dataset.constants import (
    MATTERPORT_COLORMAP,
    MATTERPORT_LABELS,
    SCANNET_COLOR_MAP_200,
    MATTERPORT_IDS,
    SCANNET_IDS,
    MATTERPORT_COLORS,
)
import torch.nn.functional as F
import torch


class VizUtils:
    def __init__(self):
        self.point_size = 15
        pass

    def visualize_instance_mesh_scannet(
        self, mesh, instances, save_path=None, zoom=0.6, lookat=None 
    ):
        color_map = []
        for hex_color in MATTERPORT_COLORMAP:
            hex_color = hex_color.lstrip("#")
            color_map.append([int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)])
        vertex_colors = np.array(mesh.vertex_colors)
        vertex_colors = np.full(vertex_colors.shape, 0.9)
        for instance in instances:
            instance_points_ids = instance["point_ids"]
            instance_label_id = instance["label_id"]
            instance_color = (
                random.choice(color_map[2:])
                if instance_label_id in SCANNET_IDS
                else color_map[1]
            )
            vertex_colors[instance_points_ids] = instance_color
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        vis.add_geometry(mesh)
        ctr = vis.get_view_control()
        if lookat is not None:
            ctr.set_lookat(lookat)  

        ctr.set_front([0, 0, 1])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(zoom) 
        vis.poll_events()
        vis.update_renderer()
        if save_path is not None:
            vis.capture_screen_image(save_path)
        vis.run()
        vis.destroy_window()

    def visualize_instance_mesh_matterport3d(
        self,
        mesh,
        instances,
        save_path=None,
        zoom=0.6,
        lookat=None,
    ):
        color_map = []
        for hex_color in MATTERPORT_COLORMAP:
            hex_color = hex_color.lstrip("#")
            color_map.append([int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)])

        custom_colors = {
            "floor": color_map[0],
            "wall": color_map[1],
            "refridgerator": color_map[0],
            "kitchen island": [176 / 255.0, 181 / 255.0, 194 / 255.0],
        }

        vertex_colors = vertex_colors = np.full((len(mesh.vertices), 3), 0.9)
        for instance in instances:
            instance_points_ids = instance["point_ids"]
            instance_label_id = instance["label_id"]
            # if instance["label"] in MATTERPORT_LABELS:
            #     instance_color = color_map[instance["label"]]
            # else:
            #     print(instance["label"])
            #     instance_color = [0.9, 0.9, 0.9]
            if instance["label"] in custom_colors:
                instance_color = custom_colors[instance["label"]]
            else:
                instance_color = (
                    random.choice(color_map[2:])
                    if instance_label_id in MATTERPORT_IDS
                    else [0.9, 0.9, 0.9]
                )
            vertex_colors[instance_points_ids] = instance_color
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        vis.add_geometry(mesh)
        ctr = vis.get_view_control()
        ctr.set_constant_z_near(15)  
        ctr.set_constant_z_far(50)
        if lookat is not None:
            ctr.set_lookat(lookat)  
        ctr.set_front([0, 0, 1])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(zoom)  
        vis.poll_events()
        vis.update_renderer()
        if save_path is not None:
            vis.capture_screen_image(save_path)
        vis.run()
        vis.destroy_window()

    def visualize_point_cloud_categories(
        self,
        scene_points: np.ndarray,
        instance_points_ids: list,
        save_dir: str = f"data/vis/",
        categories: str = None,
        positions=None,
        mesh_file=None,
    ):

        scene_points = np.array(scene_points)
        v = viz.Visualizer()

        colors = (np.random.rand(len(instance_points_ids), 3) * 0.7 + 0.3) * 255

        scene_points_colors = np.full((len(scene_points), 3), 220, dtype=np.uint8)

        for idx, instance_points in enumerate(instance_points_ids):
            scene_points_colors[instance_points] = colors[idx]
        if categories is not None:
            v.add_labels("Labels", categories, positions, colors=colors, visible=False)
        v.add_points(
            "rgb",
            scene_points,
            colors=scene_points_colors,
            point_size=self.point_size,
            visible=False,
        )

        labeled_scene_points_mask = np.where(np.sum(scene_points_colors, axis=1) != 30)
        v.add_points(
            "Instances",
            scene_points[labeled_scene_points_mask],
            scene_points_colors[labeled_scene_points_mask],
            visible=True,
            point_size=self.point_size,
        )
        if mesh_file is not None:
            v.add_mesh(mesh_file, color=(0.5, 0.5, 0.5), opacity=0.5, visible=False)
        v.save(save_dir)


@torch.no_grad()
def vis_heatmap(mesh, instances, save_path=None):
    vertex_colors = np.full((len(mesh.vertices), 3), 0.9)
    scores = instances["score"]
    # 对score 进行softmax处理
    scores = F.softmax(scores, dim=1).detach().cpu().numpy()
    scores = scores.reshape(-1)
    if scores.size > 0:
        min_score, max_score = scores.min(), scores.max()
        if max_score > min_score:
            scores = (scores - min_score) / (max_score - min_score)
    print(scores)
    for idx, point_ids in enumerate(instances["point_ids"]):
        heat_color = np.array([1.0, 1.0 - scores[idx], 1.0 - scores[idx]])
        vertex_colors[point_ids] = heat_color
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.add_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    if save_path is not None:
        vis.capture_screen_image(save_path)
    vis.run()
    vis.destroy_window()