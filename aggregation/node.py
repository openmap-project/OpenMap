import torch
import open3d as o3d
import numpy as np

class Node:
    
    def __init__(self, mask_list, visible_frame, contained_mask, point_ids, repre_mask, points_max_num):
        
        self.mask_list = mask_list # list of masks that is within this cluster,like  [(frame_id, mask_id)]
        self.visible_frame = visible_frame # one-hot vector, 1 if the node appears in the frame
        self.contained_mask = contained_mask # one-hot vector, 1 if the node is contained by the mask
        self.point_ids = point_ids # the corresponding 3D point ids
        self.repre_mask = repre_mask
        self.points_num_max = points_max_num



    @ staticmethod
    def create_node_from_list(node_list):
        mask_list = []
        visible_frame = torch.zeros(len(node_list[0].visible_frame), dtype=bool).cuda()
        contained_mask = torch.zeros(len(node_list[0].contained_mask), dtype=bool).cuda()
        point_ids = set()
        points_num_mask = np.array([node.points_num_max for node in node_list])
        for node in node_list:
            mask_list += node.mask_list
            visible_frame = visible_frame | (node.visible_frame).bool()
            contained_mask = contained_mask | (node.contained_mask).bool()
            point_ids = point_ids.union(node.point_ids)
        repre_node = node_list[np.argmax(points_num_mask)]
        points_num_max = repre_node.points_num_max
        repre_mask = repre_node.repre_mask

        return Node(mask_list, visible_frame.float(), contained_mask.float(), point_ids, repre_mask, points_num_max)
    
    def get_point_cloud(self, scene_points):
        '''
            return:
                pcld: open3d.geometry.PointCloud object, the point cloud of the node
                point_ids: list of int, the corresponding 3D point ids of the node
        '''
        point_ids = list(self.point_ids)
        points = scene_points[point_ids]
        pcld = o3d.geometry.PointCloud()
        pcld.points = o3d.utility.Vector3dVector(points)
        return pcld, point_ids
