import numpy as np
import torch
from tqdm import tqdm
import numpy as np
from pytorch3d.ops import ball_query
import torch
import open3d as o3d
from utils.geometry import denoise
from torch.nn.utils.rnn import pad_sequence
import pickle

COVERAGE_THRESHOLD = 0.3
DISTANCE_THRESHOLD = 0.03
FEW_POINTS_THRESHOLD = 25
DEPTH_TRUNC = 20
BBOX_EXPAND = 0.1

class Data:
    def __init__(self, args, frame_mask_feature_list, frame_list, dataset):
        # get number of points in the scene
        self.scene_points = dataset.get_scene_points()
        self.scene_points_num = len(self.scene_points)
        # get number of frames
        self.frame_num = len(frame_list)

        # convert to tensor
        self.frame_mask_feature_list = frame_mask_feature_list  
        self.frame_list = frame_list
        self.dataset = dataset
        self.cfg = args
        self.boundary_points = set()
        self.point_in_mask_matrix = np.zeros((self.scene_points_num, self.frame_num), dtype=np.uint16)
        self.point_frame_matrix = np.zeros((self.scene_points_num, self.frame_num), dtype=bool) # post-processing needs, record which frames the points appear in
        self.global_frame_mask_list = []
        self.mask_point_clouds = {}

        self.build_point_in_mask_matrix(args, self.scene_points, frame_list, dataset)
        
        self.undersegment_mask_ids = self.process_masks()
        
    def _process_one_mask(self, point_in_mask_matrix, boundary_points, mask_point_cloud, frame_list, global_frame_mask_list,  args):
        '''
            process one mask, get the visible frames and global masks that contain this mask
        '''
        visible_frame = torch.zeros(len(self.frame_list))
        contained_mask = torch.zeros(len(global_frame_mask_list))

        valid_mask_point_cloud = mask_point_cloud - boundary_points
        mask_point_cloud_info = point_in_mask_matrix[list(valid_mask_point_cloud), :]
        
        possibly_visible_frames = np.where(np.sum(mask_point_cloud_info, axis=0) > 0)[0] # only consider frames where there are points in this mask
        
        split_num = 0
        visible_num = 0
        
        for frame_id in possibly_visible_frames:
            mask_id_count = np.bincount(mask_point_cloud_info[:, frame_id]) # count the number of times each mask appears in this frame
            invisible_ratio = mask_id_count[0] / np.sum(mask_id_count) # the ratio of points not in this frame to the total number of points in this mask
            # If in a frame, most points in this mask are missing, then we think this mask is invisible in this frame.
            if 1 - invisible_ratio < args.mask_visible_threshold and (np.sum(mask_id_count) - mask_id_count[0]) < 500:
                continue
            visible_num += 1
            mask_id_count[0] = 0
            max_mask_id = np.argmax(mask_id_count) # the mask id that appears most in this frame
            contained_ratio = mask_id_count[max_mask_id] / np.sum(mask_id_count) # the points in this mask that appear in this frame / the total number of points in this mask 
            if contained_ratio > args.contained_threshold:
                visible_frame[frame_id] = 1 
                frame_mask_idx = global_frame_mask_list.index((frame_list[frame_id], max_mask_id)) 
                contained_mask[frame_mask_idx] = 1 
            else:
                split_num += 1 # This mask is splitted into two masks in this frame
        
        if visible_num == 0 or split_num / visible_num > args.undersegment_filter_threshold:
            return False, visible_frame, contained_mask
        else:
            return True, visible_frame, contained_mask

    def process_masks(self):
        '''
            for all the masks, we will determine whether it is visible, which global masks contain this mask, and also determine whether this mask is undersegmented.
        '''
        if self.cfg.debug:
            print('start processing masks')
        visible_frames = []
        contained_masks = []
        undersegment_mask_ids = []

        iterator = tqdm(self.global_frame_mask_list) if self.cfg.debug else self.global_frame_mask_list
        for frame_id, mask_id in iterator:
            valid, visible_frame, contained_mask = self._process_one_mask(self.point_in_mask_matrix, self.boundary_points, self.mask_point_clouds[f'{frame_id}_{mask_id}'], self.frame_list, self.global_frame_mask_list,  self.cfg)
            visible_frames.append(visible_frame)
            contained_masks.append(contained_mask)
            if not valid:
                global_mask_id = self.global_frame_mask_list.index((frame_id, mask_id))
                undersegment_mask_ids.append(global_mask_id) 

        self.visible_frames = torch.stack(visible_frames, dim=0).cuda() # (mask_num, frame_num)
        self.contained_masks = torch.stack(contained_masks, dim=0).cuda() # (mask_num, mask_num)

        # in case some undersegmented masks become observers, which will cause errors in merging different instances
        for global_mask_id in undersegment_mask_ids:
            frame_id, _ = self.global_frame_mask_list[global_mask_id]
            global_frame_id = self.frame_list.index(frame_id)
            mask_projected_idx = torch.where(self.contained_masks[:, global_mask_id])[0]
            self.contained_masks[:, global_mask_id] = False
            self.visible_frames[mask_projected_idx, global_frame_id] = False

        return undersegment_mask_ids
    
    
    def build_point_in_mask_matrix(self, args, scene_points, frame_list, dataset):
        '''
            to build the matrix that indicates which mask each point belongs to, if the mask i is in the kth mask of frame j, then M[i,j] = k, otherwise M[i,j] = 0
            Besides, it returns the set of boundary points, the point clouds corresponding to each mask, the matrix that indicates which points appear in which frames, and the global frame-mask list.
        '''
        scene_points = torch.tensor(scene_points).float().cuda()
        iterator = tqdm(enumerate(frame_list), total=len(frame_list)) if args.debug else enumerate(frame_list)
        for frame_cnt, frame_id in iterator:
            # using the mask to crop the global point cloud, and get the point cloud id list corresponding to each frame

            mask_dict, frame_point_cloud_ids = self.frame_backprojection(dataset, scene_points, frame_id)
            if len(frame_point_cloud_ids) == 0:
                continue
            self.point_frame_matrix[frame_point_cloud_ids, frame_cnt] = True # label the points that appear in the current frame
            appeared_point_ids = set()
            frame_boundary_point_index = set()

            for mask_id, mask_point_cloud_ids in mask_dict.items():
                # boundary points are the points that appear in different masks
                frame_boundary_point_index.update(mask_point_cloud_ids.intersection(appeared_point_ids))
                self.mask_point_clouds[f'{frame_id}_{mask_id}'] = mask_point_cloud_ids # use string index to get the mask point cloud
                self.point_in_mask_matrix[list(mask_point_cloud_ids), frame_cnt] = mask_id # label the points that appear in the current frame and the mask_id-th mask
                appeared_point_ids.update(mask_point_cloud_ids) 
                self.global_frame_mask_list.append((frame_id, mask_id))

            self.point_in_mask_matrix[list(frame_boundary_point_index), frame_cnt] = 0 
            self.boundary_points.update(frame_boundary_point_index) # update the boundary points set
    
    
    def backproject(self, depth, intrinisc_cam_parameters, extrinsics):

        depth = o3d.geometry.Image(depth) 
        pcld = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinisc_cam_parameters, depth_scale=1, depth_trunc=DEPTH_TRUNC)
        pcld.transform(extrinsics)
        return pcld


    def get_neighbor(self, valid_points, scene_points, lengths_1, lengths_2):
        # get the points around the valid_points in the scene_points
        _, neighbor_in_scene_pcld, _ = ball_query(valid_points, scene_points, lengths_1, lengths_2, K=20, radius=DISTANCE_THRESHOLD, return_nn=False)
        return neighbor_in_scene_pcld


    def get_depth_mask(self, depth):
        depth_tensor = torch.from_numpy(depth).cuda()
        depth_mask = torch.logical_and(depth_tensor > 0, depth_tensor < DEPTH_TRUNC).reshape(-1)
        return depth_mask


    def crop_scene_points(self, mask_points, scene_points):
        x_min, x_max = torch.min(mask_points[:, 0]), torch.max(mask_points[:, 0])
        y_min, y_max = torch.min(mask_points[:, 1]), torch.max(mask_points[:, 1])
        z_min, z_max = torch.min(mask_points[:, 2]), torch.max(mask_points[:, 2])

        # filter out the global points outside the mask bounding box
        selected_point_mask = (scene_points[:, 0] > x_min) & (scene_points[:, 0] < x_max) & (scene_points[:, 1] > y_min) & (scene_points[:, 1] < y_max) & (scene_points[:, 2] > z_min) & (scene_points[:, 2] < z_max)
        selected_point_ids = torch.where(selected_point_mask)[0]
        cropped_scene_points = scene_points[selected_point_ids]
        return cropped_scene_points, selected_point_ids


    def turn_mask_to_point(self, dataset, scene_points, mask_image, frame_id):
        intrinisc_cam_parameters = dataset.get_intrinsics(frame_id)
        extrinsics = dataset.get_extrinsic(frame_id)
        if np.sum(np.isinf(extrinsics)) > 0:
            return {}, [], set()

        mask_image = torch.from_numpy(mask_image).cuda().reshape(-1)
        if self.cfg.area_thresh >0:
            mask_id_count = torch.bincount(mask_image)
            mask_id_count[0] = 0
            mask_id_area_ratio = mask_id_count / mask_id_count.sum()
            ids = torch.where(mask_id_area_ratio > self.cfg.area_thresh)[0].cpu().numpy()
            ids.sort()
        else:
            ids = torch.unique(mask_image).cpu().numpy()
            ids.sort()
            
        depth = dataset.get_depth(frame_id)
        depth_mask = self.get_depth_mask(depth)

        colored_pcld = self.backproject(depth, intrinisc_cam_parameters, extrinsics) 
        view_points = np.asarray(colored_pcld.points)

        mask_points_list = []
        mask_points_num_list = []
        scene_points_list = []
        scene_points_num_list = []
        selected_point_ids_list = []
        initial_valid_mask_ids = []
        for mask_id in ids:
            if mask_id == 0:
                continue
            segmentation = mask_image == mask_id
            valid_mask = segmentation[depth_mask].cpu().numpy()

            mask_pcld = o3d.geometry.PointCloud()
            mask_points = view_points[valid_mask] 
            if len(mask_points) < FEW_POINTS_THRESHOLD:
                continue
            mask_pcld.points = o3d.utility.Vector3dVector(mask_points)

            
            mask_pcld = mask_pcld.voxel_down_sample(voxel_size=DISTANCE_THRESHOLD)
            mask_pcld, _ = denoise(mask_pcld) 
            mask_points = np.asarray(mask_pcld.points) 
            
            if len(mask_points) < FEW_POINTS_THRESHOLD:
                continue
            
            mask_points = torch.tensor(mask_points).float().cuda()
            cropped_scene_points, selected_point_ids = self.crop_scene_points(mask_points, scene_points)
            initial_valid_mask_ids.append(mask_id)
            mask_points_list.append(mask_points)
            scene_points_list.append(cropped_scene_points)
            mask_points_num_list.append(len(mask_points))
            scene_points_num_list.append(len(cropped_scene_points)) 
            selected_point_ids_list.append(selected_point_ids)

        if len(initial_valid_mask_ids) == 0:
            return {}, [], []
        mask_points_tensor = pad_sequence(mask_points_list, batch_first=True, padding_value=0)
        scene_points_tensor = pad_sequence(scene_points_list, batch_first=True, padding_value=0)

        lengths_1 = torch.tensor(mask_points_num_list).cuda()
        lengths_2 = torch.tensor(scene_points_num_list).cuda()
        
        neighbor_in_scene_pcld = self.get_neighbor(mask_points_tensor, scene_points_tensor, lengths_1, lengths_2)

        valid_mask_ids = []
        mask_info = {}
        frame_point_ids = set()

        for i, mask_id in enumerate(initial_valid_mask_ids):
            mask_neighbor = neighbor_in_scene_pcld[i] # P, 20
            mask_point_num = mask_points_num_list[i] # Pi
            mask_neighbor = mask_neighbor[:mask_point_num] # Pi, 20, only 0~mask_point_num-1 are valid neighbors in this mask

            valid_neighbor = mask_neighbor != -1 # Pi, 20
            neighbor = torch.unique(mask_neighbor[valid_neighbor]) 
            neighbor_in_complete_scene_points = selected_point_ids_list[i][neighbor].cpu().numpy() 
            coverage = torch.any(valid_neighbor, dim=1).sum().item() / mask_point_num

            if coverage < COVERAGE_THRESHOLD:
                continue 
            valid_mask_ids.append(mask_id) 
            mask_info[mask_id] = set(neighbor_in_complete_scene_points) 
            frame_point_ids.update(mask_info[mask_id]) 

        return mask_info, valid_mask_ids, list(frame_point_ids)

    
    def frame_backprojection(self, dataset, scene_points, frame_id):
        mask_image = dataset.get_segmentation(frame_id, align_with_depth=True)
        mask_info, _, frame_point_ids = self.turn_mask_to_point(dataset, scene_points, mask_image, frame_id)
        return mask_info, frame_point_ids