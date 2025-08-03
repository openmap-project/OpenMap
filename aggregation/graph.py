import torch
import numpy as np
from tqdm import tqdm
from aggregation.node import Node
import networkx as nx
from aggregation.data import Data
from utils.post_process import post_process
from pathlib import Path


# To build a graph
class Graph:
    def __init__(self, cfg, frame_mask_feature_list, dataset):
        with torch.no_grad():
            scene_points = dataset.get_scene_points()
            frame_list = dataset.get_frame_list(cfg.dataset.step)
            self.data = Data(cfg, frame_mask_feature_list, frame_list, dataset)
            if cfg.debug:
                print("start building point in mask matrix")
            self.get_observer_num_thresholds()
            self.frame_mask_feature_list = frame_mask_feature_list

            self.init_nodes(
                self.data.global_frame_mask_list,
                self.data.visible_frames,
                self.data.contained_masks,
                self.data.undersegment_mask_ids,
                self.data.mask_point_clouds,
            )
            del (
                self.data.point_in_mask_matrix,
                self.data.visible_frames,
                self.data.contained_masks,
            )
            self.iterative_clustering(cfg.view_consensus_threshold, cfg.debug)
            self.object_dict = post_process(
                self.nodes,
                self.data.mask_point_clouds,
                scene_points,
                self.data.point_frame_matrix,
                frame_list,
                cfg,
                dataset,
            )
            del self.data

    def init_nodes(
        self,
        global_frame_mask_list,
        mask_project_on_all_frames,
        contained_masks,
        undersegment_mask_ids,
        mask_point_clouds,
    ):
        self.nodes = []
        # construct a node for each mask
        for global_mask_id, (frame_id, mask_id) in enumerate(global_frame_mask_list):
            # skip undersegmented masks
            if global_mask_id in undersegment_mask_ids:
                continue
            mask_list = [(frame_id, mask_id)]
            repre_mask = (frame_id, mask_id)
            points_num_max = len(mask_point_clouds[f"{frame_id}_{mask_id}"])

            frame = mask_project_on_all_frames[global_mask_id]
            frame_mask = contained_masks[global_mask_id]
            point_ids = mask_point_clouds[f"{frame_id}_{mask_id}"]

            node = Node(
                mask_list, frame, frame_mask, point_ids, repre_mask, points_num_max
            )
            self.nodes.append(node)

    def get_observer_num_thresholds(self):
        """
        compute the threshold of observer number
        """
        # print(self.data.visible_frames.dtype)
        observer_num_matrix = torch.matmul(
            self.data.visible_frames, self.data.visible_frames.transpose(0, 1)
        )  # observer_num_matrix[i,j] represent the number of frames that two masks i and j can see
        observer_num_list = observer_num_matrix.flatten()
        observer_num_list = (
            observer_num_list[observer_num_list > 0].cpu().numpy()
        )  # get the list of observer numbers
        self.observer_num_thresholds = []
        for percentile in range(95, -5, -5):
            observer_num = np.percentile(
                observer_num_list, percentile
            )  
            if observer_num <= 1:
                if percentile < 50:
                    break
                else:
                    observer_num = 1
            self.observer_num_thresholds.append(observer_num)

    def cluster_into_new_nodes(self, iteration, graph):
        new_nodes = []
        for component in nx.connected_components(graph): 
            new_nodes.append(
                Node.create_node_from_list([self.nodes[node] for node in component])
            )  # construct a new node with the nodes in the connected component
        del self.nodes
        self.nodes = new_nodes

    def update_graph(self, observer_num_threshold, connect_threshold):
        # 1. transfer node data to sparse format
        visible_sparse = torch.stack(
            [node.visible_frame for node in self.nodes], dim=0
        ).to_sparse()
        contained_sparse = torch.stack(
            [node.contained_mask for node in self.nodes], dim=0
        ).to_sparse()

        # 2. compute observer_nums and supporter_nums by sparse matrix multiplication
        observer_nums = torch.sparse.mm(
            visible_sparse, visible_sparse.transpose(0, 1)
        ).to_dense()
        supporter_nums = torch.sparse.mm(
            contained_sparse, contained_sparse.transpose(0, 1)
        ).to_dense()

        # 3. compute view consensus rate (keep dense format, because it is used later)
        view_concensus_rate = supporter_nums / (observer_nums + 1e-7)

        # 4. construct a boolean matrix indicating the disconnected areas
        disconnect = torch.eye(len(self.nodes), dtype=bool).cuda()
        disconnect = disconnect | (observer_nums < observer_num_threshold)

        possible_connect_area = torch.logical_and(
            observer_nums >= 1, supporter_nums >= 1
        )
        possible_nodes_pairs = torch.where(
            torch.triu(possible_connect_area, diagonal=1)
        )  
        node_features = torch.stack([
                self.frame_mask_feature_list[node.repre_mask[0]][node.repre_mask[1] - 1]
                for node in self.nodes
            ]
        ).cuda()
            

        if len(possible_nodes_pairs[0]) > 0:
            features_i = node_features[possible_nodes_pairs[0].cpu()].cuda()
            features_j = node_features[possible_nodes_pairs[1].cpu()].cuda()
            cos_sims = torch.nn.functional.cosine_similarity(
                features_i, features_j, dim=-1
            )
            assert len(cos_sims) == len(possible_nodes_pairs[0])
            indices = torch.stack([possible_nodes_pairs[0], possible_nodes_pairs[1]])
            cos_sim_sparse = torch.sparse_coo_tensor(
                indices, cos_sims, size=(len(self.nodes), len(self.nodes))
            ).to_dense() 
            cos_sim_matrix = cos_sim_sparse + cos_sim_sparse.t() 
        else:
            cos_sim_matrix = torch.zeros_like(observer_nums)

        A = (cos_sim_matrix * view_concensus_rate) > connect_threshold
        A = A & ~disconnect
        A = A.cpu().numpy()

        G = nx.from_numpy_array(A)
        return G


    def iterative_clustering(self, connect_threshold, debug):
        if debug:
            print("====> Start iterative clustering")
        for iterate_id, observer_num_threshold in enumerate(
            self.observer_num_thresholds
        ):
            if debug:
                print(
                    f"Iterate {iterate_id}: observer_num",
                    observer_num_threshold,
                    ", number of nodes",
                    len(self.nodes),
                )
            graph = self.update_graph(observer_num_threshold, connect_threshold)
            self.cluster_into_new_nodes(iterate_id + 1, graph)