import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple
import sys

from mmengine.structures import InstanceData

from third_party.ovsam.models.model import SAMSegmentor
from third_party.ovsam.utils.transforms import ResizeLongestSide

from third_party.ovsam.ext.class_names.lvis_list import LVIS_CLASSES
LVIS_NAMES = LVIS_CLASSES

sys.path.append('..')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OVSAM_Predictor:
    def __init__(self,
                ovsam:SAMSegmentor,
                ):
        self.ovsam = ovsam
        self.transform = ResizeLongestSide(self.ovsam.image_processed_size)
        self.device = device
        
    def set_image(self,img: np.ndarray,) -> None:
        # 把图片的最长边resize到模型输入大小，并保持长宽比
        input_image = self.transform.apply_image(img)
        input_image_torch = torch.as_tensor(input_image, device=self.device,dtype=torch.float32).permute((2, 0, 1))[None]
        
        assert(
            len(input_image_torch.shape)==4
            and (input_image_torch.shape[1]==3)  
            and max(*input_image_torch.shape[2:])==self.ovsam.image_processed_size
        ),f"set_torch_image input must be BCHW with long side {self.ovsam.image_processed_size}."
        # 清空原来的状态
        self.reset_image()
        
        self.original_size = img.shape[:2]
        self.input_size = tuple(input_image_torch.shape[2:])
        # 完成图片预处理（归一化和padding）
        input_image = self.ovsam.preprocess(input_image_torch)
        
        # img_tensor = torch.tensor(img_numpy, device=device, dtype=torch.float32).permute((2, 0, 1))[None]
        # img_tensor = (img_tensor - mean) / std
        # max_size = max(new_h, new_w)
        
        # img_tensor = F.pad(img_tensor, (0, max_size - new_w, 0, max_size - new_h), 'constant', 0)
        # 提取特征
        feat_dict = self.ovsam.extract_feat(input_image)
        
        self.img_feat = feat_dict
        self.is_image_set = True
    
    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        # point_labels: Optional[torch.Tensor],
        # boxes: Optional[torch.Tensor] = None,
        # mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompts = InstanceData(
            point_coords=point_coords,
        )
        
        # TODO: 考虑extract_masks 是否支持批量并行
        masks, cls_pred, sematic_embed = self.ovsam.extract_masks(self.img_feat,prompts=prompts)
        masks = self.ovsam.postprocess_masks(masks,self.input_size,self.original_size)
        
        if not return_logits:
            masks = (masks > self.ovsam.MASK_THRESHOLD)
        
        return masks, None, cls_pred, sematic_embed

    @torch.no_grad()
    def predict_with_boxes(
        self,
        boxes: torch.Tensor,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompts = InstanceData(
            bboxes=self.transform.apply_boxes_torch(boxes,self.original_size),
        )
        
        # TODO: 考虑extract_masks 是否支持批量并行
        masks, cls_pred, sematic_embed = self.ovsam.extract_masks(self.img_feat,prompts=prompts)
        masks = self.ovsam.postprocess_masks(masks,self.input_size,self.original_size)
        
        if not return_logits:
            masks = (masks > self.ovsam.MASK_THRESHOLD)
        return masks, None, cls_pred, sematic_embed
        
    def reset_image(self,):
        self.is_image_set = False
        self.img_feat = None
        
        