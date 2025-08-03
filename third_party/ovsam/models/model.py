import torch.nn.functional as F
from mmengine.model import BaseModel

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from typing import Any, Dict, List, Tuple
import torch


@MODELS.register_module()
class SAMSegmentor(BaseModel):
    MASK_THRESHOLD = 0.5

    def __init__(
            self,
            backbone: ConfigType,
            neck: ConfigType,
            prompt_encoder: ConfigType,
            mask_decoder: ConfigType,
            data_preprocessor: OptConfigType = None,
            fpn_neck: OptConfigType = None,
            init_cfg: OptMultiConfig = None,
            use_clip_feat: bool = False,
            use_head_feat: bool = False,
            use_gt_prompt: bool = False,
            use_point: bool = False,
            enable_backbone: bool = False,
            image_processed_size: int = 1024,
            pixel_mean: List[float] = [123.675, 116.28, 103.53],
            pixel_std: List[float] = [58.395, 57.12, 57.375],
            **kwargs
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.pe = MODELS.build(prompt_encoder)
        self.mask_decoder = MODELS.build(mask_decoder)
        if fpn_neck is not None:
            self.fpn_neck = MODELS.build(fpn_neck)
        else:
            self.fpn_neck = None

        self.use_clip_feat = use_clip_feat
        self.use_head_feat = use_head_feat
        self.use_gt_prompt = use_gt_prompt
        self.use_point = use_point

        self.enable_backbone = enable_backbone
        self.image_processed_size = image_processed_size
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    def extract_feat(self, inputs):
        backbone_feat = self.backbone(inputs)
        neck_feat = self.neck(backbone_feat)

        if self.fpn_neck is not None:
            fpn_feat = self.fpn_neck(backbone_feat)
        else:
            fpn_feat = None

        return dict(
            backbone_feat=backbone_feat,
            neck_feat=neck_feat,
            fpn_feat=fpn_feat
        )

    def extract_masks(self, feat_cache, prompts):
        sparse_embed, dense_embed = self.pe(
            prompts,
            image_size=(1024, 1024),
            with_points='point_coords' in prompts,
            with_bboxes='bboxes' in prompts,
        )

        kwargs = dict()
        if self.enable_backbone:
            kwargs['backbone_feats'] = feat_cache['backbone_feat']
            kwargs['backbone'] = self.backbone
            kwargs['fpn_feats'] = feat_cache['fpn_feat']
        
        low_res_masks, iou_predictions, cls_pred, sematic_embed = self.mask_decoder(
            image_embeddings=feat_cache['neck_feat'],
            image_pe=self.pe.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embed,
            dense_prompt_embeddings=dense_embed,
            multi_mask_output=False,
            **kwargs
        )
        # masks = F.interpolate(
        #     low_res_masks,
        #     scale_factor=4.,
        #     mode='bilinear',
        #     align_corners=False,
        # )

        # masks = masks.sigmoid()
        cls_pred = cls_pred.softmax(-1)[..., :-1]
        return low_res_masks, cls_pred, sematic_embed

    def forward(self, inputs):
        return inputs
    
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.image_processed_size, self.image_processed_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks
    
    def preprocess(self, x):
        """Normalize pixel values and pad to a square input."""
        
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std
        
        h, w = x.shape[-2:]
        padh = self.image_processed_size - h
        padw = self.image_processed_size - w
        x = F.pad(x, (0, padw, 0, padh), 'constant', 0)
        return x
        
