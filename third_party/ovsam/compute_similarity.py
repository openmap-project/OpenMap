import numpy as np
import clip
import torch
import torch.nn.functional as F

class ComputeSimilarity:
    def __init__(self, model_name, device):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

    def get_text_features(self, text):
        with torch.no_grad():
            text_features = self.model.encode_text(clip.tokenize(text).to(self.device))
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def compute_similarity(self, text_feature,instance_feature_list):
        print("text_feature:",text_feature.shape)
        print("instance_feature_list:",instance_feature_list.shape)
        print("intstance_feature_list[0]:",instance_feature_list[0].shape)
        instance_pred = torch.einsum('bc,lc->bl', F.normalize(text_feature, dim=-1).float(), F.normalize(instance_feature_list,dim=-1).float())
        # instance_pred = torch.mm(F.normalize(text_feature, dim=-1).float(), instance_feature_list.transpose(0,1).float())
        instance_preds,indices = torch.topk(instance_pred, 1, dim=-1)
        # 原来代码中的logit_scale 是为了获得 逻辑分数的
        # logit_scale = torch.tensor(4.6052, dtype=torch.float32)
        # instance_pred = logit_scale.exp() * instance_pred
        return instance_preds.cpu().numpy(),indices.cpu().numpy()