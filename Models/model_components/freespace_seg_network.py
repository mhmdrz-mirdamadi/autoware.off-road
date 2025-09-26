import torch.nn as nn
from .backbone import Backbone
from .freespace_feature_fusion import FreespaceFeatureFusion
from .freespace_context import FreespaceContext
from .freespace_neck import FreespaceNeck
from .freespace_seg_head import FreespaceSegHead


class FreespaceSegNetwork(nn.Module):
    def __init__(self):
        super(FreespaceSegNetwork, self).__init__()

        self.Backbone = Backbone()
        self.FeatureFusion = FreespaceFeatureFusion()
        self.Context = FreespaceContext()
        self.Neck = FreespaceNeck()
        self.Head = FreespaceSegHead()

    def forward(self, image):
        features = self.Backbone(image)
        fused_features = self.FeatureFusion(features)
        context = self.Context(fused_features)
        neck = self.Neck(context, features)
        output = self.Head(neck, features)
        return output
