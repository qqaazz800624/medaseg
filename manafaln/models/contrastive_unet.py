from typing import Sequence

import segmentation_models_pytorch as smp
import torch


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

class DenseLayer(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_channels),
            torch.nn.PReLU(num_parameters=in_channels, init=0.25),
            torch.nn.Linear(in_features=in_channels, out_features=out_channels),
            )

    def forward(self,x):
        return self.layer(x)

class ClassificationHeadwithProjector(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_feats
    ):
        super().__init__()
        self.flatten =  torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(1)
            )
        self.projector =  torch.nn.Sequential(
            DenseLayer(in_channels, num_feats),
            )
        self.classifier = torch.nn.Sequential(
            DenseLayer(in_channels, 512),
            DenseLayer(512,128),
            DenseLayer(128,32),
            DenseLayer(32, out_channels)
            )

    def forward(self, x):
        x = self.flatten(x)
        features = self.projector(x)
        clf_output = self.classifier(x)
        return features, clf_output

class ContrastiveUNet(smp.Unet):
    def __init__(self,
        num_feats: int = 128,
        in_channels: int = 9,
        seg_channels: int = 8,
        clf_channels: int = 6,
        encoder_name: str = "tu-resnest50d",
        encoder_depth: int = 5,
        decoder_channels: Sequence[int] = [512, 256, 128, 64, 32],
        decoder_attention_type: str = "scse"
    ):
        super().__init__(
            in_channels=in_channels,
            classes=seg_channels,
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type
            )

        self.classification_head = ClassificationHeadwithProjector(
            in_channels=self.encoder.out_channels[-1],
            out_channels=clf_channels,
            num_feats=num_feats,
        )

        initialize_head(self.classification_head)

    def forward(self, x):
        seg_output, features_and_clf_output = super().forward(x)
        features, clf_output = features_and_clf_output
        return seg_output, features, clf_output
