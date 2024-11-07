import torch
import torch.nn as nn

from models.stgprompt.tsformer.positional_encoding import PositionalEncoding
from models.stgprompt.decoder.gatednet import GatedNet
import torch.nn.functional as F


class STGDecoder(nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        embed_dim = cfg['embed_dim']
        dropout = cfg['dropout']
        patch_size = cfg['task']['patch_size']
        num_patch = cfg['task']['num_patch']

        self.tod_embedding = nn.Embedding(24, embed_dim)
        nn.init.kaiming_uniform_(self.tod_embedding.weight, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        self.dow_embedding = nn.Embedding(7, embed_dim)
        nn.init.kaiming_uniform_(self.dow_embedding.weight, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        self.gwnet = GatedNet(in_dim=embed_dim,
                           out_dim=patch_size,
                           dropout=dropout,)

    def forward(self, patches, spatial_pos, feat_patch, prompter=None, **kwargs):
        """

        :param patches: Conditional information with the shape [B, N, P, D] , or [B, N+1, P+1, D] for task prompting
        :param spatial_pos: adjacency matrix with the shape [N, N]
        :return:
        """
        time_embedding = self.tod_embedding(feat_patch[:, :, :, 0]) + self.dow_embedding(feat_patch[:, :, :, 1])
        patches = patches + time_embedding
        if prompter is not None: patches = prompter(patches, **kwargs)
        decoded_patches = self.gwnet(patches, [spatial_pos / spatial_pos.sum(1, keepdim=True)
            , spatial_pos.transpose(0, 1) / spatial_pos.sum(0, keepdim=True)])

        return decoded_patches


class PredictionHead(nn.Module):

    def __init__(self, cfg, dropout=None) -> None:
        super().__init__()
        patch_size = cfg['task']['patch_size']
        # dropout here is in case of overwritting dropout in config
        if dropout is None: dropout = cfg['dropout']
        self.end_conv_1 = nn.Conv2d(in_channels=128,
                                  out_channels=256,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=256,
                                    out_channels=patch_size,
                                    kernel_size=(1,1),
                                    bias=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, patches):
        patches = F.relu(patches)
        patches = F.relu(self.end_conv_1(patches))
        patches = self.dropout(patches)
        patches = self.end_conv_2(patches)
        patches = patches.permute(0, 2, 3, 1)

        B, N, _, _ = patches.shape
        patches = patches.reshape(B, N, -1).unsqueeze(-1)  # B, N, P*D, 1
        return patches
