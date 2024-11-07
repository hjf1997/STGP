import torch.nn as nn

from .mask import MaskGenerator
from ..tsformer.tsformer import TSFormerEncoder
from .fusion import EncoderFusion
from ..graphormer.graphormer import GraphormerEncoder


class STGEncoder(nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        # Encoder
        self.t_encoder = TSFormerEncoder(cfg['embed_dim'], cfg['num_heads'], cfg['mlp_ratio'], cfg['mask_ratio'], cfg['encoder_depth'], cfg['dropout'])
        self.s_encoder = GraphormerEncoder(cfg['num_in_degree'], cfg['num_out_degree'], cfg['num_spatial'],
                                           cfg['embed_dim'], cfg['num_heads'], cfg['mlp_ratio'], cfg['mask_ratio'], cfg['encoder_depth'], cfg['dropout'])
        ## fusion
        self.en_fusion = EncoderFusion(cfg['embed_dim'])
        self.mask = MaskGenerator()
        self.mask_ratio = cfg['mask_ratio']

    def forward(self, patches, spatial_pos, in_degree, out_degree, feat_patch,
                sprompter=None, tprompter=None, random_mask=True, task=None, **kwargs):
        """
        temporal_unmask (list), spatial_unmask (list) -> extrapolation
        temporal_unmask (list), spatial_unmask (None) -> forecasting
        temporal_unmask (None), spatial_unmask (list) -> kriging
        temporal_unmask (None), spatial_unmask (None) -> pre-training, domain prompting
        :param patches: Conditional information with the shape [B, N, P, D]
        :param spatial_pos: adjacency matrix with the shape [N, N]
        :param in_degree: in-degree matrix with the shape [N, D]
        :param out_degree: out-degree matrix with the shape [N, D]
        :param feat_patch: feature patch with the shape [B, N, P, 2]
        :param (bool) random_mask: True in pre-training stage and False in the rest.
        :param down_indexes (dir or None):  indexes for downstream tasks
        :return:
        """
        if task is not None: assert random_mask is False
        B, N, P, D = patches.shape

        t_uti, t_mti, s_uti, s_mti = None, None, None, None
        if random_mask:
            t_uti, t_mti = self.mask(P, self.mask_ratio)
            s_uti, s_mti = self.mask(N, self.mask_ratio)
        else:
            if task in ['forecasting', 'extrapolation']:
                his_num = self.cfg['task']['forecasting']['history_patch']
                fu_num = self.cfg['task']['forecasting']['future_patch'] + his_num
                assert fu_num == self.cfg['task']['num_patch']
                t_uti = [i for i in range(his_num)]
                t_mti = [i for i in range(his_num, fu_num)]
            if task in ['kriging', 'extrapolation']:
                s_uti = kwargs['s_uti']
                s_mti = kwargs['s_mti']

        if s_uti is not None:
            t_patches, t_feat = patches[:, s_uti], feat_patch[:, s_uti]
        else:
            t_patches, t_feat = patches, feat_patch
        t_patches = self.t_encoder(t_patches, t_feat, prompter=tprompter, t_uti=t_uti)

        if t_uti is not None:
            s_patches = patches[:, :, t_uti]
        else:
            s_patches = patches
        s_patches = self.s_encoder(s_patches, in_degree, out_degree, spatial_pos, prompter=sprompter, s_uti=s_uti)

        full_patches = self.en_fusion(t_patches, t_mti, t_uti, s_patches, s_mti, s_uti)
        return full_patches, t_uti, t_mti, s_uti, s_mti
