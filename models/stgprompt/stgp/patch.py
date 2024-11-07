import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """Patchify time series."""

    def __init__(self, cfg, norm_layer=None):
        super().__init__()
        self.output_channel = cfg['embed_dim']
        self.len_patch = cfg['task']['patch_size']             # L
        self.input_channel = cfg['in_channel']
        self.output_channel = cfg['embed_dim']
        self.input_embedding = nn.Conv2d(
                                        cfg['in_channel'],
                                        cfg['embed_dim'],
                                        kernel_size=(self.len_patch, 1),
                                        stride=(self.len_patch, 1))
        self.feature_patching = nn.AvgPool2d(
                                            kernel_size=(self.len_patch, 1),
                                            stride=(self.len_patch, 1)
        )
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()

    def forward(self, long_term_history, long_feat):
        """
        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, L, D],
                                                which is used in the TSFormer.
                                                P is the number of segments (patches).

        Returns:
            torch.Tensor: patchified time series with shape [B, N, P, D]
        """

        batch_size, num_nodes, len_time_series, num_feat = long_term_history.shape
        ## for time of day, day of week first
        # todo: hard coding part
        long_feat = long_feat.transpose(-1, -2).unsqueeze(-1)  # B, N, 2, L, 1
        long_feat = long_feat.reshape(batch_size*num_nodes, 2, len_time_series, 1)
        feat_patch = self.feature_patching(long_feat)
        feat_patch = feat_patch.squeeze(-1).view(batch_size, num_nodes, 2, -1)    # B, N, 2, P
        feat_patch = feat_patch.transpose(-1, -2)
        feat_patch = torch.round(feat_patch).long()

        long_term_history = long_term_history.transpose(-1, -2).unsqueeze(-1)  # B, N, D, L, 1
        # B*N,  C, L, 1
        long_term_history = long_term_history.reshape(batch_size*num_nodes, num_feat, len_time_series, 1)
        # B*N,  d, L/P, 1
        output = self.input_embedding(long_term_history)
        # norm
        output = self.norm_layer(output)
        # reshape
        output = output.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)    # B, N, d, P
        assert output.shape[-1] == len_time_series / self.len_patch
        output = output.transpose(-1, -2)   # B, N, P, d
        return output, feat_patch
