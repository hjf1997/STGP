import torch.nn as nn
import torch


class EncoderFusion(nn.Module):

    def __init__(self, embed_dim) -> None:
        super().__init__()
        self.w_t = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.w_s = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.b = nn.Parameter(torch.randn(embed_dim))

        #  mask token
        self.t_mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.s_mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        nn.init.trunc_normal_(self.t_mask_token, std=.02)
        nn.init.trunc_normal_(self.s_mask_token, std=.02)

    def forward(self, t_x, t_mti, t_uti, s_x, s_mti, s_uti):
        """
        temporal_mask (list); spatial_mask (None) -> forecasting
        temporal_mask (None); spatial_mask (list) -> kriging
        temporal_mask (list); spatial_mask (list) -> extrapolation
        temporal_mask (None); spatial_mask (None) -> pre-training, domain prompting

        :param t_x: temporal information with the shape [B, N, P, D]
        :param t_mti: (list): temporal masked token index
        :param t_uti: (list): temporal unmasked token index
        :param s_x: spatial information with the shape [B, N, P, D]
        :param s_mti: (list): spatial masked token index
        :param s_uti: (list): spatial unmasked token index
        :return:
        """

        t_patches = self.recover_mask(t_x, self.t_mask_token, t_uti, t_mti, s_uti, s_mti)
        s_patches = self.recover_mask(s_x, self.s_mask_token, t_uti, t_mti, s_uti, s_mti)

        gate = torch.sigmoid(torch.matmul(t_patches, self.w_t) + torch.matmul(s_patches, self.w_s) + self.b)
        full_patches = gate * t_patches + (1 - gate) * s_patches  # [B, N, P, D]

        return full_patches

    def recover_mask(self, patches, mask_token, t_uti, t_mti, s_uti, s_mti):
        B, N_unmask, P_unmask, D = patches.shape
        if t_uti is not None and s_uti is not None:
            patches_temp1 = mask_token.repeat(B, N_unmask, len(t_uti) + len(t_mti), 1)
            patches_temp1[:, :, t_uti, :] = patches
            patches_temp2 = mask_token.repeat(B, len(s_uti) + len(s_mti), len(t_uti) + len(t_mti), 1)
            patches_temp2[:, s_uti, :] = patches_temp1
            unmasked_patches = patches_temp2
        elif s_uti is not None:
            unmasked_patches = mask_token.repeat(B, len(s_uti) + len(s_mti), P_unmask, 1)
            unmasked_patches[:, s_uti, :] = patches
        elif t_uti is not None:
            unmasked_patches = mask_token.repeat(B, N_unmask, len(t_uti) + len(t_mti), 1)
            unmasked_patches[:, :, t_uti, :] = patches
        else:
            unmasked_patches = patches
        return unmasked_patches

class DecoderFusion(nn.Module):
    """Fusion and prediction module."""
    def __init__(self, patch_size, embed_dim) -> None:
        super().__init__()
        self.w_t = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.w_s = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.b = nn.Parameter(torch.randn(embed_dim))

        self.output_layer = nn.Linear(embed_dim, patch_size)

    def forward(self, t_x, s_x):
        """

        :param t_x: temporal features with the shape [B, N, P, D,]
        :param s_x: spatial features with the shape [B, N, P, D]
        :return:
        """

        gate = torch.sigmoid(torch.matmul(t_x, self.w_t) + torch.matmul(s_x, self.w_s) + self.b)
        full_patches = gate * t_x + (1 - gate) * s_x  # [B, N, P, D]

        return self.output_layer(full_patches)


