import torch
from torch import nn

from ..stgp.mask import MaskGenerator
from .positional_encoding import PositionalEncoding
from .transformer_layers import TransformerLayers


class TSFormerEncoder(nn.Module):
    """An efficient unsupervised pre-training model for Time Series based on transFormer blocks. (TSFormer)"""

    def __init__(self, embed_dim, num_heads, mlp_ratio, mask_ratio, encoder_depth, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.mlp_ratio = mlp_ratio

        # norm layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        # encoder specifics
        # # positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        # encoder
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)

        self.initialize_weights()

    def initialize_weights(self):
        # positional encoding
        nn.init.uniform_(self.positional_encoding.position_embedding, -.02, .02)

    def encoding(self, long_term_patches, feat_patch, prompter, t_uti):
        """
        Args:
            long_term_patches (torch.Tensor): patch embeddings with shape [B, N, P, D]
            feat_patch (torch.Tensor): feature patch with the shape [B, N, P, 2]
        Returns:
            torch.Tensor: hidden states of unmasked tokens
            list: unmasked token index
            list: masked token index
        """

        batch_size, num_nodes, num_patches, _ = long_term_patches.shape
        # positional embedding
        patches = self.positional_encoding(long_term_patches, feat_patch)
        if t_uti is not None:
            patches = patches[:, :, t_uti, :]
        if prompter is not None: patches = prompter(patches)

        # encoding
        hidden_states_unmasked = self.encoder(patches)
        hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)

        return hidden_states_unmasked

    def forward(self, history_data: torch.Tensor, feat_patch, prompter=None, t_uti=None):
        """
        Args:
            history_data (torch.Tensor): very long-term historical time series patches with shape [B, N, P, D]
        Returns:
        """
        hidden_states_unmasked = self.encoding(history_data, feat_patch, prompter, t_uti)
        return hidden_states_unmasked