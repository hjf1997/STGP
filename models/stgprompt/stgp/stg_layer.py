import torch
import torch.nn as nn

from models.stgprompt.graphormer import GraphormerGraphEncoderLayer
from torch.nn import TransformerEncoderLayer
import math


# class STGLayer(nn.Module):
#
#     def __init__(self, cfg) -> None:
#         super().__init__()
#         embed_dim = cfg['embed_dim']
#         num_attention_heads = cfg['num_heads']
#         mlp_ratio = cfg['mlp_ratio']
#         dropout = cfg['dropout']
#         self.embedding_dim = embed_dim
#
#         self.s_layer = GraphormerGraphEncoderLayer(
#                     embedding_dim=self.embedding_dim,
#                     ffn_embedding_dim=mlp_ratio * embed_dim,
#                     num_attention_heads=num_attention_heads,
#                     dropout=dropout,
#                     attention_dropout=dropout,
#                     activation_dropout=dropout,
#                     pre_layernorm=False,
#                 )
#         self.t_layer = TransformerEncoderLayer(embed_dim, mlp_ratio, num_attention_heads, dropout,  norm_first=False)
#         # self.w_t = nn.Parameter(torch.randn(embed_dim, embed_dim))
#         # self.w_s = nn.Parameter(torch.randn(embed_dim, embed_dim))
#         # self.b = nn.Parameter(torch.randn(embed_dim))
#
#     def forward(self, patches, attn_bias):
#
#         # no need to reshape
#         t_patches = self.t_layer(patches)
#
#         # B, N, P, D -> N, BxP, D
#         num_batch, n_node, n_patch = patches.size()[:3]
#         s_patches = patches.transpose(0, 1).contiguous().view(n_node, -1, self.embedding_dim)
#         attn_bias = attn_bias.repeat(n_patch, 1, 1, 1)
#         s_patches, _ = self.s_layer(s_patches, attn_bias)
#         s_patches = s_patches.view(n_node, num_batch, -1, self.embedding_dim).transpose(0, 1)
#
#         full_patches = t_patches + s_patches
#         # gate = torch.sigmoid(torch.matmul(t_patches, self.w_t) + torch.matmul(s_patches, self.w_s) + self.b)
#         # full_patches = gate * t_patches + (1 - gate) * s_patches  # [B, N, P, D]
#         return full_patches


class STGLayerSep(nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()
        embed_dim = cfg['embed_dim']
        num_attention_heads = cfg['num_heads']
        mlp_ratio = cfg['mlp_ratio']
        dropout = cfg['dropout']
        self.embedding_dim = embed_dim

        self.s_layer = GraphormerGraphEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=mlp_ratio * embed_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                    attention_dropout=dropout,
                    activation_dropout=dropout,
                    pre_layernorm=False,
                )
        self.t_layer = TransformerEncoderLayer(embed_dim, mlp_ratio, num_attention_heads, dropout,  norm_first=False)

    def forward(self, t_patches, s_patches, attn_bias):

        # no need to reshape
        B, N, L, D = t_patches.shape
        t_patches = t_patches * math.sqrt(self.embedding_dim)
        t_patches = t_patches.view(B*N, L, D)
        t_patches = t_patches.transpose(0, 1)
        t_patches = self.t_layer(t_patches)
        t_patches = t_patches.transpose(0, 1).view(B, N, L, D)
        # if t_mti is not None:
        #     t_patches[:, s_mti] = t_token

        # B, N, P, D -> N, BxP, D
        num_batch, n_node, n_patch = s_patches.size()[:3]
        s_patches = s_patches.transpose(0, 1).contiguous().view(n_node, -1, self.embedding_dim)
        attn_bias = attn_bias.repeat(n_patch, 1, 1, 1)
        s_patches, _ = self.s_layer(s_patches, attn_bias)
        s_patches = s_patches.view(n_node, num_batch, -1, self.embedding_dim).transpose(0, 1)
        # if s_mti is not None:
        #     s_patches[:, :, t_mti] = s_token

        full_patches = t_patches + s_patches
        return full_patches
