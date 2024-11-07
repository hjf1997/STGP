# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .multihead_attention import MultiheadAttention
from .graphormer_layers import GraphNodeFeature, GraphAttnBias
from .graphormer_encoder_layer import GraphormerGraphEncoderLayer
from ..stgp.mask import MaskGenerator



def init_graphormer_params(module):
    """
    Initialize the weights specific to the Graphormer Model.
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class GraphormerEncoder(nn.Module):
    def __init__(
        self,
        num_in_degree: int,
        num_out_degree: int,
        num_spatial: int,
        embedding_dim: int,
        num_attention_heads: int,
        mlp_ratio: int,
        mask_ratio: float,
        num_encoder_layers: int,
        dropout: float,
        ####################################
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        pre_layernorm: bool = False,
        apply_graphormer_init: bool = False,
    ) -> None:

        super().__init__()
        self.dropout_module = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim
        self.apply_graphormer_init = apply_graphormer_init
        self.mask_ratio = mask_ratio

        self.graph_node_feature = GraphNodeFeature(
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
        )

        self.graph_attn_bias = GraphAttnBias(
            num_heads=num_attention_heads,
            num_spatial=num_spatial,
            n_layers=num_encoder_layers,
        )

        self.mask = MaskGenerator()

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_graphormer_graph_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=mlp_ratio * embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    # activation_fn=activation_fn,
                    pre_layernorm=pre_layernorm,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # Apply initialization of model params after building the model
        if self.apply_graphormer_init:
            self.apply(init_graphormer_params)

    def build_graphormer_graph_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        pre_layernorm,
    ):
        return GraphormerGraphEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            pre_layernorm=pre_layernorm,
        )

    def forward(
        self,
        x,
        in_degree,
        out_degree,
        spatial_pos,
        s_uti = None,
        attn_mask: Optional[torch.Tensor] = None,
        prompter=None,
    ):
        """

        :param x: [B, N, P, D]
        :param in_degree: [N, D]
        :param out_degree: [N, D]
        :param spatial_pos: [N, N]
        :param attn_mask:
        :return:
        """
        num_batch, n_node, n_patch = x.size()[:3]

        x = self.graph_node_feature(x, in_degree, out_degree)
        if prompter is not None: x = prompter(x)
        attn_bias = self.graph_attn_bias(spatial_pos, x)
        if s_uti is not None:
            x = x[:, s_uti]
            attn_bias = attn_bias[:, :, s_uti][..., s_uti]
            n_node = len(s_uti)
        x = self.dropout_module(x)

        # B, N, P, D -> N, BxP, D
        x = x.transpose(0, 1).contiguous().view(n_node, -1, self.embedding_dim)
        attn_bias = attn_bias.repeat(n_patch, 1, 1, 1)
        for layer in self.layers:
            x, _ = layer(
                x,
                self_attn_padding_mask=None,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
            )

        inner_states = x.view(n_node, num_batch, -1, self.embedding_dim).transpose(0, 1)

        #  N, BxP, D -> B, N, P, D
        return inner_states


class GraphormerDecoder(nn.Module):
    def __init__(
        self,
        num_in_degree: int,
        num_out_degree: int,
        num_spatial: int,
        embedding_dim: int,
        num_attention_heads: int,
        mlp_ratio: float,
        num_encoder_layers: int,
        dropout: float,
        ####################################
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        pre_layernorm: bool = False,
        apply_graphormer_init: bool = False,
    ) -> None:

        super().__init__()
        self.dropout_module = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim
        self.apply_graphormer_init = apply_graphormer_init

        self.graph_node_feature = GraphNodeFeature(
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
        )

        self.graph_attn_bias = GraphAttnBias(
            num_heads=num_attention_heads,
            num_spatial=num_spatial,
            n_layers=num_encoder_layers,
        )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_graphormer_graph_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=self.embedding_dim * mlp_ratio,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    pre_layernorm=pre_layernorm,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # Apply initialization of model params after building the model
        if self.apply_graphormer_init:
            self.apply(init_graphormer_params)

    def build_graphormer_graph_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        pre_layernorm,
    ):
        return GraphormerGraphEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            pre_layernorm=pre_layernorm,
        )

    def forward(
        self,
        x,
        in_degree,
        out_degree,
        spatial_pos,
        attn_mask: Optional[torch.Tensor] = None,
        prompting = False,
    ) -> torch.Tensor:
        # compute padding mask. This is needed for multi-head attention
        """

        :param x: [B, N, P, D]
        :param in_degree: [N, D]
        :param out_degree: [N, D]
        :param spatial_pos: [N, N]
        :param attn_mask:
        :param prompting:
        :return:
        """
        num_batch, n_node, n_patch = x.size()[:3]
        if prompting:
            assert x.shape[1] == in_degree.shape[0] + 1
            x_prompting = self.graph_node_feature(x[:, 1:, 1:], in_degree, out_degree)
            x_prompting = torch.cat([x[:, 0:1, 1:, :], x_prompting], dim=1)
            x_prompting = torch.cat([x[:, :, 0:1, :], x_prompting], dim=2)
            x = x_prompting
        else:
            x = self.graph_node_feature(x, in_degree, out_degree)
        attn_bias = self.graph_attn_bias(spatial_pos, x)

        x = self.dropout_module(x)

        # B, N, P, D -> N, BxP, D
        x = x.transpose(0, 1).contiguous().view(n_node, -1, self.embedding_dim)
        attn_bias = attn_bias.repeat(n_patch, 1, 1, 1)

        for layer in self.layers:
            x, _ = layer(
                x,
                self_attn_padding_mask=None,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
            )
        inner_states = x.view(n_node, num_batch, -1, self.embedding_dim).transpose(0, 1)

        return inner_states

