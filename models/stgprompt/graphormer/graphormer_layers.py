# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class GraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(
        self, num_in_degree, num_out_degree, hidden_dim, n_layers
    ):
        super(GraphNodeFeature, self).__init__()

        # 1 for graph token
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(
            num_out_degree, hidden_dim, padding_idx=0
        )

        # self.graph_token = nn.Embedding(1, hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, x, in_degree, out_degree):
        """

        :param x: [B, N, P, D]
        :param in_degree: [N, D]
        :param out_degree: [N, D]
        :return:
        """
        # if x.shape[1] == in_degree.shape[0] + 1:  # prompting
        #     node_feature = x[:, 1:, :, :]
        # else:
        node_feature = x
        node_feature = (
            node_feature
            + self.in_degree_encoder(in_degree).unsqueeze(0).unsqueeze(2)
            + self.out_degree_encoder(out_degree).unsqueeze(0).unsqueeze(2)
        )
        # if x.shape[1] == in_degree.shape[0] + 1:  # prompting
        #     node_feature = torch.cat([x[:, 0:1, :, :], node_feature], dim=1)

        return node_feature


class GraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(
        self,
        num_heads,
        num_spatial,
        n_layers,
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.num_spatial = num_spatial
        self.spatial_pos_encoder = nn.Embedding(num_spatial+1, num_heads, padding_idx=0)
        # here 1 for a node itself
        # self.graph_token_virtual_distance = nn.Embedding(1, num_heads)
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, spatial_pos, x):

        batch_size, num_nodes = x.size()[0], x.size()[1]
        graph_attn_bias = torch.floor(spatial_pos / (1 / self.num_spatial)).long()
        graph_attn_bias = self.spatial_pos_encoder(graph_attn_bias).permute(2, 0, 1)  # [H, N, N]
        graph_attn_bias = graph_attn_bias.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [B, H, N, N]

        if x.shape[1] == spatial_pos.shape[1] + 1:  # prompting
            # use max adj weight as graph token virtual distance
            padding = self.spatial_pos_encoder.weight[-1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, H, 1, 1]
            # [B, H, N, N] -> [B, H, N+1, N+1]
            # padding must be the first
            graph_attn_bias = torch.cat([padding.repeat(batch_size, 1, 1, num_nodes-1)
                                            , graph_attn_bias], dim=2)
            graph_attn_bias = torch.cat([padding.repeat(batch_size, 1, num_nodes, 1),
                                         graph_attn_bias], dim=3)
        return graph_attn_bias
