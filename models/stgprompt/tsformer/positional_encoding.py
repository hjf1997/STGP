import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, hidden_dim, dropout=0.1, max_len: int = 30):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Parameter(torch.randn(max_len, hidden_dim), requires_grad=True)
        nn.init.kaiming_uniform_(self.position_embedding, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        self.tod_embedding = nn.Embedding(24, hidden_dim)
        nn.init.kaiming_uniform_(self.tod_embedding.weight, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        self.dow_embedding = nn.Embedding(7, hidden_dim)
        nn.init.kaiming_uniform_(self.dow_embedding.weight, nonlinearity='leaky_relu', mode='fan_in', a=0.01)

    def forward(self, input_data, feat_patch):
        """Positional encoding

        Args:
            input_data (torch.tensor): input sequence with shape [B, N, P, D].
            feat_patch
            prompting: whether to add positional embedding to the first token (prompt).

        Returns:
            torch.tensor: output sequence
        """

        batch_size, num_nodes, num_patches, num_feat = input_data.shape
        input_data = input_data.reshape(batch_size*num_nodes, num_patches, num_feat)
        # positional encoding
        # do not add positional embedding to the first token (prompt)
        # if prompting is False:
        pe = self.position_embedding[:input_data.size(1), :].unsqueeze(0)
        input_data = input_data + pe
        # else:
        #     pe = self.position_embedding[:input_data.size(1)-1, :].unsqueeze(0)
        #     prompting_data = input_data[:, 1:] + pe
        #     input_data = torch.cat([input_data[:, 0:1], prompting_data], dim=1)

        # reshape
        input_data = input_data.reshape(batch_size, num_nodes, num_patches, num_feat)

        # tod and dow encoding
        time_embedding = self.tod_embedding(feat_patch[:, :, :, 0]) + self.dow_embedding(feat_patch[:, :, :, 1])
        # if prompting is False:
        input_data = input_data + time_embedding
        # else:
        #     prompting_data = input_data[:, 1:] + time_embedding
        #     input_data = torch.cat([input_data[:, 0:1], prompting_data], dim=1)

        input_data = self.dropout(input_data)

        return input_data
