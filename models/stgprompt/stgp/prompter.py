import torch
import torch.nn as nn


class AttentionPrompt(nn.Module):
    def __init__(self, cfg, num_prompt=None) -> None:
        super().__init__()
        if num_prompt is None:
            num_prompt = cfg['num_prompt']
        embed_dim, prompt_threshold = cfg['embed_dim'], cfg['prompt_threshold']
        self.prompt = nn.Embedding(num_prompt, embed_dim)
        nn.init.kaiming_uniform_(self.prompt.weight, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        self.prompt_threshold = prompt_threshold

    def forward(self, patches, **kwargs):
        """

        :param patches: [B, N, P, D]
        :return: prompted_patches
        """
        # spatial dimension
        score_map = torch.sigmoid(torch.matmul(patches, self.prompt.weight.t()))  # [B, N, P, num_prompt]
        score_map = score_map * (score_map > self.prompt_threshold).float()
        prompted_patches = patches + torch.sum(score_map.unsqueeze(-1) * self.prompt.weight, dim=-2)
        return prompted_patches

class AttentionPromptKriging(nn.Module):
    def __init__(self, cfg, num_prompt=None) -> None:
        super().__init__()
        if num_prompt is None:
            num_prompt = cfg['num_prompt']
        embed_dim, prompt_threshold = cfg['embed_dim'], cfg['prompt_threshold']
        self.u_prompt = nn.Embedding(num_prompt, embed_dim)
        nn.init.kaiming_uniform_(self.u_prompt.weight, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        self.m_prompt = nn.Embedding(num_prompt, embed_dim)
        # self.m_prompt = nn.Parameter(torch.randn(1, 1, 1, embed_dim))
        nn.init.kaiming_uniform_(self.m_prompt.weight, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        self.prompt_threshold = prompt_threshold

    def forward(self, patches, **kwargs):
        """

        :param patches: [B, N, P, D]
        :return: prompted_patches
        """
        s_mti = kwargs['s_mti']
        s_uti = kwargs['s_uti']

        score_map = torch.sigmoid(torch.matmul(patches, self.u_prompt.weight.t()))  # [B, N, P, num_prompt]
        score_map = score_map * (score_map > self.prompt_threshold).float()
        score_map[:, s_mti] = 0  # only prompt unmasked patches
        prompted_patches = patches + torch.sum(score_map.unsqueeze(-1) * self.u_prompt.weight, dim=-2)

        score_map = torch.sigmoid(torch.matmul(patches, self.m_prompt.weight.t()))  # [B, N, P, num_prompt]
        score_map = score_map * (score_map > self.prompt_threshold).float()
        score_map[:, s_uti] = 0  # only prompt masked patches
        prompted_patches = prompted_patches + torch.sum(score_map.unsqueeze(-1) * self.m_prompt.weight, dim=-2)

        # masked_prompt = self.m_prompt.repeat(patches.shape[0], patches.shape[1], patches.shape[2], 1)
        # masked_prompt[:, s_uti] = 0
        # prompted_patches = prompted_patches + masked_prompt

        return prompted_patches

class AttentionPromptForecasting(nn.Module):
    """Only Prompt masked patches"""
    def __init__(self, cfg, num_prompt=None) -> None:
        super().__init__()
        self.cfg = cfg
        his_num = self.cfg['task']['forecasting']['history_patch']
        fu_num = self.cfg['task']['forecasting']['future_patch'] + his_num
        assert fu_num == self.cfg['task']['num_patch']
        self.t_uti = [i for i in range(his_num)]
        self.t_mti = [i for i in range(his_num, fu_num)]

        if num_prompt is None:
            num_prompt = cfg['num_prompt']
        embed_dim, prompt_threshold = cfg['embed_dim'], cfg['prompt_threshold']
        self.u_prompt = nn.Embedding(num_prompt, embed_dim)
        nn.init.kaiming_uniform_(self.u_prompt.weight, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        self.m_prompt = nn.Embedding(num_prompt, embed_dim)
        # self.m_prompt = nn.Parameter(torch.randn(1, 1, 1, embed_dim))
        nn.init.kaiming_uniform_(self.m_prompt.weight, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        self.prompt_threshold = prompt_threshold

    def forward(self, patches, **kwargs):
        """

        :param patches: [B, N, P, D]
        :return: prompted_patches
        """
        score_map = torch.sigmoid(torch.matmul(patches, self.u_prompt.weight.t()))  # [B, N, P, num_prompt]
        score_map = score_map * (score_map > self.prompt_threshold).float()
        score_map[:, :, self.t_mti] = 0  # only prompt last patch
        prompted_patches = patches + torch.sum(score_map.unsqueeze(-1) * self.u_prompt.weight, dim=-2)

        score_map = torch.sigmoid(torch.matmul(patches, self.m_prompt.weight.t()))  # [B, N, P, num_prompt]
        score_map = score_map * (score_map > self.prompt_threshold).float()
        score_map[:, :, self.t_uti] = 0  # only prompt last patch
        prompted_patches = prompted_patches + torch.sum(score_map.unsqueeze(-1) * self.m_prompt.weight, dim=-2)

        return prompted_patches


class AttentionPromptExtrapolation(nn.Module):

    def __init__(self, cfg, num_prompt=None) -> None:
        super().__init__()
        self.cfg = cfg
        his_num = self.cfg['task']['forecasting']['history_patch']
        fu_num = self.cfg['task']['forecasting']['future_patch'] + his_num
        assert fu_num == self.cfg['task']['num_patch']
        self.t_uti = [i for i in range(his_num)]
        self.t_mti = [i for i in range(his_num, fu_num)]
        assert self.t_mti == [24], 'Extrapolation only supports last patch inference'

        if num_prompt is None:
            num_prompt = cfg['num_prompt']
        embed_dim, prompt_threshold = cfg['embed_dim'], cfg['prompt_threshold']
        self.u_prompt = nn.Embedding(num_prompt, embed_dim)
        nn.init.kaiming_uniform_(self.u_prompt.weight, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        self.m_prompt = nn.Embedding(num_prompt, embed_dim)
        # self.m_prompt = nn.Parameter(torch.randn(1, 1, 1, embed_dim))
        nn.init.kaiming_uniform_(self.m_prompt.weight, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        self.prompt_threshold = prompt_threshold

    def forward(self, patches, **kwargs):
        """

        :param patches: [B, N, P, D]
        :return: prompted_patches
        """
        s_mti = kwargs['s_mti']
        s_uti = kwargs['s_uti']

        score_map = torch.sigmoid(torch.matmul(patches, self.u_prompt.weight.t()))  # [B, N, P, num_prompt]
        score_map = score_map * (score_map > self.prompt_threshold).float()
        score_map[:, s_mti] = 0
        score_map[:, :, self.t_mti] = 0
        prompted_patches = patches + torch.sum(score_map.unsqueeze(-1) * self.u_prompt.weight, dim=-2)

        score_map = torch.sigmoid(torch.matmul(patches, self.m_prompt.weight.t()))  # [B, N, P, num_prompt]
        score_map = score_map * (score_map > self.prompt_threshold).float()
        score_map_copy = score_map.clone()
        score_map_copy[:, s_mti] =  0
        score_map_copy[:, :, self.t_mti] = 0
        score_map = score_map - score_map_copy
        prompted_patches = prompted_patches + torch.sum(score_map.unsqueeze(-1) * self.m_prompt.weight, dim=-2)

        return prompted_patches

class EmbeddingPrompt(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        embed_dim, num_prompt = cfg['embed_dim'], cfg['num_prompt']
        self.prompt = nn.Parameter(torch.randn(embed_dim))
        nn.init.uniform_(self.prompt, -.02, .02)

    def forward(self, patches):
        """
        :param patches: [B, N, P, D]
        :return: prompted_patches
        """
        patches = patches + self.prompt
        return patches