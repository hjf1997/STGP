from typing import List

import utils.util
from models import init_net, BaseModel
import torch
import numpy as np
from copy import deepcopy

from .stgp.patch import PatchEmbedding
from .stgp.stg_encoder import STGEncoder
from .stgp.stg_decoder import STGDecoder, PredictionHead
from .stgp.prompter import AttentionPrompt, EmbeddingPrompt, AttentionPromptForecasting, AttentionPromptKriging, \
    AttentionPromptExtrapolation
import functools
import os


class TaskPrompting(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # modify options for the model
        return parser

    def __init__(self, opt, cfg):
        super().__init__(opt, cfg)
        """
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.cfg = cfg
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['mae']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['Patch_Emb', 'STG_Encoder', 'Domain_SPrompter', 'Domain_TPrompter', 'STG_Decoder', 'Head']
        self.training_model_names = ['Task_Prompter']
        if opt.stage in ['extrapolation_prompting']:
            # empirically we found fine-tuning the head is helpful for extrapolation
            self.training_model_names.append('Head')
            head_dropout = 0.3
        else:
            head_dropout = None
        # specify metrics you want to evaluate the model. The training/test scripts will call functions in order:
        # <BaseModel.compute_metrics> compute metrics for current batch
        # <BaseModel.get_current_metrics> compute and return mean of metrics, clear evaluation cache for next evaluation
        self.metric_names = ['MAE', 'RMSE', 'MAPE']

        # STGPrompt has four training stages : pre-training, domain_prompting, task_prompting, fine-tune
        assert opt.stage in ['forecasting_prompting', 'kriging_prompting', 'extrapolation_prompting']

        # define networks. The model variable name should begin with 'self.net'
        cfg['dropout'] = 0
        self.netPatch_Emb = PatchEmbedding(cfg)
        self.netPatch_Emb = init_net(self.netPatch_Emb, opt.init_type, opt.init_gain, opt.gpu_ids)  # initialize parameters, move to cuda if applicable
        self.netSTG_Encoder = STGEncoder(cfg)
        self.netSTG_Encoder = init_net(self.netSTG_Encoder, opt.init_type, opt.init_gain, opt.gpu_ids)  # initialize parameters, move to cuda if applicable
        self.netSTG_Decoder = STGDecoder(cfg)
        self.netSTG_Decoder = init_net(self.netSTG_Decoder, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.netDomain_TPrompter = AttentionPrompt(cfg)
        self.netDomain_TPrompter = init_net(self.netDomain_TPrompter, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.netDomain_SPrompter = AttentionPrompt(cfg)
        self.netDomain_SPrompter = init_net(self.netDomain_SPrompter, opt.init_type, opt.init_gain, opt.gpu_ids)
        if opt.stage == 'forecasting_prompting':
            self.netTask_Prompter = AttentionPromptForecasting(cfg, num_prompt=cfg['num_task_prompt'])
        elif opt.stage == 'kriging_prompting':
            self.netTask_Prompter = AttentionPromptKriging(cfg, num_prompt=cfg['num_task_prompt'])
        elif opt.stage == 'extrapolation_prompting':
            self.netTask_Prompter = AttentionPromptExtrapolation(cfg, num_prompt=cfg['num_task_prompt'])

        self.netTask_Prompter = init_net(self.netTask_Prompter, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.netHead = PredictionHead(cfg, dropout=head_dropout)
        self.netHead = init_net(self.netHead, opt.init_type, opt.init_gain, opt.gpu_ids)
        if self.opt.phase != 'test':
            self.load_pretrained_networks(self.model_names, cfg['checkpoint_stamp'])

        self.model_names = list(set(self.training_model_names) | set(self.model_names))
        if self.opt.phase == 'test':
            self.load_pretrained_networks(self.model_names, self.opt.checkpoint_stamp, load_epoch='best')

        # define loss functions
        if self.isTrain:
            self.criterion = functools.partial(utils.util.masked_mae, null_val=0)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = self.optimizer_arrange()
            self.optimizers.append(self.optimizer)
        self.patch_size = cfg['task']['patch_size']
        self.num_patch = cfg['task']['num_patch']
        self.future_patch = cfg['task']['forecasting']['future_patch']

    def load_pretrained_networks(self, networks, dir_name, load_epoch='best'):
        """
        :param networks: List: name of networks to be loaded
        :param dir_name: directory names of trained networks
        :return:
        """
        if not isinstance(networks, List):
            networks = [networks]
        for name in networks:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the %s model %s from %s' % (load_epoch, name, dir_name))
                load_dir = os.path.join(self.opt.checkpoints_dir, dir_name)
                load_path = os.path.join(load_dir, load_epoch + '_net_%s.pth' % name)
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                net.load_state_dict(state_dict)

    def optimizer_arrange(self):
        params = []
        for model in self.training_model_names: params += list(getattr(self, 'net' + model).parameters())
        return torch.optim.Adam(params, lr=self.opt.lr, betas=(0.9, 0.999))

    def set_input(self, input):
        """
        parse input for one epoch. data should be stored as self.xxx which would be adopted in self.forward().
        :param input: dict
        :return: None
        """
        self.dataset_name = input['dataset_name']

        self.in_degree = input['static']['in_degree'].to(self.device)
        self.out_degree = input['static']['out_degree'].to(self.device)
        self.spatial_pos = input['static']['spatial_pos'].to(self.device)
        self.mean = input['static']['mean']
        self.std = input['static']['std']

        self.X = input['X'].to(self.device)  # [B, N, L, D]
        self.feat = input['feat'].to(self.device)  # [B, N, L, D]

    def mask_unmasked_patches(self, patches, t_uti, s_uti):
        if t_uti is None: t_uti = [i for i in range(patches.shape[2])]
        if s_uti is None: s_uti = [i for i in range(patches.shape[1])]
        s_mask = torch.zeros_like(self.X)
        s_mask[:, s_uti] = 1
        t_mask = torch.zeros_like(self.X)
        t_mask[:, :, t_uti] = 1
        mask_merge = s_mask * t_mask
        patches = patches * (1 - mask_merge)  # unmasked patches are replaced by zeros
        # loss / metric will not compute on zeros
        return patches

    def backward(self):
        assert self.decoded_x.shape == self.X.shape
        # only compute masked tokens
        self.X = self.mask_unmasked_patches(self.X, self.t_uti, self.s_uti)
        self.decoded_x = self.mask_unmasked_patches(self.decoded_x, self.t_uti, self.s_uti)
        self.loss_mae = self.criterion(self.decoded_x, self.X)
        self.loss_mae.backward()

    def optimize_parameters(self):
        self.set_requires_grad([self.netPatch_Emb, self.netSTG_Encoder, self.netDomain_SPrompter, self.netDomain_TPrompter], False)
        self.set_requires_grad([self.netSTG_Decoder, self.netTask_Prompter, self.netHead], True)

        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
