from typing import List

import utils.util
from models import init_net, BaseModel
import torch
import numpy as np
from .task_prompting import TaskPrompting


class ExtrapolationPrompting(TaskPrompting):

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

    def set_input(self, input):
        """
        parse input for one epoch. data should be stored as self.xxx which would be adopted in self.forward().
        :param input: dict
        :return: None
        """
        super().set_input(input)
        if self.opt.phase == 'train' and self.cfg['task']['inductive']:
            node_index = np.arange(self.X.shape[1])
            num_train = int(self.X.shape[1] * (1 - self.cfg['task']['kriging']['test_node_ratio']))
            self.train_node_index = np.random.choice(node_index, num_train, replace=False).tolist()
            self.test_node_index = np.setdiff1d(node_index, self.train_node_index).tolist()
        else:
            self.test_node_index = input['static']['test_node_index']
            self.train_node_index = input['static']['train_node_index']
            assert self.X.shape[1] == len(self.test_node_index) + len(self.train_node_index)

    def forward(self, training=True):
        patches, feat_patch = self.netPatch_Emb(self.X, self.feat)  # [B, N, P, D]
        encoded_patches, t_uti, t_mti, s_uti, s_mti = \
            self.netSTG_Encoder(patches, self.spatial_pos, self.in_degree, self.out_degree, feat_patch,
                                tprompter=self.netDomain_TPrompter,
                                sprompter=self.netDomain_SPrompter,
                                random_mask=False,
                                task='extrapolation',
                                s_uti=self.train_node_index,
                                s_mti=self.test_node_index)
        decoded_patches = self.netSTG_Decoder(encoded_patches, self.spatial_pos, feat_patch,
                                             prompter=self.netTask_Prompter, s_mti=s_mti, s_uti=s_uti)  # [B, N, L, 1]
        self.decoded_x = self.netHead(decoded_patches)  # [B, N, L, 1]

        B, N = self.decoded_x.shape[:2]
        self.X = self.X.reshape(B, -1, self.num_patch, self.patch_size)
        self.decoded_x = self.decoded_x.reshape(B, -1, self.num_patch, self.patch_size)
        self.t_uti, self.s_uti = t_uti, s_uti
        self.t_mti, self.s_mti = t_mti, s_mti

    def cache_results(self):
        self.X = self.X[:, self.s_mti][:, :, self.t_mti] * self.std + self.mean
        self.decoded_x = self.decoded_x[:, self.s_mti][:, :, self.t_mti] * self.std + self.mean
        B, N = self.decoded_x.shape[:2]
        self.X = self.X.reshape(B, N, -1)  # [B, N, P*L]
        self.decoded_x = self.decoded_x.reshape(B, N, -1)  # [B, N, P*L]
        self._add_to_cache('X', self.X)
        self._add_to_cache('decoded_x', self.decoded_x)

    def compute_metrics(self):
        X = self.results['X']
        X = torch.from_numpy(X)
        decoded_x = self.results['decoded_x']
        decoded_x = torch.from_numpy(decoded_x)
        mae, mape, rmse = [], [], []
        for i in range(decoded_x.shape[-1]):
            mae.append(utils.util.masked_mae(decoded_x[..., i], X[..., i], null_val=0))
            mape.append(utils.util.masked_mape(decoded_x[..., i], X[..., i], null_val=0))
            rmse.append(utils.util.masked_rmse(decoded_x[..., i], X[..., i], null_val=0))
            # print('Horizon {} : MAE : {:.3f}, RMSE : {:.3f}, MAPE: {:.3f}'.format(i, mae[-1], rmse[-1], mape[-1]))
        self.metric_MAE, self.metric_MAPE, self.metric_RMSE = np.mean(mae), np.mean(mape), np.mean(rmse)
        self.metric_MAE_list, self.metric_MAPE_list, self.metric_RMSE_list = mae, mape, rmse
