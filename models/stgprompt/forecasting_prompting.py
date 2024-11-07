from typing import List

import utils.util
import numpy as np
from .task_prompting import TaskPrompting


class ForecastingPrompting(TaskPrompting):

    def __init__(self, opt, cfg):
        super().__init__(opt, cfg)

    def forward(self, training=True):
        patches, feat_patch = self.netPatch_Emb(self.X, self.feat)  # [B, N, P, D]
        encoded_patches, t_uti, t_mti, s_uti, s_mti = \
            self.netSTG_Encoder(patches, self.spatial_pos, self.in_degree, self.out_degree, feat_patch,
                                tprompter=self.netDomain_TPrompter,
                                sprompter=self.netDomain_SPrompter,
                                random_mask=False,
                                task='forecasting')
        decoded_patches = self.netSTG_Decoder(encoded_patches, self.spatial_pos, feat_patch,
                                             prompter=self.netTask_Prompter)  # [B, N, L, 1]
        self.decoded_x = self.netHead(decoded_patches)  # [B, N, L, 1]

        B, N = self.decoded_x.shape[:2]
        self.X = self.X.reshape(B, -1, self.num_patch, self.patch_size)
        self.decoded_x = self.decoded_x.reshape(B, -1, self.num_patch, self.patch_size)
        self.t_uti, self.s_uti = t_uti, s_uti

    def cache_results(self):
        self.X = self.mask_unmasked_patches(self.X * self.std + self.mean, self.t_uti, self.s_uti)
        self.decoded_x = self.mask_unmasked_patches(self.decoded_x * self.std + self.mean, self.t_uti, self.s_uti)
        mae = utils.util.masked_mae_item(self.decoded_x, self.X, null_val=0).unsqueeze(0)  # [1, P*D]
        mape = utils.util.masked_mape_item(self.decoded_x, self.X, null_val=0).unsqueeze(0)  # [1, P*D]
        rmse = utils.util.masked_rmse_item(self.decoded_x, self.X, null_val=0).unsqueeze(0)  # [1, P*D]
        self._add_to_cache('mae', mae[:, -self.patch_size*self.future_patch:])
        self._add_to_cache('mape', mape[:, -self.patch_size*self.future_patch:])
        self._add_to_cache('rmse', rmse[:, -self.patch_size*self.future_patch:])

    def compute_metrics(self):
        mae = np.mean(self.results['mae'], axis=0)   # [L]
        mape = np.mean(self.results['mape'], axis=0)   # [L]
        rmse = np.mean(self.results['rmse'], axis=0)   # [L]
        self.metric_MAE, self.metric_MAPE, self.metric_RMSE = np.mean(mae), np.mean(mape), np.mean(rmse)
        self.metric_MAE_list, self.metric_MAPE_list, self.metric_RMSE_list = mae, mape, rmse

