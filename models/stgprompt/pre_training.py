from typing import List

import utils.util
from models import init_net, BaseModel
import torch

from .stgp.patch import PatchEmbedding
from .stgp.stg_encoder import STGEncoder
from .stgp.stg_decoder import STGDecoder, PredictionHead
import functools

class PreTraining(BaseModel):

    def __init__(self, opt, cfg):
        super().__init__(opt, cfg)
        """
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['mae']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['Patch_Emb', 'STG_Encoder', 'STG_Decoder', 'Head']
        self.training_model_names = ['Patch_Emb', 'STG_Encoder', 'STG_Decoder', 'Head']

        # specify metrics you want to evaluate the model. The training/test scripts will call functions in order:
        # <BaseModel.compute_metrics> compute metrics for current batch
        # <BaseModel.get_current_metrics> compute and return mean of metrics, clear evaluation cache for next evaluation
        self.metric_names = ['MAE']

        # STGPrompt has four training stages : pre-training, domain_prompting, task_prompting, fine-tune
        assert opt.stage == 'pre_training'

        # define networks. The model variable name should begin with 'self.net'
        self.netPatch_Emb = PatchEmbedding(cfg)
        self.netPatch_Emb = init_net(self.netPatch_Emb, opt.init_type, opt.init_gain, opt.gpu_ids)  # initialize parameters, move to cuda if applicable
        self.netSTG_Encoder = STGEncoder(cfg)
        self.netSTG_Encoder = init_net(self.netSTG_Encoder, opt.init_type, opt.init_gain, opt.gpu_ids)  # initialize parameters, move to cuda if applicable
        self.netSTG_Decoder = STGDecoder(cfg)
        self.netSTG_Decoder = init_net(self.netSTG_Decoder, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.netHead = PredictionHead(cfg)
        self.netHead = init_net(self.netHead, opt.init_type, opt.init_gain, opt.gpu_ids)
        # self.load_pretrained_networks(self.model_names, cfg['checkpoint_stamp'])
        self.model_names = list(set(self.training_model_names) | set(self.model_names))

        # define loss functions
        if self.isTrain:
            self.criterion = functools.partial(utils.util.masked_mae, null_val=0)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = self.optimizer_arrange()
            self.optimizers.append(self.optimizer)
        self.patch_size = cfg['task']['patch_size']
        self.num_patch = cfg['task']['num_patch']
        assert cfg['dropout'] > 0, 'dropout should be lager than zero in pre-training'
                
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
        # to reduce loading time, store static data in memory
        static = {}
        self.dataset_name = input['dataset_name']
        for key in input['static'].keys():
            if not input['dataset_name'] in static.keys():
                static[input['dataset_name']] = {}
            if type(input['static'][key]) == torch.Tensor:
                static[input['dataset_name']][key] = input['static'][key].to(self.device)

        self.in_degree = static[input['dataset_name']]['in_degree']
        self.out_degree = static[input['dataset_name']]['out_degree']
        self.spatial_pos = static[input['dataset_name']]['spatial_pos']
        self.mean = input['static']['mean']
        self.std = input['static']['std']

        self.X = input['X'].to(self.device)  # [B, N, L, D]
        self.feat = input['feat'].to(self.device)  # [B, N, L, D]

    def forward(self, training=True):
        patches, feat_patch = self.netPatch_Emb(self.X, self.feat)  # [B, N, P, D]
        encoded_patches, t_uti, t_mti, s_uti, s_mti = \
            self.netSTG_Encoder(patches, self.spatial_pos, self.in_degree, self.out_degree, feat_patch,
                                random_mask=True)
        decoded_patches = self.netSTG_Decoder(encoded_patches, self.spatial_pos, feat_patch)  # [B, D, N, P]
        self.decoded_x = self.netHead(decoded_patches)  # [B, N, L, 1]

        B, N = self.decoded_x.shape[:2]
        self.X = self.X.reshape(B, -1, self.num_patch, self.patch_size)
        self.decoded_x = self.decoded_x.reshape(B, -1, self.num_patch, self.patch_size)
        self.t_uti, self.s_uti = t_uti, s_uti

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

    def cache_results(self):
        self.X = self.mask_unmasked_patches(self.X * self.std + self.mean, self.t_uti, self.s_uti)
        self.decoded_x = self.mask_unmasked_patches(self.decoded_x * self.std + self.mean, self.t_uti, self.s_uti)
        self._add_to_cache(self.dataset_name+'_decoded', self.decoded_x)
        self._add_to_cache(self.dataset_name+'_X', self.X)

    def compute_metrics(self):
        # find datasets
        dataset_name = []
        for key in self.results.keys():
            name = key.split('_decoded')
            if len(name) > 1:
                dataset_name.append(name[0])
        mae_list = []
        for name in dataset_name:
            mae_list.append(self.criterion(torch.from_numpy(self.results[name+'_decoded']),
                                           torch.from_numpy(self.results[name+'_X'])).item())
            print('Dataset {} : MAE : {:.3f}'.format(str(name), mae_list[-1]))
        self.metric_MAE = sum(mae_list) / len(mae_list)

    def optimize_parameters(self):
        self.set_requires_grad([self.netPatch_Emb, self.netSTG_Encoder, self.netSTG_Decoder, self.netHead], True)
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
