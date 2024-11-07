# implemented by p0werHu
import datetime

from data.base_dataset import BaseDataset
import numpy as np
import torch


class TrafficDataset(BaseDataset):
    """
    Note that the beijing air quality dataset contains a lot of missing values, we need to handle this explicitly.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        return parser

    def __init__(self, opt, config):
        super(TrafficDataset, self).__init__(opt, config)
        self.opt = opt
        self.cfg = config
        self.task_config = config['task']
        self.t_len = self.task_config['patch_size'] * self.task_config['num_patch']
        X, feat, self.A, time_list, self.mean, self.std = self.load_data(config)
        self.raw_data = {
            'X': X,
            'feat': feat,
            'time': time_list
        }
        # task_name here is only for baselines
        task_name = config['task']['task_name'] if 'task_name' in config['task'].keys() else None # this is for baselines
        # domain_prompting here only for inductive kriging and extrapolation
        if (opt.stage in ['domain_prompting', 'kriging_prompting', 'extrapolation_prompting']
                or task_name in ['kriging', 'extrapolation']):
            test_node_index = self.get_node_division(config['test_node_path'], X.shape[0],
                                                     config['task']['kriging']['test_node_ratio'])
            self.train_node_index = np.setdiff1d(np.arange(self.raw_data['X'].shape[0]), np.array(test_node_index)).tolist()
            self.test_node_index = test_node_index
            # inductive setting for kriging and extrapolation
            if opt.phase == 'train' and self.task_config['inductive']:
                self.raw_data = {
                    'X': X[self.train_node_index],
                    'feat': feat[self.train_node_index],
                    'time': time_list
                }
                self.A = self.A[self.train_node_index][:, self.train_node_index]
                print('The number of nodes for training: {}'.format(self.raw_data['X'].shape[0]))

    def load_data(self, cfg):

        A = np.load(cfg['adjacency_matrix_path'])
        X = np.load(cfg['dataset_path'])
        # [L, N, 4]
        # [:,:,0] : speed, [:,:,1] : some symbol of time?
        # (N, D, L)

        X = X.transpose((1, 2, 0))  # [N, 4, L]
        X = torch.tensor(X, dtype=torch.double)
        X = torch.cat((X[:, 0, :].unsqueeze(1), X[:, -1, :].unsqueeze(1)), dim=1)

        # Interpolation. Chengdu and Shenzhen interpolated to 5min level.
        if cfg['name'] in ['chengdu_m', 'shenzhen']: interp = True
        else: interp = False

        if interp:
            interp_X = torch.nn.functional.interpolate(X, size=2 * X.shape[-1] - 1, mode='linear', align_corners=True)
            interp_X = torch.cat((interp_X[:, :, :1], interp_X), dim=-1)
            interp_X[:, 1, 0] = ((interp_X[:, 1, 1] - 1) + 2016) % 2016  # 2016 is the week slot (from 2014 -> 2015)
            X = interp_X

        week_feat = X[:, 1, :].unsqueeze(1).numpy()  # [N, L, 1]
        week_feat = week_feat.transpose((0, 2, 1))
        X = X[:, 0, :].unsqueeze(1)
        # data norm
        X = X.numpy().astype(np.float64)
        mean = np.mean(X)
        X = X - mean
        std = np.std(X)
        X = X / std
        X = X.transpose((0, 2, 1))

        # time info
        # start_time = datetime.datetime.strptime(cfg['start_time'], '%Y-%m-%d %H:%M:%S')
        start_time = cfg['start_time']
        time_list = [np.datetime64(start_time) + np.timedelta64(t * 5, 'm') for t in range(X.shape[1])]
        time_list = np.array(time_list)
        time_list = ((time_list - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')).astype(np.int64)

        # day of time/ day of week features
        feat = self.load_feat(week_feat)
        if self.opt.stage == 'pre_training' and self.opt.phase == 'train':
            # 8 training 2 val
            train_length = int(X.shape[1] * 0.8)
            X = X[:, :train_length]
            feat = feat[:, :train_length]
            time_list = time_list[:train_length]
        elif self.opt.stage == 'pre_training' and self.opt.phase == 'val':
            train_length = int(X.shape[1] * 0.8)
            self.test_slide_step = 12  # 1 hour
            X = X[:, train_length:]
            feat = feat[:, train_length:]
            time_list = time_list[train_length:]
        elif self.opt.stage == 'domain_prompting' and self.opt.phase == 'train':
            length = self.task_config['target_training_size'] * 288
            X = X[:, :length]
            feat = feat[:, :length]
            time_list = time_list[:length]
        elif self.opt.stage == 'domain_prompting' and self.opt.phase == 'val':
            length_s = self.task_config['target_training_size'] * 288
            length_end = self.task_config['target_training_size'] * 288 * 2
            X = X[:, length_s:length_end]
            time_list = time_list[length_s:length_end]
            feat = feat[:, length_s:]
        elif self.opt.stage in ['forecasting_prompting', 'kriging_prompting', 'extrapolation_prompting'] and self.opt.phase == 'train':
            length = self.task_config['target_training_size'] * 288
            X = X[:, :length]
            feat = feat[:, :length]
            time_list = time_list[:length]
        elif self.opt.stage in ['forecasting_prompting', 'kriging_prompting', 'extrapolation_prompting'] and self.opt.phase in ['val']:
            self.test_slide_step = 12
            length_s = self.task_config['target_training_size'] * 288
            length_end = self.task_config['target_training_size'] * 288 * 2
            X = X[:, length_s:length_end]
            time_list = time_list[length_s:length_end]
            feat = feat[:, length_s:]
        elif self.opt.stage in ['forecasting_prompting', 'kriging_prompting', 'extrapolation_prompting'] and self.opt.phase in ['test']:
            self.test_slide_step = 12
            length_s = self.task_config['target_training_size'] * 288 * 2
            X = X[:, length_s:]
            time_list = time_list[length_s:]
            feat = feat[:, length_s:]
        else:
            raise NotImplementedError('Dataset for stage {} and phase {} is invalid'.format(self.opt.stage, self.opt.phase))

        return X, feat, A, time_list, mean, std

    def load_feat(self, week_feat):

        # [N, L, 1]
        dow = np.floor(week_feat / 288)
        assert dow.max() < 7
        tod = np.floor(week_feat % 288 / 12)  # to hours
        assert tod.max() < 24
        feat = np.concatenate((tod, dow), axis=-1) # order cannot change
        return feat

    def load_static_data(self):
        spatial_pos = torch.from_numpy(self.A).float()  # [N, N]
        in_degree = torch.from_numpy(np.sum(self.A > 0, axis=0))  # [N]
        out_degree = torch.from_numpy(np.sum(self.A > 0, axis=1))  # [N]
        static = {'spatial_pos': spatial_pos,
                'in_degree': in_degree,
                'out_degree': out_degree,
                'mean': self.mean,
                'std': self.std}
        if self.test_node_index is not None:
            static['test_node_index'] = self.test_node_index
            static['train_node_index'] = self.train_node_index
        return static
