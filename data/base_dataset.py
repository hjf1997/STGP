"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.
It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random

import torch
import torch.utils.data as data
from abc import ABC, abstractmethod
from data.data_util import *


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.
    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt, config):
        """Initialize the class; save the options in the class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
            config (dict) -- stores all the dataset config
        In this function, we instantiate the following variables:
        raw_data -- a dictionary that stores the raw data:
            'pred' -- target variables of shape (num_nodes, num_time_steps, num_features1)
            'feat' (optional) -- covariates of shape (num_nodes, num_time_steps, num_features2)
            'time' -- time stamps of shape (num_time_steps, )
            'missing' -- missing masks of shape (num_nodes, num_time_steps, num_features1)
        A -- adjacency matrix of shape (num_nodes, num_nodes)
        test_node_index -- a numpy array of shape (num_test_nodes, ) that stores the indices of test nodes
        train_node_index -- a numpy array of shape (num_train_nodes, ) that stores the indices of train nodes
        """
        self.opt = opt
        self.cfg = config
        self.time_division = {}
        self.raw_data = {}
        self.A = None
        self.t_len = None
        self.test_node_index, self.train_node_index = None, None
        self.test_slide_step = 1

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

    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.opt.phase == 'train':
            length = self.raw_data['X'].shape[1] - self.t_len
        else:
            length = int((self.raw_data['X'].shape[1] - self.t_len) / self.test_slide_step)
        return length

    def __getitem__(self, t_index):
        """Return a data point and its metadata information.
        Parameters:
            t_index - - a random integer for data indexing
        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        node_index = None
        start_index, end_index = self._get_start_index(t_index, self.t_len, self.opt.phase, self.test_slide_step)
        # start_index = t_index
        # end_index = t_index + self.t_len

        X, feat, _ = BaseDataset._fetch_data_item_from_dict(self.raw_data, start_index, end_index, station_index=node_index)
        time = self.raw_data['time'][start_index:end_index]

        batch_data = {
            'X': X.float(),  # [num_n, time, d_x]
            'time': time,  # [time]
            'feat': feat.float()  # [num_n, time, D]
        }
        return batch_data

    def load_static_data(self):
        pass

    ##################################################
    # utility functions
    ##################################################

    @staticmethod
    def _fetch_data_item_from_dict(data, start_index, end_index, station_index=None):
        """
        Fetch data from the time series
        Args:
            data (dict: {'feat'(optional), 'missing', 'pred'}): time series data dictionary.
            Key feat is optional, depending on the dataset.
            station_index (ndarray): station indexes
            start_index (int): start index of the time series
            end_index (int): end index of the time series

        Returns:
            data_item (tensor): data of the time series
        """
        if station_index is None:
            # return all stations
            station_index = np.arange(data['X'].shape[0])
        pred = torch.from_numpy(data['X'][station_index, start_index:end_index])
        feat = torch.from_numpy(data['feat'][station_index, start_index:end_index]) if 'feat' in data.keys() else None
        missing = torch.from_numpy(data['missing'][station_index, start_index:end_index]) if 'missing' in data.keys() else None
        return pred, feat, missing

    @staticmethod
    def _get_start_index(index, t_len, phase='train', test_slide_step=0):
        """
        Get the start index of the time series
        Training phase: current index + t_len
        Test phase: no overlap between time series
        Args:
            index (int): index of the time series
            t_len (int): length of the time series
            phase (str): phase of the model

        Returns:
            start_index (int): start index of the time series
            start_index (int): end index of the time series
        """
        if phase == 'train':
            start_index = index
            end_index = index + t_len
        else:
            assert test_slide_step > 0
            start_index = index * test_slide_step
            end_index = start_index + t_len
        return start_index, end_index

    def get_node_division(self, test_nodes_path, num_nodes=None, test_node_ratio=1/3):
        if os.path.isfile(test_nodes_path):
            test_nodes = np.load(test_nodes_path)
        else:
            print('No testing nodes. Randomly divide nodes for testing!')
            rand = np.random.RandomState(4)  # Fixed random output
            test_nodes = np.sort(rand.choice(list(range(0, num_nodes)), int(num_nodes * test_node_ratio + 0.5), replace=False))
            np.save(test_nodes_path, test_nodes)
        test_nodes = test_nodes.tolist()
        return test_nodes
