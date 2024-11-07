#  implemented by p0werHu
import argparse
import os
from utils import util
import torch
import models
import data
import time
import yaml
import numpy as np
import random
import sys

class BaseOptions():
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--config_file', type=str, required=True, help='config file name')
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--stage', type=str, required=True, help='[pre_training, domain_prompting, task_prompting, fine-tuning]')
        # dataset parameters
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data.')
        parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
        parser.add_argument('--max_dataset_size', type=int, default=float('inf'), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='best', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--seed', type=int, default=2023, help='random seed for initialization')
        # add you customized parameters below
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        # model_name = opt.model
        # model_option_setter = models.get_option_setter(model_name)
        # parser = model_option_setter(parser, self.isTrain)
        # opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        # dataset_name = opt.dataset_mode
        # dataset_option_setter = data.get_option_setter(dataset_name)
        # parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser and model config
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt, model_config):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Framework Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------\n'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.checkpoint_name)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        # save system error outputs
        logger_file_name = os.path.join(expr_dir, '{}_error.log'.format(opt.phase))
        sys.stderr = Logger(filename=logger_file_name, stream=sys.stdout)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
        opt.checkpoints_dir = os.path.join(opt.checkpoints_dir, opt.config_file)
        if opt.phase == 'test':
            if opt.checkpoint_stamp == '':
                raise RuntimeError('Please specify checkpoint time!')
            else:
                opt.checkpoint_name = opt.checkpoint_stamp # dir name
        else:
            current_time = time.strftime("%Y%m%dT%H%M%S", time.localtime())
            current_time = opt.stage.split('_')[0] + '_' + current_time
            opt.checkpoint_name = current_time

        if opt.phase == 'train':
            expr_dir = os.path.join(opt.checkpoints_dir, opt.checkpoint_name)
            if not os.path.exists(expr_dir): # in case of continuing training
                util.mkdirs(expr_dir)

        # load model configurations from .yaml
        config = None
        if opt.phase == 'test':
            yaml_path = os.path.join(opt.checkpoints_dir, opt.checkpoint_name, 'config.yaml')
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r') as config_file:
                    config = yaml.safe_load(config_file)
            else:
                raise RuntimeError('Cannot find model configuration file in the checkpoint dir.')
        # load model configuration to .yaml and save it to checkpoints_dir
        if opt.phase == 'train':
            yaml_path = os.path.join('configs', opt.config_file + '.yaml')
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r') as config_file:
                    config = yaml.safe_load(config_file)
                with open(os.path.join(opt.checkpoints_dir, opt.checkpoint_name, 'config.yaml'), 'w') as config_file:
                    yaml.safe_dump(config, config_file)
            else:
                raise FileNotFoundError('Cannot find configuration file.')

        if opt.phase != 'val': self.print_options(opt, config)
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.manual_seed_all(opt.seed)
            torch.cuda.set_device(opt.gpu_ids[0])
        self.opt = opt
        return self.opt, config


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.filename = filename

    def write(self, message):
        self.terminal.write(message)
        with open(self.filename, 'a') as log:
            log.write(message)

    def flush(self):
        pass