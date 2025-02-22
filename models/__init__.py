"""This package contains modules related to objective functions, optimizations, and network architectures.
To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel inherited from BaseModel.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
    -- <set_input>:                     unpack data from dataset and apply preprocessing.
    -- <forward>:                       produce intermediate results.
    -- <optimize_parameters>:           calculate loss, gradients, and update network weights.
    -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
In the function <__init__>, you need to define four lists:
    -- self.loss_names (str list):          specify the training losses that you want to plot and save.
    -- self.model_names (str list):         define networks used in our training.
    -- self.visual_names (str list):        specify the images that you want to display and save.
    -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an usage.
Now you can use the model class by specifying flag '--model dummy'.
See our template model class 'template_model.py' for more details.
"""

import importlib
from models.base_model import BaseModel
from torch.nn import init
import torch
import glob

def find_model_using_name(dir_name, file_name):
    """
    
    :param dir_name: model files dictionary
    :param file_name: model file to load
    :return: 
    """"""

    """
    model = None
    patterns = f'models/{dir_name}/{file_name}.py'
    model_name = patterns.replace('/', '.')[:-3]  # remove .py
    try:
        modellib = importlib.import_module(model_name)
        target_model_name = model_name.split('.')[-1].replace('_', '')
        for name, cls in modellib.__dict__.items():
            if name.lower() == target_model_name.lower() and issubclass(cls, BaseModel):
                model = cls
    except Exception as e:
        print(e)
    if model is None:
        raise ModuleNotFoundError(f"File {model_name}.py not found in models folder.")
    return model

# def get_option_setter(model_name):
#     """Return the static method <modify_commandline_options> of the model class."""
#     model_class = find_model_using_name(model_name)
#     return model_class.modify_commandline_options


def create_model(opt, config):
    """Create a model given the option.
    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'
    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model_cfg = config['model']
    model_cfg['task'] = config['task']

    if model_cfg['name'] == 'stgprompt':
        model = find_model_using_name(model_cfg['name'], opt.stage)
    else:
        model = find_model_using_name(model_cfg['name'], model_cfg['name']) # todo hard coding part
    instance = model(opt, model_cfg)
    print("model [%s] was created" % model_cfg['name'])
    return instance


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net