"""This package includes all the modules related to data loading and preprocessing
 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
import random


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".
    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt, config):
    """Create a dataset given the option.
    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'
    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = FewShotDatasetDataLoader(opt, config)
    # data_loader = CustomDatasetDataLoader(opt)
    # dataset = data_loader.load_data()
    return data_loader


class FewShotDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, config):
        """Initialize this class
        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        self.dataloaders = []
        self.static_data = []
        self.name_list = []
        for name, dataset_cfg in config['data'].items():
            if opt.stage == 'pre_training' and config['task']['target_domain'] == name:
                continue
            if opt.stage != 'pre_training' and config['task']['target_domain'] != name:
                # only target dataset
                continue
            dataset_cfg['name'] = name
            dataset_cfg['task'] = config['task']
            dataset_class = find_dataset_using_name(dataset_cfg['type'])
            dataset = dataset_class(opt, dataset_cfg)
            print("dataset [%s] for [%s] was created" % (name, opt.stage))
            self.dataloaders.append(torch.utils.data.DataLoader(
                dataset,
                batch_size=opt.batch_size,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.num_threads),
                drop_last=False))
            self.static_data.append(dataset.load_static_data())
            self.name_list.append(name)

    def __len__(self):
        """Return the number of data in the dataset
        Note that here the len means how many batch the dataloader has
        """
        dataset_len = sum([len(dataloader) for dataloader in self.dataloaders])
        return min(dataset_len, self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        # random pick one dataset from the self.dataloaders list
        iterators = [d._get_iterator() for d in self.dataloaders]
        iters_len = len(self.dataloaders)
        empty_list = [False] * iters_len
        yield_cont = 0
        while True:
            idx = random.randint(0, iters_len - 1)
            try:
                loader = iterators[idx]
                batch = next(loader)
                batch['static'] = self.static_data[idx]
                batch['dataset_name'] = self.name_list[idx]
                yield_cont += 1
                if yield_cont > self.opt.max_dataset_size:
                    break
                yield batch
            except StopIteration:
                empty_list[idx] = True
                if all(empty_list):
                    break
                else:
                    continue
