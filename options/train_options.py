#  implemented by p0werHu
# time: 5/6/2021

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        self.isTrain = True

        parser = BaseOptions.initialize(self, parser)
        # Loss visualization parameters
        parser.add_argument('--enable_neptune',  action='store_true', help='use neptune platform for experiment tracking')
        parser.add_argument('--neptune_project', type=str, default='', help='neptune project name')
        parser.add_argument('--neptune_token', type=str, default='', help='neptune api token')
        parser.add_argument('--print_freq', type=int, default=20, help='frequency of showing training results on console')
        # network saving and loading parameters
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_best', action='store_true', help='save best model')
        parser.add_argument('--eval_epoch_freq', type=int, default=1, help='epoch frequency of showing validation results on console')
        parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=0, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--early_stop', type=int, default=3, help='early stopping patience')
        parser.add_argument('--enable_val', action='store_true', help='evaluate model during training')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='cosine', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        return parser
