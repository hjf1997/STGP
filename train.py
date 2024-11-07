# implemented by p0werHu
import os
import time
from options.train_options import TrainOptions
from options.val_options import Valptions
from data import create_dataset
from models import create_model
from utils.logger import Logger
import subprocess
from tqdm import tqdm

if __name__ == '__main__':
    opt, config = TrainOptions().parse()   # get training options
    visualizer = Logger(opt)  # create a visualizer that display/save and plots
    dataset = create_dataset(opt, config)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of samples in the dataset.
    print('The number of training batches = %d' % dataset_size)

    if opt.enable_val:
        val_opt, _ = Valptions().parse()  # get validation options
        val_dataset = create_dataset(val_opt, config)  # create a validation dataset given opt.dataset_mode and other options
        val_dataset_size = len(val_dataset)  # get the number of samples in the dataset.
        print('The number of validation batches = %d' % val_dataset_size)

    model = create_model(opt, config)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    best_metric = float('inf')  # best metric
    early_stop_trigger = 0
    for epoch in range(0, opt.n_epochs + opt.n_epochs_decay):
        epoch_start_time, iter_data_time = time.time(), time.time()
        model.train()
        for i, data in enumerate(dataset):  # inner loop within one epoch
            data_loader_time = time.time()
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            iter_start_time = time.time()  # timer for computation per iteration
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            total_iters += 1
            if total_iters % opt.print_freq == 0:   # display
                losses = model.get_current_losses()
                t_comp = time.time() - iter_start_time  # time for one iter optimization
                t_data_gpu = iter_start_time - data_loader_time  # time for one iter data loading
                t_data_loader = data_loader_time - iter_data_time
                visualizer.print_current_losses(data['dataset_name'][:3], epoch, total_iters, losses, t_comp, t_data_loader, t_data_gpu)
            iter_data_time = time.time()
        epoch_end_time = time.time()

        if opt.enable_val and (epoch + 1) % opt.eval_epoch_freq == 0:
            print('Start evaluation on validation set')
            model.eval()
            val_start_time = time.time()
            for i, data in tqdm(enumerate(val_dataset), total=val_dataset_size):  # inner loop within one epoch
                model.set_input(data)  # unpack data from dataset and apply preprocessing
                model.test()
                model.cache_results()  # store current batch results
            t_val = time.time() - val_start_time
            model.compute_metrics()
            metrics = model.get_current_metrics()
            visualizer.print_current_metrics(epoch, total_iters, metrics, t_val)

            if opt.stage in ['forecasting_prompting', 'extrapolation_prompting']:
                metrics_horizontal = model.get_current_horizontal_metrics()
                visualizer.print_current_horizontal_metrics(metrics_horizontal)

            if best_metric > metrics['MAE']:
                print('saving the best model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('best')
                best_metric = metrics['MAE']
                early_stop_trigger = 0
            else:
                early_stop_trigger += opt.eval_epoch_freq
            model.clear_cache()

            # check early stopping
            early_stopping_threshold = opt.early_stop
            if early_stop_trigger >= early_stopping_threshold:
                print('Trigger early stopping!')
                break

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.n_epochs + opt.n_epochs_decay, epoch_end_time - epoch_start_time))
        new_lr = model.update_learning_rate()  # update learning rates in the beginning of every epoc

    if opt.stage in ['forecasting_prompting', 'kriging_prompting', 'extrapolation_prompting']:
        print('Run evaluation')
        with open(os.path.join(model.save_dir,'run_test.sh'), 'w') as f:
            cmd = 'python test.py --config {} --batch_size {} --gpu_ids {} --stage {} --checkpoint_stamp {}'.format(
                opt.config_file,
                opt.batch_size,
                opt.gpu_ids[0],
                opt.stage,
                opt.checkpoint_name)
            f.write(cmd)
        #
        os.system('chmod u+x '+ os.path.join(model.save_dir,'run_test.sh'))
        subprocess.Popen(os.path.join(model.save_dir, 'run_test.sh'), shell=True)
