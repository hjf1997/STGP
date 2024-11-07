# implemented by p0werHu

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils.logger import Logger
import time
from tqdm import tqdm

if __name__ == '__main__':
    opt, config = TestOptions().parse()   # get training options
    visualizer = Logger(opt)  # create a visualizer that display/save and plots
    dataset = create_dataset(opt, config)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of samples in the dataset.
    print('The number of testing samples = %d' % dataset_size)

    model = create_model(opt, config)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()

    val_start_time = time.time()
    for i, data in tqdm(enumerate(dataset)):  # inner loop within the test dataset
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.test()
        model.cache_results()  # store current batch results
    t_val = time.time() - val_start_time

    model.compute_metrics()
    metrics = model.get_current_metrics()
    visualizer.print_current_metrics(-1, 0, metrics, t_val)
    if opt.stage in ['forecasting_prompting', 'extrapolation_prompting']:
        metrics_horizontal = model.get_current_horizontal_metrics()
        visualizer.print_current_horizontal_metrics(metrics_horizontal)
