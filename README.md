# Prompt-Based Spatio-Temporal Graph Transfer Learning

## Dependencies
- Python 3.11
- PyTorch 2.0

## Datasets
The original dataset is available at [Google Drive](https://drive.google.com/drive/folders/1UrKTgR27YmP9PjJ-FWv4SCDH3zUxtc5R?usp=share_link).
Download the data and save the folders in `./dataset`.
Note that the test_node.npy file in each dataset folder in very important for kriging and extrapolation.
Please do not delete them, as different divisions lead to different results.

## Structure of the repository
Our model is implemented and trained in a general deep learning framework, automatically training and evaluating the models.
The structure of the repository is as follows:
- `./checkpoint`: Trained models are saved here.
- `./configs`: The default configs of the four datasets are set in `./configs`.
- `./dataset`: The processed data.
- `./dataset`: Dataset files.
- `./models`: The model of the project.
- `./options`: The framework options for training and evaluation
- `./utils`: The utility functions.

## Config Parameters
We list important model configs in the table.

| setting   | values                                 | help                                                  |
|-----------|----------------------------------------|-------------------------------------------------------|
| target_domain     | metr-la; pems-bay; chengdu_m; shenzhen | target dataset, the rest three are source datasets    |
| target_training_size | 3                                      | available training days for target domain             |
| checkpoint_stamp | pre_20231227T144938                    | time stamp of the pre-trained / domain-prompted model |

When domain or task prompting the model, please fill in the corresponding checkpoint stamp of the previous training stage to the [checkpoint_stamp] key in the config file.

For framework configs, please refer to the `./options` folder.

## Train STGP
### Pre-Training
To pre-train the model, run the following command:
```
bash ./pre_training.sh [CONFIG] [BATCH_SIZE] [GPU_ID] [SEED] 
```
The framework will create a folder with a time stamp to store the pre-trained checkpoints.

Copy the time stamp and paste it to the [checkpoint_stamp] key in the corresponding config file to use the pre-trained model for domain prompting

### Domain Prompting
The transductive and inductive settings have different domain prompting configs.
As for the inductive setting, model training has no access to the unobserved nodes.
To enable or disable the inductive setting, set the [inductive] key in the config file to True or False.
```
bash ./domain_prompting.sh [CONFIG] [BATCH_SIZE] [GPU_ID] [SEED] 
```

### Task Prompting
To train the model with task prompting, run the following command:
```
bash ./forecasting_prompting.sh [CONFIG] [BATCH_SIZE] [GPU_ID] [SEED] 
```
```
bash ./kriging_prompting.sh [CONFIG] [BATCH_SIZE] [GPU_ID] [SEED] 
```
```
bash ./extrapolation_prompting.sh [CONFIG] [BATCH_SIZE] [GPU_ID] [SEED] 
```

## Evaluation
After training of task prompting, the framework will create a run_test.sh file to evaluate the model.
By default, the framework will run the evaluation file automatically.
If you want to evaluate the model manually, run the following command:
```
bash ./run_test.sh
```

## Reproduce Results
We saved the pretrained checkpoints for our STGP at [Google Drive](https://drive.google.com/drive/folders/1GATpd5GlcFFk8ZZlPzgCw6CLZKf6H6Np?usp=sharing).
Download the checkpoints files and put them under [checkpoints](checkpoints) folder.
Each checkpoint file contains a *run_test.sh* script to calculate metrics.
```bash[README.md](..%2F..%2F..%2F..%2FDesktop%2FREADME.md)
chmod u+x run_test.sh
./run_test.sh
```
The numerical results will be saved at *metrics.txt*.