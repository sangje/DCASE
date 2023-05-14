import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import torch
from trainer.trainer import Task
from tools.config_loader import get_config
from pathlib import Path
from data_handling.DataLoader import get_dataloader

from lightning.pytorch import LightningModule, Trainer, seed_everything


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    parser = argparse.ArgumentParser(description='Settings.')
    parser.add_argument('-n', '--exp_name', default='exp_name', type=str,
                        help='Name of the experiment.')
    parser.add_argument('-d', '--dataset', default='Clotho', type=str,
                        help='Dataset used')
    parser.add_argument('-l', '--lr', default=0.0001, type=float,
                        help='Learning rate')
    parser.add_argument('-c', '--config', default='settings', type=str,
                        help='Name of the setting file.')
    parser.add_argument('-o', '--loss', default='ntxent',  type=str,
                        help='Name of the loss function.')
    parser.add_argument('-f', '--freeze', default='False', type=str,
                        help='Freeze or not.')
    parser.add_argument('-e', '--batch', default=24, type=int,
                        help='Batch size.')
    parser.add_argument('-m', '--margin', default=0.2, type=float,
                        help='Margin value for loss')
    parser.add_argument('-s', '--seed', default=20, type=int,
                        help='Training seed')
    parser.add_argument('-p', '--epochs',default=50, type=int,
                        help='Epoch')

    args = parser.parse_args()

    config = get_config(args.config)

    config.exp_name = args.exp_name
    config.dataset = args.dataset
    config.training.lr = args.lr
    config.training.loss = args.loss
    config.training.freeze = eval(args.freeze)
    config.data.batch_size = args.batch
    config.training.margin = args.margin
    config.training.seed = args.seed
    config.training.epochs = args.epochs

    # Set Up Seed
    seed_everything(config.training.seed, workers=True)

    # Set up Path Names
    folder_name = '{}_freeze_{}_lr_{}_' \
                    'seed_{}'.format(config.exp_name, str(config.training.freeze),
                                                config.training.lr,
                                                config.training.seed)
    config.csv_output_dir = Path('outputs', folder_name, 'csv')
    config.pickle_output_dir = Path('outputs', folder_name, 'pickle')
    config.model_output_dir = Path('outputs', folder_name, 'models')
    config.folder_name = folder_name
    config.csv_output_dir.mkdir(parents=True, exist_ok=True)
    config.pickle_output_dir.mkdir(parents=True, exist_ok=True)


    # if config.training.csv:
    #     config.csv_output_dir = Path('outputs', config.folder_name, 'csv')
    #     config.csv_output_dir.mkdir(parents=True, exist_ok=True)

    # set up data loaders
    eval_loader = get_dataloader('eval', config)

    config.data.eval_datasets_size = len(eval_loader.dataset)
    print(f'Size of validation set: {len(eval_loader.dataset)}, size of batches: {len(eval_loader)}')
    
    # Model Defined
    eval_model=Task(config)
    checkpoint = torch.load(Path(config.model_output_dir,"best_checkpoint.ckpt"))
    eval_model.load_state_dict(checkpoint['state_dict'])

    trainer = Trainer(
            accelerator="gpu",
            devices=1
        )
    
    trainer.predict(model=eval_model, dataloaders=eval_loader)