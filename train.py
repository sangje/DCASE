import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import torch
from trainer.trainer import Task
from tools.config_loader import get_config
from pathlib import Path
from data_handling.DataLoader import get_dataloader
# from tools.make_csvfile import make_csv

from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger

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
    config.model_output_dir = Path('outputs', folder_name, 'models')
    config.log_output_dir = Path('outputs', folder_name, 'logging')
    config.pickle_output_dir = Path('outputs', folder_name, 'pickle')
    config.csv_output_dir = Path('outputs', folder_name, 'csv')
    config.folder_name = folder_name
    config.log_output_dir.mkdir(parents=True, exist_ok=True)
    config.model_output_dir.mkdir(parents=True, exist_ok=True)
    config.pickle_output_dir.mkdir(parents=True, exist_ok=True)
    

    # if config.training.csv:
    #     config.csv_output_dir = Path('outputs', config.folder_name, 'csv')
    #     config.csv_output_dir.mkdir(parents=True, exist_ok=True)

    # set up data loaders
    train_loader = get_dataloader('train', config)
    val_loader = get_dataloader('val', config)
    # test_loader = get_dataloader('test', config)
    config.data.val_datasets_size = len(val_loader.dataset)
    # config.data.test_datasets_size = len(test_loader.dataset)
    print(f'Size of training set: {len(train_loader.dataset)}, size of batches: {len(train_loader)}')
    print(f'Size of validation set: {len(val_loader.dataset)}, size of batches: {len(val_loader)}')
    # print(f'Size of test set: {len(test_loader.dataset)}, size of batches: {len(test_loader)}')
    
    # Model Defined
    train_model=Task(config)

    # Checkpoint and LR Monitoring
    checkpoint_callback = ModelCheckpoint(monitor='validation_epoch_loss',
        filename="best_checkpoint", dirpath=config.model_output_dir)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    ddp_strategy = DDPStrategy(find_unused_parameters=False)

    trainer = Trainer(
        logger=TensorBoardLogger(save_dir=config.log_output_dir),
        max_epochs=config.training.epochs,
        strategy=ddp_strategy,
        num_sanity_val_steps=-1,
        sync_batchnorm=True,
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=config.log_output_dir,
        reload_dataloaders_every_n_epochs=1,
        accumulate_grad_batches=1,
        log_every_n_steps=1,
        )
    
    trainer.fit(model=train_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    

    torch.distributed.destroy_process_group()
    if trainer.is_global_zero:
        # Batch Size 12 for Testing.
        config.data.batch_size=12
        test_loader = get_dataloader('val', config)
        trainer = Trainer(
            logger=TensorBoardLogger(save_dir=config.log_output_dir),
            accelerator="gpu",
            devices=1
        )
        checkpoint = torch.load(Path(config.model_output_dir,"best_checkpoint.ckpt"))
        test_model = Task(config)
        test_model.load_state_dict(checkpoint['state_dict'])
        trainer.test(model=test_model, dataloaders=test_loader)
