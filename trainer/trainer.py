import sys
import time
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from torchinfo import summary
#from loguru import logger
#from pprint import PrettyPrinter
#from torch.utils.tensorboard import SummaryWriter
from tools.utils import setup_seed, AverageMeter, a2t, t2a
from tools.loss import BiDirectionalRankingLoss, TripletLoss, NTXent
from tools.info_loss import InFoNCELoss
from tools.make_csvfile import make_csv


from models.ASE_model import ASE

import lightning.pytorch as pl

class Task(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = ASE(config)
        self.return_ranks = config.training.csv

        #Print SubModules of Task
        summary(self.model.audio_enc)
        summary(self.model.audio_linear)
        summary(self.model.text_enc)
        summary(self.model.text_linear)

        #Set-up for CSV file
        if config.training.csv:
            self.csv_output_dir = Path('outputs', config.folder_name, 'csv')
            self.csv_output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_embs= None  
        self.cap_embs= None
        self.audio_names_= None
        self.caption_names= None
        self.top10 = None

        '''
        This is for logger

        logger.add(sys.stdout, format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
                filter=lambda record: record['extra']['indent'] == 1)
        logger.add(config.log_output_dir.joinpath('output.txt'), format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
                filter=lambda record: record['extra']['indent'] == 1)

        self.main_logger = logger.bind(indent=1)

        # setup TensorBoard
        writer = SummaryWriter(log_dir=str(self.log_output_dir) + '/tensorboard')
    

        # print training settings
        printer = PrettyPrinter()
        self.main_logger.info('Training setting:\n'
                        f'{printer.pformat(config)}')
        '''

        # set up model
        # if torch.cuda.is_available():
        #     device, device_name = ('cuda',torch.cuda.get_device_name(torch.cuda.current_device()))
        # else: 
        #     device, device_name = ('cpu', None)
        # print(f'Process on {device}:{device_name}')

        # Set up Loss function
        if config.training.loss == 'triplet':
            self.criterion = TripletLoss(margin=config.training.margin)
        
        elif config.training.loss == 'ntxent':
            self.criterion = NTXent()
        
        elif config.training.loss == 'weight':
            self.criterion = WeightTriplet(margin=config.training.margin)
            
        elif config.training.loss == 'infonce':
            self.criterion = InFoNCELoss()
            
        else: #config.training.loss == 'bidirect': 'contrastive'??
            self.criterion = BiDirectionalRankingLoss(margin=config.training.margin)

        ep = 1

        # resume from a checkpoint
        if config.training.resume:
            checkpoint = torch.load(config.path.resume_model)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            ep = checkpoint['epoch']

    # def on_train_start(self):
    #     self.recall_sum =[]

    # def on_train_epoch_start(self):
    #     self.epoch_loss = AverageMeter()

    def training_step(self, batch, batch_idx):

        audios, captions, audio_ids, _, _ = batch

        audio_embeds, caption_embeds = self.model(audios, captions)

        loss = self.criterion(audio_embeds, caption_embeds, audio_ids)
        self.log('train_loss',loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # set up optimizer
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config.training.lr)
        # set up scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=5,threshold=0.005,threshold_mode='abs',min_lr=0.000001,verbose=True)
        return {"optimizer":optimizer, 
                "lr_scheduler":{"scheduler":scheduler,
                                "monitor": 'validation_loss',
                                "frequency": 1}}

    # def on_validation_start(self):
    #     self.audio_embs, self.cap_embs , self.audio_names_, self.caption_names= None, None, None, None

    def on_validation_epoch_start(self):
        self.audio_embs, self.cap_embs, self.audio_names_, self.caption_names, self.top10 = None, None, None, None, None
        
    def validation_step(self, batch, batch_idx):
        audios, captions, audio_ids, indexs, audio_names = batch
        data_size = self.config.data.val_datasets_size
        audio_embeds, caption_embeds = self.model(audios, captions)

        if self.audio_embs is None:
            self.audio_embs = np.zeros((data_size, audio_embeds.shape[1]))
            self.cap_embs = np.zeros((data_size, caption_embeds.shape[1]))
            if self.return_ranks:
                self.audio_names_ = np.array([None for i in range(data_size)], dtype=object)
                self.caption_names = np.array([None for i in range(data_size)], dtype=object)
        
        loss = self.criterion(audio_embeds, caption_embeds, audio_ids)
        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.audio_embs[indexs] = audio_embeds.cpu().numpy()
        self.cap_embs[indexs] = caption_embeds.cpu().numpy()

        if self.return_ranks:
            self.audio_names_[indexs] = np.array(audio_names)
            self.caption_names[indexs] = np.array(captions)
        return loss
    
    def on_validation_epoch_end(self):
        if self.return_ranks:
            r1, r5, r10, mAP10, medr, meanr, ranks, self.top10 = t2a(self.audio_embs, self.cap_embs, return_ranks=True)
        else:
            r1, r5, r10, mAP10, medr, meanr = t2a(self.audio_embs, self.cap_embs)
        self.logger.experiment.add_scalars('val_metric',{'r1':r1, 'r5':r5, 'r10':r10, 'mAP10':mAP10, 'medr':medr, 'meanr':meanr})

    # def on_test_start(self):
    #     self.on_validation_start()
    
    def on_test_epoch_start(self):
        self.on_validation_epoch_start()
    
    def test_step(self, batch, batch_idx):
        audios, captions, audio_ids, indexs, audio_names = batch
        data_size = self.config.data.test_datasets_size
        audio_embeds, caption_embeds = self.model(audios, captions)

        if self.audio_embs is None:
            self.audio_embs = np.zeros((data_size, audio_embeds.shape[1]))
            self.cap_embs = np.zeros((data_size, caption_embeds.shape[1]))
            if self.return_ranks:
                self.audio_names_ = np.array([None for i in range(data_size)],dtype=object)
                self.caption_names = np.array([None for i in range(data_size)],dtype=object)
        
        loss = self.criterion(audio_embeds, caption_embeds, audio_ids)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.audio_embs[indexs] = audio_embeds.cpu().numpy()
        self.cap_embs[indexs] = caption_embeds.cpu().numpy()

        if self.return_ranks:
            self.audio_names_[indexs] = np.array(audio_names)
            self.caption_names[indexs] = np.array(captions)
        return loss

    def on_test_end(self):
        if self.return_ranks:
            r1, r5, r10, mAP10, medr, meanr, ranks, self.top10 = t2a(self.audio_embs, self.cap_embs, return_ranks=True)
            print(self.top10[:5])
        else:
            r1, r5, r10, mAP10, medr, meanr = t2a(self.audio_embs, self.cap_embs)
        self.logger.experiment.add_scalars('test_metric',{'r1':r1, 'r5':r5, 'r10':r10, 'mAP10':mAP10, 'medr':medr, 'meanr':meanr})


class CSVCallback(pl.Callback):
    def on_test_end(self, trainer, pl_module):

        # Do something with all test epoch ends.
        print("CSV File Ready...")
        make_csv(pl_module.caption_names, pl_module.audio_names_, pl_module.top10, csv_output_dir=pl_module.csv_output_dir)
        print('CSV File was completly made at {}!'.format(pl_module.csv_output_dir))

        # free up the memory
        pl_module.caption_names.clear()
        pl_module.audio_names_.clear()
        pl_module.audio_embs.clear()
        pl_module.cap_embs.clear()
        pl_module.top10.clear()