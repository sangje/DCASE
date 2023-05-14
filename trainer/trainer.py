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
from tools.utils import setup_seed, AverageMeter, a2t, t2a, t2a_retrieval
from tools.loss import BiDirectionalRankingLoss, WeightTriplet, TripletLoss, NTXent, VICReg, InfoNCE, InfoNCE_VICReg
from tools.make_csvfile import make_csv
import pickle

from models.ASE_model import ASE

import lightning.pytorch as pl

class Task(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.model = ASE(config)
        # self.return_ranks = config.training.csv
        self.pickle_output_path=Path(config.pickle_output_dir,'temporal_embeddings.pkl')
        self.csv_pickle_output_path=Path(config.pickle_output_dir,'temporal_csv_embeddings.pkl')
        self.csv_output_path=Path(config.csv_output_dir,'results.csv')
        self.train_step_outputs = []
        self.validate_step_outputs = []
        
        #Print SubModules of Task
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            # do nothing, only run on main process
            None
        else:
            summary(self.model.audio_enc)
            summary(self.model.audio_linear)
            summary(self.model.text_enc)
            summary(self.model.text_linear)

        # #Set-up for CSV file
        # if config.training.csv:
        #     self.csv_output_dir = Path('outputs', config.folder_name, 'csv')
        #     self.csv_output_dir.mkdir(parents=True, exist_ok=True)


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
            self.criterion = InfoNCE()

        elif config.training.loss == 'infonce+vicreg':
            self.criterion = InfoNCE_VICReg(info_weight=1,vic_weight=0.4)
            
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
        self.log('train_step_loss',loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.train_step_outputs.append(loss)
        return loss
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_step_outputs).mean()
        self.log('train_epoch_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.train_step_outputs.clear()

    def configure_optimizers(self):
        # set up optimizer
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config.training.lr)
        # set up scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=5,threshold=0.005,threshold_mode='abs',min_lr=0.000001,verbose=True)
        return {"optimizer":optimizer, 
                "lr_scheduler":{"scheduler":scheduler,
                                "monitor": 'validation_epoch_loss',
                                "frequency": 1}}

    # def on_validation_start(self):
    #     self.audio_embs, self.cap_embs , self.audio_names_, self.caption_names= None, None, None, None
        
    def validation_step(self, batch, batch_idx):
        # Tensor(N,E), list, Tensor(N), array, list
        audios, captions, audio_ids, indexs, audio_names = batch
        data_size = self.config.data.val_datasets_size
        audio_embeds, caption_embeds = self.model(audios, captions)
            # if self.return_ranks:
            #     Task.audio_names_ = np.array([None for i in range(data_size)], dtype=object)
            #     Task.caption_names = np.array([None for i in range(data_size)], dtype=object)
        
        loss = self.criterion(audio_embeds, caption_embeds, audio_ids)
        self.log('validation_step_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.validate_step_outputs.append(loss)
        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validate_step_outputs).mean()
        self.log('validation_epoch_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.validate_step_outputs.clear()

    # def on_test_start(self):
    #     self.on_validation_start()
    '''
    def on_test_epoch_start(self):
        self.on_validation_epoch_start()
    
    def test_step(self, batch, batch_idx):
        audios, captions, audio_ids, indexs, audio_names = batch
        data_size = self.config.data.test_datasets_size
        audio_embeds, caption_embeds = self.model(audios, captions)

        if Task.audio_embs is None:
            Task.audio_embs = np.zeros((data_size, audio_embeds.shape[1]))
            Task.cap_embs = np.zeros((data_size, caption_embeds.shape[1]))
            if self.return_ranks:
                Task.audio_names_ = np.array([None for i in range(data_size)],dtype=object)
                Task.caption_names = np.array([None for i in range(data_size)],dtype=object)
        
        loss = self.criterion(audio_embeds, caption_embeds, audio_ids)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        Task.audio_embs[indexs] = audio_embeds.cpu().numpy()
        Task.cap_embs[indexs] = caption_embeds.cpu().numpy()

        if self.return_ranks:
            Task.audio_names_[indexs] = np.array(audio_names)
            Task.caption_names[indexs] = np.array(captions)
        return loss

    def on_test_end(self):
        if self.return_ranks:
            r1, r5, r10, mAP10, medr, meanr, ranks, Task.top10 = t2a(Task.audio_embs, Task.cap_embs, return_ranks=True)
            print("Top10 Shape:",Task.top10.shape,"Audio Embeddings:",Task.audio_embs.shape)
        else:
            r1, r5, r10, mAP10, medr, meanr = t2a(Task.audio_embs, Task.cap_embs)
        self.logger.experiment.add_scalars('test_metric',{'r1':r1, 'r5':r5, 'r10':r10, 'mAP10':mAP10, 'medr':medr, 'meanr':meanr})
    
    def on_after_backward(self):
        # call on_test_end() only once after accumulating the results of each process
        if self.trainer.local_rank == 0:
            self.on_test_end()
'''
    def on_test_start(self):
        temporal_dict={'audio_embs':None, 'cap_embs':None, 'audio_names_':None, 'caption_names':None}
        with open(self.pickle_output_path, 'wb') as f:  
            pickle.dump(temporal_dict,f, protocol=pickle.HIGHEST_PROTOCOL)
        
    def test_step(self, batch, batch_idx):
        with open(self.pickle_output_path, 'rb') as f:  
            temporal_dict=pickle.load(f)
        # Tensor(N,E), list, Tensor(N), array, list
        audios, captions, audio_ids, indexs, audio_names = batch
        data_size = self.config.data.val_datasets_size
        audio_embeds, caption_embeds = self.model(audios, captions)
        if temporal_dict['audio_embs'] is None:
            temporal_dict['audio_embs'] = np.zeros((data_size, audio_embeds.shape[1]))
            temporal_dict['cap_embs'] = np.zeros((data_size, caption_embeds.shape[1]))
            # if self.return_ranks:
            #     Task.audio_names_ = np.array([None for i in range(data_size)], dtype=object)
            #     Task.caption_names = np.array([None for i in range(data_size)], dtype=object)
        temporal_dict['audio_embs'][indexs] = audio_embeds.cpu().numpy()
        temporal_dict['cap_embs'][indexs] = caption_embeds.cpu().numpy()

        with open(self.pickle_output_path, 'wb') as f:  
            pickle.dump(temporal_dict,f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def on_test_end(self):
        with open(self.pickle_output_path, 'rb') as f:  
            temporal_dict=pickle.load(f)
        r1, r5, r10, mAP10, medr, meanr = t2a(temporal_dict['audio_embs'], temporal_dict['cap_embs'])
        print(f'r1:{r1}, r5:{r5}, r10:{r10}, mAP10:{mAP10}')
        self.logger.experiment.add_scalars('metric',{'r1':r1, 'r5':r5, 'r10':r10, 'mAP10':mAP10, 'medr':medr, 'meanr':meanr},self.current_epoch)



    def on_predict_start(self):
        temporal_dict={'audio_embs':None, 'cap_embs':None, 'audio_names_':None, 'caption_names':None}
        with open(self.csv_pickle_output_path, 'wb') as f:  
            pickle.dump(temporal_dict,f, protocol=pickle.HIGHEST_PROTOCOL)
        
    def predict_step(self, batch, batch_idx):
        with open(self.csv_pickle_output_path, 'rb') as f:  
            temporal_dict=pickle.load(f)
        # Tensor(N,E), list, Tensor(N), array, list
        audios, captions, audio_ids, indexs, audio_names = batch
        data_size = self.config.data.eval_datasets_size
        audio_embeds, caption_embeds = self.model(audios, captions)
        if temporal_dict['audio_embs'] is None:
            temporal_dict['audio_embs'] = np.zeros((data_size, audio_embeds.shape[1]))
            temporal_dict['cap_embs'] = np.zeros((data_size, caption_embeds.shape[1]))
            # if self.return_ranks:
            temporal_dict['audio_names_'] = np.array([None for i in range(data_size)], dtype=object)
            temporal_dict['caption_names'] = np.array([None for i in range(data_size)], dtype=object)
        temporal_dict['audio_embs'][indexs] = audio_embeds.cpu().numpy()
        temporal_dict['cap_embs'][indexs] = caption_embeds.cpu().numpy()
        temporal_dict['audio_names_'][indexs] = np.array(audio_names)
        temporal_dict['caption_names'][indexs] = np.array(captions)

        with open(self.csv_pickle_output_path, 'wb') as f:  
            pickle.dump(temporal_dict,f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def on_predict_end(self):
        with open(self.csv_pickle_output_path, 'rb') as f:  
            temporal_dict=pickle.load(f)
        top10 = t2a_retrieval(temporal_dict['audio_embs'], temporal_dict['cap_embs'],return_ranks=True)
        make_csv(temporal_dict['caption_names'], temporal_dict['audio_names_'], top10, self.csv_output_path)

