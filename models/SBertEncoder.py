import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import lightning.pytorch as pl

from transformers import AutoTokenizer, AutoModel
from transformers import BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer,\
    RobertaModel, RobertaTokenizer, DistilBertModel, DistilBertTokenizer,\
    CLIPTokenizer, CLIPTextModel
from sentence_transformers import SentenceTransformer


from models.BERT_Config import MODELS



class SenBERTEncoder(pl.LightningModule):
    def __init__(self, config):
        super(SenBERTEncoder, self).__init__()
        
        # bert_type = config.bert_encoder.type #settings.yaml에서 'sentence-transformers/all-MiniLM-L6-v2' 로 바꿔야함 ㄴㄴ
        dropout = 0.1
        
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.bert_encoder = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens', add_pooling_layer = False) #pooling_layer 제외
        
        self.text_width = 768
        
        '''
        freeze = True
        if freeze:
        '''
        if config.training.freeze:
            for name, param in self.bert_encoder.named_parameters():
                param.requires_grad = False
        else:
            for name, param in self.bert_encoder.named_parameters():
                param.requires_grad = True
                
        
    def forward(self, caption):
        device = torch.device('cuda')
        
        text_input = self.tokenizer(caption, add_special_tokens=True, truncation=True,
                                    padding=True, return_tensors='pt').to(device)
        
        
        text_output = self.bert_encoder(input_ids = text_input.input_ids,
                                        attention_mask = text_input.attention_mask)[0]
        
        cls = text_output[:,0,:]
        
        return cls
                                
        
        

#         self.tokenizer = AutoTokenizer.from_pretrained()
#         self.model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        