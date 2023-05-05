import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import lightning.pytorch as pl
from sentence_transformers import SentenceTransformer



class SenBERTEncoder(pl.LightningModule):
    def __init__(self, config):
        super(SenBERTEncoder, self).__init__()
        
        self.bert_encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') # Input:Text Output:Embeddings(768-dimensions)
        
        if config.training.freeze:
            for name, param in self.bert_encoder.named_parameters():
                param.requires_grad = False
        else:
            for name, param in self.bert_encoder.named_parameters():
                param.requires_grad = True
                
        
    def forward(self, captions):        
        
        text_output = self.bert_encoder.encode(captions)
        
        return text_output