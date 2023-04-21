import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sentence_transformers import util



class InFoNCELoss(nn.Module):
    
    def __init__(self, temperature=0.07):
        super(InFoNCELoss, self).__init__()
        
        self.loss = nn.CrossEntropyLoss()
        self.tau = temperature
        
    
    def forward(self, audio_embeds, text_embeds, labels):
        """
        :param audio_embeds: tensor, (N,E)
        :param text_embeds
        :param labels(item_batch) # audio-text info 
        :return:
        """
        
        loss = torch.tensor(0., device=audio_embeds.device, requires_grad=True)

        n = audio_embeds.size(0) # 배치 사이즈
              
        
        a2t = util.cos_sim(audio_embeds, text_embeds) / self.tau
        t2a = util.cos_sim(text_embeds, audio_embeds) / self.tau
        
        labels = torch.tensor(np.eye(n)).to(a2t.device)
        
        a2t_loss = self.loss(a2t, labels)
        t2a_loss = self.loss(t2a, labels)
        
        loss = loss + (a2t_loss + t2a_loss)/2
        
        return loss
    