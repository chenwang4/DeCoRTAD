#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add the past window in the input
"""
import math
import torch
import torch.nn as nn
from torch import  nn
from torch import Tensor
from tools.embeddings import (TimestepEmbedding, Timesteps)
from position_encoding_layer import PositionalEncoding
from cross_attention import (SkipTransformerEncoder,
                             TransformerDecoder,
                             TransformerDecoderLayer,
                             TransformerEncoder,
                             TransformerEncoderLayer,
                             TransformerOutputLayer)
from position_encoding import build_position_encoding
from temos_utils import lengths_to_mask
from typing import List, Optional
import torch.nn.functional as F
class MldDenoiser(nn.Module):

    def __init__(self,
                 nfeats: int = 4,
                 condition: str = "None",
                 latent_dim: list = [1, 256],
                 #output_dim = 512,
                 ff_size: int = 1024,
                 num_layers: int = 7,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 flip_sin_to_cos: bool = True,
                 return_intermediate_dec: bool = False,
                 position_embedding: str = "learned",
                 arch: str = "trans_enc",
                 freq_shift: int = 0,
                 guidance_scale: float = 7.5,
                 guidance_uncondp: float = 0.1,
                 clean_encoded_dim: int = 32,
                 nclasses: int = 10,
                 **kwargs) -> None:

        super().__init__()

        self.latent_dim = latent_dim[-1]
        self.clean_encoded_dim = clean_encoded_dim
        self.condition = condition
        self.abl_plus = False
        self.ablation_skip_connection = True
        #self.diffusion_only = ablation.VAE_TYPE == "no"
        self.arch = arch
        self.pe_type = 'mld' #positional encoding

        self.state_embd = nn.Linear(nfeats, self.latent_dim)       
        self.state_proj = nn.Linear(self.latent_dim, nfeats)

        # emb proj
        
        self.time_proj = Timesteps(clean_encoded_dim, flip_sin_to_cos,
                                       freq_shift)
        self.time_embedding = TimestepEmbedding(clean_encoded_dim,
                                                    self.latent_dim)
        
        if clean_encoded_dim != self.latent_dim:
            self.emb_proj = nn.Sequential(nn.ReLU(), nn.Linear(clean_encoded_dim, self.latent_dim))
        # elif self.condition in ["text", "text_uncond"]:
        #     # text condition
        #     # project time from clean_encoded_dim to latent_dim
        #     self.time_proj = Timesteps(clean_encoded_dim, flip_sin_to_cos,
        #                                freq_shift)
        #     self.time_embedding = TimestepEmbedding(clean_encoded_dim,
        #                                             self.latent_dim)
        #     # project time+text to latent_dim
        #     if clean_encoded_dim != self.latent_dim:
        #         # todo 10.24 debug why relu
        #         self.emb_proj = nn.Sequential(
        #             nn.ReLU(), nn.Linear(clean_encoded_dim, self.latent_dim))
        # elif self.condition in ['action']:
        #     self.time_proj = Timesteps(self.latent_dim, flip_sin_to_cos,
        #                                freq_shift)
        #     self.time_embedding = TimestepEmbedding(self.latent_dim,
        #                                             self.latent_dim)
        #     self.emb_proj = EmbedAction(nclasses,
        #                                 self.latent_dim,
        #                                 guidance_scale=guidance_scale,
        #                                 guidance_uncodp=guidance_uncondp)
        # else:
        #     raise TypeError(f"condition type {self.condition} not supported")

        if self.pe_type == "actor":
            self.query_pos = PositionalEncoding(self.latent_dim, dropout)
            self.mem_pos = PositionalEncoding(self.latent_dim, dropout)
        elif self.pe_type == "mld":
            self.query_pos = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
            self.mem_pos = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
        else:
            raise ValueError("Not Support PE type")
        
        if self.arch == "trans_enc":
            if self.ablation_skip_connection:
                # use DETR transformer
                encoder_layer = TransformerEncoderLayer(
                    self.latent_dim,
                    num_heads,
                    ff_size,
                    dropout,
                    activation,
                    normalize_before,
                )
                encoder_norm = nn.LayerNorm(self.latent_dim)
                self.encoder = SkipTransformerEncoder(encoder_layer,
                                                      num_layers, encoder_norm)
            else:
                # use torch transformer
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation)
                self.encoder = nn.TransformerEncoder(encoder_layer,
                                                     num_layers=num_layers)
        elif self.arch == "trans_dec":
            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = TransformerDecoder(
                decoder_layer,
                num_layers,
                decoder_norm,
                return_intermediate=return_intermediate_dec,
            )
        else:
            raise ValueError(f"Not supported architechure{self.arch}!")

    def forward(self,
                sample,
                timestep,
                encoder_hidden_states=None,
                lengths=None,
                **kwargs):
        # 0.  dimension matching
        # sample [latent_dim[0], batch_size, latent_dim] <= [batch_size, latent_dim[0], latent_dim[1]]
        sample = sample.permute(1, 0, 2)

        # 0. check lengths for no vae (diffusion only)
        # if lengths not in [None, []]:
        #     mask = lengths_to_mask(lengths, sample.device)

        # 1. time_embedding
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timestep.expand(sample.shape[1]).clone()
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=sample.dtype)
        # [1, bs, latent_dim] <= [bs, latent_dim]
        time_emb = self.time_embedding(time_emb).unsqueeze(0)
        #print('time_emb',time_emb.shape)
        
        # 2. condition + time embedding
        latent_clean = encoder_hidden_states.permute(1,0,2)
        if self.clean_encoded_dim != self.latent_dim:
            emb_clean = self.emb_proj(latent_clean)
        else:
            emb_clean = latent_clean
        #print('encoder_hidden_states',encoder_hidden_states.shape)
        # unconditional 
        if self.condition in ["None"]:
            emb_latent = time_emb
        elif self.condition in ["CleanSample"]:
            emb_latent = torch.cat((time_emb, emb_clean), 0)
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # 4. transformer
        if self.arch == "trans_enc":
            sample = self.state_embd(sample)
            xseq = torch.cat((sample, emb_latent), axis=0)
            #positional encoding
            xseq = self.query_pos(xseq)
            tokens = self.encoder(xseq)
            sample = tokens[emb_latent.shape[0]:]
            sample = self.state_proj(sample)
            #sample[~mask.T] = 0

            
            #sample = tokens[:sample.shape[0]]

        elif self.arch == "trans_dec":
            

            # tgt    - [1 or 5 or 10, bs, latent_dim]
            # memory - [token_num, bs, latent_dim]
            sample = self.query_pos(sample)
            emb_latent = self.mem_pos(emb_latent)
            sample = self.decoder(tgt=sample, memory=emb_latent).squeeze(0)

            
        else:
            raise TypeError("{self.arch} is not supoorted")

        # 5. [batch_size, latent_dim[0], latent_dim[1]] <= [latent_dim[0], batch_size, latent_dim[1]]
        sample = sample.permute(1, 0, 2)

        return sample

   

class Latent_Encoder(nn.Module):
    def __init__(self, 
                  input_size,
                  latent_dim,
                  win_len,
                  representation_dim=64):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.win_len = win_len
        self.representation_dim = representation_dim
        self.dropout = 0.1
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.pos = PositionalEncoding(self.latent_dim, self.dropout, batch_first=True)
        self.emb_proj = nn.Sequential(
            nn.ReLU(), nn.Linear(input_size, latent_dim))
        self.out = nn.Linear(latent_dim*win_len, representation_dim)
    def forward(self, seq):
        # transform input to latent dim
        z_seq = self.emb_proj(seq)
        z_seq = self.pos(z_seq)
        output = self.encoder(z_seq)
        output = self.out(torch.flatten(output, start_dim=1))
        return output

class Latent_Encoder_MLP(nn.Module):
    def __init__(self, 
                  input_size,
                  latent_dim):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.linear = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, latent_dim)
        self.relu = nn.ReLU()
        
    def forward(self, sample):
        output = self.relu(self.linear(sample))
        output = self.relu(self.linear2(output))
        return output   
class Diffusion(nn.Module):
    def __init__(self, feat_size, current_win_len, past_win_len, t_range, latent_dim=64):
        super().__init__()
        #self.time_embedding = TimeEmbedding(320)
        
        #self.final = UNET_OutputLayer(320, 4)
        self.beta_small = 1e-4
        self.beta_large = 0.02
        self.t_range = t_range
        self.latent_dim = latent_dim
        self.feat_size = feat_size
        self.in_size = feat_size * current_win_len
        
        self.autoencoder_past = Latent_Encoder_MLP(input_size=feat_size*past_win_len, latent_dim=self.latent_dim)
        self.autoencoder = Latent_Encoder(input_size=feat_size, latent_dim=64, win_len=current_win_len+self.latent_dim//feat_size, representation_dim=self.latent_dim)
        self.unet = MldDenoiser(nfeats=feat_size, latent_dim=[1, 256], condition='CleanSample', clean_encoded_dim=self.latent_dim)
      # self.ContrastiveLoss = ContrastiveLoss(batch_size=batch_size, device='mps')
        # self.linear = nn.Linear(self.latent_dim, 16)
        # self.relu = nn.ReLU()
        #self.unet = MldDenoiser(text_encoded_dim=text_encoded_dim, output_dim=self.in_size)
    def forward(self, sample, time, context=None):
        # latent: (Batch_Size, 1, 256)
        #latent = self.autoencoder(sample)
        if self.unet.condition in ["None"]:
            output = self.unet(sample, time)
        else:
            output = self.unet(sample, time, context)
        
        return output
    
    def get_loss(self, batch, setting='HighNoise', context=None):
        batch_cur, batch_last = batch
        # get a random time step for each input in the batch
        device = batch_cur.device
        if setting == 'HighNoise':
            ts = torch.randint(self.t_range//2, self.t_range, [batch_cur.shape[0]]).to(device) #high noise
        elif setting == 'LowNoise':
            ts = torch.randint(5, self.t_range//3, [batch_cur.shape[0]]).to(device) #low noise
        elif setting == 'AllLevel':
            ts = torch.randint(1, self.t_range, [batch_cur.shape[0]]).to(device) 
        else:
            return 'seeting error'
        #ts = torch.randint(1, self.t_range//6, [batch.shape[0]]).to(device) #high noise
        # generate noise
        noised_sample = []
        epsilons = torch.randn(batch_cur.shape).to(device)
        for i in range(len(ts)):
            a_hat = torch.tensor(self.alpha_bar(ts[i])).to(device)
            #batch = batch.to(self.device)
            
            noised_sample.append(
                (torch.sqrt(a_hat) * batch_cur[i]) + (torch.sqrt(1 - a_hat) * epsilons[i])
                )
            #print(noise_latent)
        noised_sample = torch.stack(noised_sample, dim=0).to(device)
        
       
        latent_past_sample = self.autoencoder_past(batch_last.reshape(batch_last.shape[0],-1)).unsqueeze(1)
        
        batch_cond = torch.cat((batch_cur, latent_past_sample.reshape(batch_cur.shape[0], -1, self.feat_size)), axis=1)
        
        latent_clean_sample = self.autoencoder(batch_cond).unsqueeze(1)
        
        context = torch.cat((latent_clean_sample, latent_past_sample), 1)
        
        #latent_clean_last_sample = self.autoencoder_past(batch_last)
        #batch_cond = torch.cat((batch_cur, latent_clean_last_sample), 2)
        #latent_clean_sample = self.autoencoder(batch_cond)
        # add latent representation z of the clean data as additional input to the diffuser
        #noise_latent = torch.cat((noised_sample, latent_clean_sample, latent_clean_last_sample), 2)
        # run the noisy obs through the model, to get predicted noise
        e_hat = self.forward(noised_sample, ts, context).to(device)
        #print(e_hat.shape)
        # compute the loss - MSE between the predicted noise and the actual noise
        loss = nn.functional.mse_loss(e_hat.reshape(-1, self.in_size), epsilons.reshape(-1, self.in_size))
        return loss
    
    # def get_nceLoss(self, batch, sigma):
    #     device = batch.device
    #     latent_clean_sample = self.autoencoder(batch)
    #     noised_batch = batch + torch.randn(batch.shape).to(device) * sigma
    #     latent_noise_sample = self.autoencoder(noised_batch)
    #     cl_latent_clean = self.relu(self.linear(latent_clean_sample))
    #     cl_latent_noise = self.relu(self.linear(latent_noise_sample))
    #     return self.ContrastiveLoss(cl_latent_clean, cl_latent_noise)
    def beta(self, t):
        # Just a simple linear interpolation between beta_small and beta_large based on t
        return self.beta_small + (t / self.t_range) * (self.beta_large - self.beta_small)

    def alpha(self, t):
        return 1 - self.beta(t)

    def alpha_bar(self, t):
        # Product of alphas from 0 to t
        return math.prod([self.alpha(j) for j in range(t)])
    
        

def lengths_to_mask(lengths: List[int],
                    device: torch.device,
                    max_len: int = None) -> Tensor:
    lengths = torch.tensor(lengths, get=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask



# class ContrastiveLoss(nn.Module):
#     def __init__(self, batch_size, temperature=0.2, device='cpu'):
#         super().__init__()
#         self.batch_size = batch_size
#         self.temperature = torch.tensor(temperature).to(device)
        
#     def forward(self, emb_n, emb_a):
#         z_n = F.normalize(emb_n, dim=1)
#         z_a = F.normalize(emb_a, dim=1)
#         similarity_matrix_n = F.cosine_similarity(z_n.unsqueeze(1), z_n.unsqueeze(0), dim=2)
#         positives = similarity_matrix_n[0]
#         nominator = torch.exp(positives / self.temperature)
        
#         similarity_matrix_a = torch.exp(F.cosine_similarity(z_n.unsqueeze(1), z_a.unsqueeze(0), dim=2) / self.temperature)
        
#         loss_partial = -torch.log(nominator / torch.sum(similarity_matrix_a, dim=1))
#         loss = torch.sum(loss_partial) / self.batch_size
#         return loss
    

