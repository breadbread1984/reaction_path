#!/usr/bin/python3

from collections import namedtuple
from transformers.models.bert import BertLayer
import torch
from torch import nn
import torch.nn.functional as F

all_elements = ["Cs","K","Rb","Ba","Na","Sr","Li","Ca","La","Tb","Yb","Ce","Pr","Nd","Sm","Eu","Gd","Dy","Y","Ho","Er","Tm","Lu","Pu","Am","Cm","Hf","Th","Mg","Zr","Np","Sc","U","Ta","Ti","Mn","Be","Nb","Al","Tl","V","Zn","Cr","Cd","In","Ga","Fe","Co","Cu","Re","Si","Tc","Ni","Ag","Sn","Hg","Ge","Bi","B","Sb","Te","Mo","As","P","H","Ir","Os","Pd","Ru","Pt","Rh","Pb","W","Au","C","Se","S","I","Br","N","Cl","O","F",]

class MaterialEncoder(nn.Module):
  def __init__(self, mat_feature_len = 83, ele_dim_features = 32, num_attention_layers = 3, hidden_activation = 'gelu'):
    super(MaterialEncoder, self).__init__()
    self.ele_dim_features = ele_dim_features
    self.hidden_activation = hidden_activation
    self.shift = nn.Parameter(torch.tensor(-0.5), requires_grad = True)
    self.layers = nn.ModuleList([nn.Linear(mat_feature_len if i == 0 else ele_dim_features, ele_dim_features) for i in range(num_attention_layers)])
  def forward(self, inputs):
    # inputs.shape = (batch, mat_feature_len)
    mask = torch.any(torch.not_equal(inputs, 0), dim = -1)
    processed_inputs = inputs[mask] # processed_inputs.shape = (reduced batch, mat_feature_len)
    x = torch.where(processed_inputs == 0, self.shift, processed_inputs) # x.shape = (reduced batch, mat_feature_len)
    for layer in self.layers:
      x = layer(x) # x.shape = (reduced batch, ele_dim_features)
      if self.hidden_activation is None:
        pass
      elif self.hidden_activation == 'gelu':
        x = F.gelu(x)
      else:
        raise Exception('unknown activation!')
    # map back samples to its position in the original batch
    x = x * torch.rsqrt(torch.sum(x ** 2, dim = -1, keepdim = True)) # x.shape = (reduced batch, ele_dim_features)
    index = torch.unsqueeze(torch.arange(start = 0, end = inputs.shape[0])[mask], dim = -1) # index.shape = (reduced batch, 1)
    index = torch.tile(index, (1, self.ele_dim_features)) # index.shape = (reduced batch, ele_dim_features)
    x = torch.zeros((inputs.shape[0], self.ele_dim_features)).scatter_(dim = 0, index = index, src = x) # x.shape = (batch, ele_dim_features)
    return x

class MaterialDecoder(nn.Module):
  def __init__(self, mat_feature_len = 83, ele_dim_features = 32, final_activation = None):
    super(MaterialDecoder, self).__init__()
    self.final_activation = final_activation
    self.element_layer = nn.Linear(ele_dim_features, mat_feature_len)
  def forward(self, inputs):
    # inputs.shape = (batch, ele_dim_features)
    mask = torch.any(torch.not_equal(inputs, 0), dim = -1)
    processed_inputs = inputs[mask] # processed_inputs.shape = (reduced batch, ele_dim_features)
    x = self.element_layer(processed_inputs) # x.shape = (reduced batch, num_eles)
    if self.final_activation is None:
      pass
    elif self.final_activation == 'gelu':
      x = F.gelu(x)
    else:
      raise Exception('unknown activation!')
    # map back samples to is position in the original batch
    

def TransformerLayer(max_mats_num,
                     hidden_size = 768,
                     num_attention_heads = 12,
                     intermediate_size = 3072,
                     intermediate_activation = 'gelu',
                     hidden_dropout_prob = 0.0,
                     attention_probs_dropout_prob = 0.0):
  config = {'chunk_size_feed_forward': 0,
            'is_decoder': False,
            'add_cross_attention': False,
            'num_attention_heads': num_attention_heads,
            'hidden_size': hidden_size,
            'intermediate_size': intermediate_size,
            'attention_probs_dropout_prob': attention_probs_dropout_prob,
            'position_embedding_type': 'absolute',
            'max_position_embeddings': max_mats_num,
            'layer_norm_eps': 1e-12,
            'hidden_dropout_prob': hidden_dropout_prob,
            'hidden_act': intermediate_activation}
  Config = namedtuple('config', config.keys())
  config = Config(**config)
  return BertLayer(config)

if __name__ =="__main__":
  me = MaterialEncoder()
  inputs = torch.Tensor([[0 for i in range(83)],[1,] + [0 for i in range(82)]]).to(torch.float32)
  outputs = me(inputs)
  print(outputs)
  #layer = TransformerLayer(1024)

