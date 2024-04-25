#!/usr/bin/python3

from collections import namedtuple
from transformers.models.bert import BertLayer
import torch
from torch import nn
import torch.nn.functional as F

class MaterialEncoder(nn.Module):
  def __init__(self, mat_feature_len = 83, ele_dim_features = 32, num_attention_layers = 3, hidden_activation = 'gelu'):
    super(MaterialEncoder, self).__init__()
    self.shift = nn.parameter.Parameter(-0.5, requires_grad = True)
    self.layers = nn.ModuleList([nn.Linear(mat_feature_len if i == 0 else ele_dim_features, ele_dim_features) for i in range(num_attention_layers)])
  def forward(self, inputs):
    # inputs.shape = (batch, hidden_size)
    mask = torch.any(torch.not_equal(inputs, 0), dim = -1)
    processed_inputs = inputs[mask] # processed_inputs.shape = (reduced batch, hidden size)
    x = torch.where(torch.equal(processed_inputs, 0), self.shift, processed_inputs) # x.shape = (reduced batch, hidden size)
    for layer in self.layers:
      x = layer(x) # x.shape = (reduced batch, ele_dim_features)
      if hidden_activation == 'gelu':
        x = F.gelu(x)
      else:
        raise Exception('unknown activation!')
    x = x * torch.rsqrt(torch.sum(x ** 2, dim = -1, keepdim = True)) # x.shape = (reduced batch, ele_dim_features)
    index = torch.range(inputs.shape[0])[mask]
    x = torch.zeros((inputs.shape[0], x.shape[-1])).scatter_(dim = 0, index, x)

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

if __name__ == "__main__":
  layer = TransformerLayer(1024)

