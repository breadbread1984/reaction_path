#!/usr/bin/python3

from collections import namedtuple
from transformers.models.bert import BertLayer, BertConfig
import torch
from torch import nn
import torch.nn.functional as F

# encoder encoding compound's element weight vector (83) to feature vector (32)
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
      elif self.hidden_activation == 'relu':
        x = F.relu(x)
      else:
        raise Exception('unknown activation!')
    # simplified RMS Normalization
    x = x * torch.rsqrt(torch.sum(x ** 2, dim = -1, keepdim = True)) # x.shape = (reduced batch, ele_dim_features)
    # map back samples to its position in the original batch
    index = torch.unsqueeze(torch.arange(start = 0, end = inputs.shape[0]).to(mask.device)[mask], dim = -1) # index.shape = (reduced batch, 1)
    index = torch.tile(index, (1, self.ele_dim_features)) # index.shape = (reduced batch, ele_dim_features)
    x = torch.zeros((inputs.shape[0], self.ele_dim_features)).to(x.device).scatter_(dim = 0, index = index, src = x) # x.shape = (batch, ele_dim_features)
    return x, mask

# decoder decoding feature vector (32) to compound's element weight vector (83)
class MaterialDecoder(nn.Module):
  def __init__(self, mat_feature_len = 83, ele_dim_features = 32, final_activation = None, norm_in_element_projection = False):
    super(MaterialDecoder, self).__init__()
    self.mat_feature_len = mat_feature_len
    self.final_activation = final_activation
    self.norm_in_element_projection = norm_in_element_projection
    self.element_layer = nn.Linear(ele_dim_features, mat_feature_len)
  def forward(self, inputs):
    # inputs.shape = (batch, ele_dim_features)
    mask = torch.any(torch.not_equal(inputs, 0), dim = -1)
    processed_inputs = inputs[mask] # processed_inputs.shape = (reduced batch, ele_dim_features)
    x = self.element_layer(processed_inputs) # x.shape = (reduced batch, mat_feature_len)
    if self.final_activation is None:
      pass
    elif self.final_activation == 'gelu':
      x = F.gelu(x)
    elif self.final_activation == 'relu':
      x = F.relu(x)
    else:
      raise Exception('unknown activation!')
    # map back samples to is position in the original batch
    index = torch.unsqueeze(torch.arange(start = 0, end = inputs.shape[0]).to(mask.device)[mask], dim = -1) # index.shape = (reduced batch, 1)
    index = torch.tile(index, (1, self.mat_feature_len)) # index.shape = (reduced batch, mat_feature_len)
    x = torch.zeros((inputs.shape[0], self.mat_feature_len)).to(x.device).scatter_(dim = 0, index = index, src = x) # x.shape = (batch, mat_feature_len)
    # simplified RMS normalization
    if self.norm_in_element_projection:
      x = x * torch.rsqrt(torch.sum(self.element_layer.weight ** 2, dim = -1)) # x.shape = (batch, mat_feature_len)
    # prediction head
    x = torch.sigmoid(x) # x.shape = (batch, mat_feature_len)
    return x, mask

# create bert layer with transformers
def TransformerLayer(max_mats_num, 
                     hidden_size = 768,
                     num_attention_heads = 12,
                     intermediate_size = 3072, 
                     intermediate_activation = 'gelu',
                     hidden_dropout_prob = 0.0,
                     attention_probs_dropout_prob = 0.0):
  config = BertConfig(
    hidden_size = hidden_size,
    num_attention_heads = num_attention_heads,
    intermediate_size = intermediate_size,
    hidden_dropout_prob = hidden_dropout_prob,
    attention_probs_dropout_prob = attention_probs_dropout_prob,
    max_position_embeddings = max_mats_num,
  )
  return BertLayer(config)

# compound's element weight vector -> feature vector
# material index vector (like token sequence) representing given precursors -> a sequence of feature vector
# concat compound's feature vector and precursors' vector sequence -> a sequence of feature vector
# feed feature vector to transformer encoder to get a sequence of embedding vector, only use the first embedding vector for classification
# predict a multi hot vector with embedding vector. the set locations represent corresponding materials as precursors
class PrecursorPredictor(nn.Module):
  def __init__(self, vocab_size, max_mats_num = 6, attention_num_heads = 2, hidden_dropout = 0.1, attention_dropout = 0.1, num_reserved_ids = 10, mat_feature_len = 83, ele_dim_features = 32, num_attention_layers = 3, hidden_activation = 'gelu'):
    super(PrecursorPredictor, self).__init__()
    self.mat_encoder = MaterialEncoder(mat_feature_len, ele_dim_features, num_attention_layers, hidden_activation)
    self.precursor_layer = nn.Linear(ele_dim_features, vocab_size - num_reserved_ids) # embedding and predict head share weights
    self.incomplete_reaction_atten_layer = TransformerLayer(max_mats_num = max_mats_num, hidden_size = ele_dim_features, num_attention_heads = attention_num_heads, intermediate_size = ele_dim_features, intermediate_activation = hidden_activation, hidden_dropout_prob = hidden_dropout, attention_probs_dropout_prob = attention_dropout)
  def forward(self, targets, precursors_conditional_indices = None):
    # targets.shape = (batch, mat_feature_len), dtype = float32
    # precursors_conditional_indices.shape = (batch, max_mats_num - 1), dtype = int64, min = 0, max = vocab_size - num_reserved_ids - 1
    targets_emb, _ = self.mat_encoder(targets) # targets_emb.shape = (batch, ele_dim_features)
    if precursors_conditional_indices is not None:
      mask = precursors_conditional_indices >= 0 # mask.shape = (batch, max_mats_num - 1)
      precursors_conditional_indices = torch.where(mask, precursors_conditional_indices, 0) # precursors_conditional_indices.shape = (batch, max_mats_num - 1)
      precursors_conditional_emb = self.precursor_layer.weight[precursors_conditional_indices] # precursors_conditional_emb.shape = (batch, max_mats_num - 1, ele_dim_features)
      precursors_conditional_emb = torch.where(torch.tile(torch.unsqueeze(mask, dim = -1), (1, 1, precursors_conditional_emb.shape[-1])), precursors_conditional_emb, 0.) # precursors_conditional_emb.shape = (batch, max_mats_num - 1, ele_dim_features)
      incomplete_reaction_emb = torch.cat([torch.unsqueeze(targets_emb, dim = 1), precursors_conditional_emb], dim = 1) # incomplete_reaction_emb.shape = (batch, max_mats_num, ele_dim_features)
      incomplete_reaction_mask = torch.cat([torch.ones((targets_emb.shape[0], 1), dtype = torch.int32).to(mask.device), mask.to(torch.int32)], dim = 1).to(torch.float32) # incomplete_reaction_mask.shape = (batch, max_mats_num)
      incomplete_reaction_mask = torch.reshape(incomplete_reaction_mask, (incomplete_reaction_mask.shape[0],1,1,incomplete_reaction_mask.shape[-1])) # incomplete_reaction_mask.shape = (batch, 1, 1, max_mats_num)
      reactions_emb, = self.incomplete_reaction_atten_layer(hidden_states = incomplete_reaction_emb, attention_mask = incomplete_reaction_mask) # reactions_emb.shape = (batch, max_mats_num, ele_dim_features)
      # only use the first token for classification
      reactions_emb = reactions_emb[:,0,:] # reactions_emb.shape = (batch, ele_dim_features)
    else:
      reactions_emb = targets_emb # reactions_emb.shape = (batch, ele_dim_features)
    y_pred = self.precursor_layer(reactions_emb) # y_pred.shape = (batch, vocab_size - num_reserved_ids)
    y_pred = torch.clip(y_pred, min = -10., max = 10.)
    y_pred = torch.sigmoid(y_pred)
    return y_pred

if __name__ =="__main__":
  me = MaterialEncoder()
  inputs = torch.Tensor([[0 for i in range(83)],[1,] + [0 for i in range(82)]]).to(torch.float32)
  outputs, _ = me(inputs)
  print(outputs, outputs.shape)
  md = MaterialDecoder()
  outputs, _ = md(outputs)
  print(outputs, outputs.shape)
  predictor = PrecursorPredictor(50)
  inputs = torch.randn(4, 83)
  outputs = predictor(inputs)
  print(outputs, outputs.shape)
  sampler = torch.distributions.uniform.Uniform(low = 0, high = 50 - 10 - 1)
  precursors_conditional_indices = sampler.sample(sample_shape = (4,5)).to(torch.int64)
  outputs = predictor(inputs, precursors_conditional_indices)
  print(outputs, outputs.shape)
  #layer = TransformerLayer(1024)

