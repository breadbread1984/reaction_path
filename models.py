#!/usr/bin/python3

from collections import namedtuple
from transformers.models.bert import BertLayer
import torch
from torch import nn
import torch.nn.functional as F

all_elements = ["Cs","K","Rb","Ba","Na","Sr","Li","Ca","La","Tb","Yb","Ce","Pr","Nd","Sm","Eu","Gd","Dy","Y","Ho","Er","Tm","Lu","Pu","Am","Cm","Hf","Th","Mg","Zr","Np","Sc","U","Ta","Ti","Mn","Be","Nb","Al","Tl","V","Zn","Cr","Cd","In","Ga","Fe","Co","Cu","Re","Si","Tc","Ni","Ag","Sn","Hg","Ge","Bi","B","Sb","Te","Mo","As","P","H","Ir","Os","Pd","Ru","Pt","Rh","Pb","W","Au","C","Se","S","I","Br","N","Cl","O","F",]
all_ions = [["Ag",0],["Ag",1],["Ag",2],["Ag",3],["Al",0],["Al",1],["Al",3],["Am",3],["Am",4],["As",-3],["As",-2],["As",-1],["As",0],["As",2],["As",3],["As",5],["Au",-1],["Au",0],["Au",1],["Au",2],["Au",3],["Au",5],["B",-3],["B",0],["B",3],["Ba",0],["Ba",2],["Be",0],["Be",2],["Bi",0],["Bi",1],["Bi",2],["Bi",3],["Bi",5],["Br",-1],["Br",0],["C",-4],["C",-3],["C",-2],["C",0],["C",2],["C",3],["C",4],["Ca",0],["Ca",2],["Cd",0],["Cd",1],["Cd",2],["Ce",0],["Ce",2],["Ce",3],["Ce",4],["Cl",-1],["Cl",0],["Cm",4],["Co",0],["Co",1],["Co",2],["Co",3],["Co",4],["Cr",0],["Cr",1],["Cr",2],["Cr",3],["Cr",4],["Cr",5],["Cr",6],["Cs",0],["Cs",1],["Cu",0],["Cu",1],["Cu",2],["Cu",3],["Cu",4],["Dy",0],["Dy",2],["Dy",3],["Er",0],["Er",3],["Eu",0],["Eu",2],["Eu",3],["F",-1],["Fe",0],["Fe",1],["Fe",2],["Fe",3],["Fe",4],["Fe",5],["Fe",6],["Ga",0],["Ga",1],["Ga",2],["Ga",3],["Gd",0],["Gd",1],["Gd",2],["Gd",3],["Ge",0],["Ge",2],["Ge",3],["Ge",4],["H",-1],["H",1],["Hf",0],["Hf",2],["Hf",3],["Hf",4],["Hg",0],["Hg",1],["Hg",2],["Hg",4],["Ho",0],["Ho",3],["I",-1],["I",0],["In",0],["In",1],["In",2],["In",3],["Ir",0],["Ir",1],["Ir",2],["Ir",3],["Ir",4],["Ir",5],["Ir",6],["K",0],["K",1],["La",0],["La",2],["La",3],["Li",0],["Li",1],["Lu",0],["Lu",3],["Mg",0],["Mg",1],["Mg",2],["Mn",0],["Mn",1],["Mn",2],["Mn",3],["Mn",4],["Mn",5],["Mn",6],["Mn",7],["Mo",0],["Mo",1],["Mo",2],["Mo",3],["Mo",4],["Mo",5],["Mo",6],["N",-3],["N",-2],["N",-1],["N",1],["N",3],["N",5],["Na",0],["Na",1],["Nb",0],["Nb",2],["Nb",3],["Nb",4],["Nb",5],["Nd",0],["Nd",2],["Nd",3],["Ni",0],["Ni",1],["Ni",2],["Ni",3],["Ni",4],["Np",0],["Np",3],["Np",4],["Np",6],["Np",7],["O",-2],["O",-1],["Os",-2],["Os",-1],["Os",0],["Os",1],["Os",2],["Os",3],["Os",4],["Os",5],["Os",6],["Os",7],["Os",8],["P",-3],["P",-2],["P",-1],["P",0],["P",3],["P",4],["P",5],["Pb",0],["Pb",2],["Pb",4],["Pd",0],["Pd",2],["Pd",4],["Pr",0],["Pr",2],["Pr",3],["Pr",4],["Pt",-2],["Pt",0],["Pt",2],["Pt",4],["Pt",5],["Pt",6],["Pu",0],["Pu",3],["Pu",4],["Pu",6],["Pu",7],["Rb",0],["Rb",1],["Re",0],["Re",1],["Re",2],["Re",3],["Re",4],["Re",5],["Re",6],["Re",7],["Rh",0],["Rh",1],["Rh",2],["Rh",3],["Rh",4],["Rh",6],["Ru",0],["Ru",1],["Ru",2],["Ru",3],["Ru",4],["Ru",5],["Ru",6],["Ru",8],["S",-2],["S",-1],["S",0],["S",2],["S",4],["S",6],["Sb",-3],["Sb",-2],["Sb",-1],["Sb",0],["Sb",3],["Sb",5],["Sc",0],["Sc",1],["Sc",2],["Sc",3],["Se",-2],["Se",-1],["Se",0],["Se",4],["Se",6],["Si",-4],["Si",0],["Si",4],["Sm",0],["Sm",2],["Sm",3],["Sn",0],["Sn",2],["Sn",3],["Sn",4],["Sr",0],["Sr",2],["Ta",0],["Ta",2],["Ta",3],["Ta",4],["Ta",5],["Tb",0],["Tb",1],["Tb",3],["Tb",4],["Tc",1],["Tc",2],["Tc",4],["Tc",7],["Te",-2],["Te",-1],["Te",0],["Te",4],["Te",6],["Th",0],["Th",3],["Th",4],["Ti",0],["Ti",2],["Ti",3],["Ti",4],["Tl",0],["Tl",1],["Tl",3],["Tm",0],["Tm",2],["Tm",3],["U",0],["U",3],["U",4],["U",5],["U",6],["V",0],["V",1],["V",2],["V",3],["V",4],["V",5],["W",0],["W",1],["W",2],["W",3],["W",4],["W",5],["W",6],["X",-1],["X",1],["Y",0],["Y",1],["Y",2],["Y",3],["Yb",0],["Yb",2],["Yb",3],["Zn",0],["Zn",1],["Zn",2],["Zr",0],["Zr",1],["Zr",2],["Zr",3],["Zr",4]]

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

class PrecursorPredictor(nn.Module):
  def __init__(self, vocab_size, max_mats_num = 6, attention_num_heads = 2, hidden_dropout = 0.1, attention_dropout = 0.1, num_reserved_ids = 10, mat_feature_len = 83, ele_dim_features = 32, num_attention_layers = 3, hidden_activation = 'gelu'):
    super(PrecursorPredictor, self).__init__()
    self.mat_encoder = MaterialEncoder(mat_feature_len, ele_dim_features, num_attention_layers, hidden_activation)
    self.precursor_layer = nn.Linear(ele_dim_features, vocab_size - num_reserved_ids)
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

