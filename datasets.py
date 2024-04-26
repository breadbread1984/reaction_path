#!/usr/bin/python3

import numpy as np
from random import randrange, randint, sample
from torch.utils.data import Dataset

class MaterialDataset(Dataset):
  def __init__(self, npz_path, drop_n = 1, divide = 'train'):
    assert divide in {'train', 'val', 'test'}
    data = np.load(npz_path)
    self.samples = data[divide + 'reactions']
    self.drop_n = drop_n
  def random_drop_in_list(self, input_data, sample_shape):
    if self.drop_n < 0:
      drop_n = randint(1, max(min(-self.drop_n, len(input_data)), 1))
    else:
      drop_n = self.drop_n
    samples = sample(input_data, max(len(input_data) - drop_n, 0))
    if len(samples) == 0:
      samples.append(np.zeros(shape = sample_shape, dtype = np.float32))
    return samples
  def get_max_firing_T(self, reaction):
    temperature_dict_list = []
    for op in reaction["operations"]:
      if (
        op["type"] == "Firing"
        and op["conditions"]
        and op["conditions"]["heating_temperature"]
      ):
        temperature_dict_list.extend(op["conditions"]["heating_temperature"])
    if temperature_dict_list:
      firing_T = get_max_temperature(temperature_dict_list)
    else:
      firing_T = 1000
    return firing_T
  def __len__(self):
    return len(self.samples)
  def __getitem__(self, index):
    r = self.samples[index]
    r_target_index = randrange(len(r['target_comp']))
    r_target = r['target_comp'][r_target_index]
    r_target_featurized = r['target_comp_featurized'][r_target_index]
    r_precursors_index = [randrange(len(comps)) for comps in r['precursors_comp']]
    r_precursors = [r['precursors_comp'][i][j] for i, j in enumerate(r_precursors_index)]
    r_precursors_featurized = [r['precursors_comp_featurized'][i][j] for i, j in enumerate(r_precursors_index)]
    return {'reaction': [r_target] + r_precursors,
            'reaction_featurized': [r_target_featurized] + r_precursors_featurized,
            'precursors_conditional': self.random_drop_in_list(r_precursors, sample_shape = r_target.shape),
            'temperature': self/get_max_firing_T(r),
            'synthesis_type': r['synthesis_type']}

