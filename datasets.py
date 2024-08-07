#!/usr/bin/python3

import numpy as np
from random import randrange, randint, sample
from torch.utils.data import Dataset

class MaterialDataset(Dataset):
  def __init__(self, npz_path, drop_n = -5, divide = 'train', max_mats_num = 6):
    assert divide in {'train', 'val', 'test'}
    data = np.load(npz_path, allow_pickle = True)
    self.samples = data[divide + '_reactions']
    self.drop_n = drop_n
    self.max_mats_num = max_mats_num
  def random_drop_in_list(self, input_data, sample_shape):
    if self.drop_n < 0:
      drop_n = randint(1, max(min(-self.drop_n, len(input_data)), 1))
    else:
      drop_n = self.drop_n
    samples = sample(input_data, max(len(input_data) - drop_n, 0))
    if len(samples) == 0:
      samples.append(np.zeros(shape = sample_shape, dtype = np.float32))
    return samples
  def get_max_temperature(self, temperature_dict_list):
    # goal
    max_T = 1000

    assert len(temperature_dict_list) > 0
    all_Ts = [tmp_T["max_value"] for tmp_T in temperature_dict_list]
    max_T = max(all_Ts)
    return max_T
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
      firing_T = self.get_max_temperature(temperature_dict_list)
    else:
      firing_T = 1000
    return firing_T
  def __len__(self):
    return len(self.samples)
  def __getitem__(self, index):
    r = self.samples[index]
    # NOTE: if multiple materials are generated in a reaction, random pick one
    r_target_index = randrange(len(r['target_comp']))
    r_target = r['target_comp'][r_target_index]
    r_target_featurized = r['target_comp_featurized'][r_target_index]
    # NOTE: multiple reactions can generate same targets, there are mulitple sets of precursors for a same set of targets
    # random pick one precursor in each set of precursors
    r_precursors_index = [randrange(len(comps)) for comps in r['precursors_comp']]
    r_precursors = [r['precursors_comp'][i][j] for i, j in enumerate(r_precursors_index)]
    r_precursors_featurized = [r['precursors_comp_featurized'][i][j] for i, j in enumerate(r_precursors_index)]
    precursors_conditional = self.random_drop_in_list(r_precursors, sample_shape = r_target.shape)
    data = {'reaction': np.stack([r_target] + r_precursors + [np.zeros_like(r_target) for i in range(self.max_mats_num - 1 - len(r_precursors))]),
            'reaction_featurized': np.stack([r_target_featurized] + r_precursors_featurized + [np.zeros_like(r_target_featurized) for i in range(self.max_mats_num - 1 - len(r_precursors_featurized))]),
            'precursors_conditional': np.stack(precursors_conditional + [np.zeros_like(precursors_conditional[0]) for i in range(self.max_mats_num - 1 - len(precursors_conditional))]),
            'temperature': self.get_max_firing_T(r),
            'synthesis_type': r['synthesis_type']}
    return data

if __name__ == "__main__":
  md = MaterialDataset('rsc/data_split.npz')
  for s in md:
    print(s)
    break
