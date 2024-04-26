#!/usr/bin/python3

from random import randrange
from torch.utils.data import Dataset

class MaterialDataset(Dataset):
  def __init__(self, npz_path, divide = 'train'):
    assert divide in {'train', 'val', 'test'}
    data = np.load(npz_path)
    self.samples = data[divide + 'reactions']
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
            'precursors_conditional': }
