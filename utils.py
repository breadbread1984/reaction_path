#!/usr/bin/python3

import collections
import numpy as np

def get_composition_string(materials):
  # convert an array of weights representing material
  # to an array of strings representing material
  comp_str = np.char.mod('%.6f', materials)
  comp_str = np.array([bytes(' '.join(e), encoding = 'utf-8') for e in comp_str])
  return comp_str

def convert_list_to_dico(all_labels, count_weights, num_reserved_ids = 10, least_count = 5):
  out_labels = list()
  out_counts = list()
  if count_weights is None:
    count_weights = [1.0] * len(all_labels)
  else:
    assert len(count_weights) == len(all_labels)
  # count times of appearance of each material
  # only leave ones of times of appearnce over 5
  big_dico = collections.Counter()
  for i in range(len(all_labels)):
    w = count_weights[i]
    s = all_labels[i]
    if not isinstance(s, bytes):
      s = bytes(s, encoding = 'utf-8')
    big_dico[s] += w
  big_dico = big_dico.most_common()
  big_dico = list(filter(lambda x: x[1] >= least_count, big_dico))
  print('len(big_dico)', len(big_dico))
  # insert extra tokens besides material to big_dico
  if num_reserved_ids >= 2:
    placeholder = ["<PLACEHOLDER>_{}".format(i) for i in range(num_reserved_ids)]
    placeholder[0] = "<MASK>"
    placeholder[1] = "<UNK>"
    for p in reversed(placeholder):
      big_dico.insert(0,(bytes(p, encoding="utf-8"),0,))

  out_labels = [x[0] for x in big_dico]
  out_counts = [x[1] for x in big_dico]
  # return materials and their times of appearance
  return out_labels, out_counts

def get_mat_dico(npz_path, mode = "all", num_reserved_ids = 10, least_count = 5):
  reactions = np.load(npz_path, allow_pickle = True)['train_reactions']
  mat_labels = list()
  mat_compositions = list()
  mat_counts = list()
  all_mats = list() # all molecules appeared in the dataset
  all_count_weights = list()
  # put all precursors and all targets into list all_mats
  # insert 1 for each material into list all_count_weights
  for r in reactions:
    if mode in {'all', 'target'}:
      targets = r['target_comp']
      all_mats.extend(targets)
      all_count_weights.extend([r.get('count_weight', 1.0)] * len(targets))
    if mode in {'all', 'precursor'}:
      precursors = sum(r['precursors_comp'], [])
      all_mats.extend(precursors)
      all_count_weights.extend([r.get('count_weight', 1.0)] * len(precursors))
  # convert all materials into strings as keys to materials
  all_mats_str = get_composition_string(np.array(all_mats))
  mat_str_comp = {s: comp for (s, comp) in zip(all_mats_str, all_mats)}
  comp_shape = reactions[0]["target_comp"][0].shape # 83
  # get all materials (distinct molecules) and their counts in dataset
  mat_labels, mat_counts = convert_list_to_dico(
    all_labels = all_mats_str,
    count_weights = all_count_weights,
    num_reserved_ids = num_reserved_ids,
    least_count = least_count)
  # create atom count array for each material
  mat_compositions = [mat_str_comp.get(l, np.zeros(shape = comp_shape, dtype = np.float32)) for l in mat_labels]
  return mat_labels, mat_compositions, mat_counts

def get_ele_counts(npz_path, num_reserved_ids = 10):
  # tar_labels: distinct materials
  # tar_counts: time of appearance of distinct materials in dataset
  # tar_compositions: atom counts for each material
  tar_labels, tar_compositions, tar_counts = get_mat_dico(npz_path, mode = 'precursor', num_reserved_ids = num_reserved_ids, least_count = 0)
  assert len(tar_compositions) > 0
  ele_counts = np.zeros_like(tar_compositions[0]) # ele_counts.shape = (83,)
  for i in range(len(tar_compositions)):
    comp = tar_compositions[i]
    weight = tar_counts[i]
    ele_counts += (comp > 0) * weight
  # ele_counts: atom time of appearance in whole dataset
  return ele_counts, tar_labels

def generate_labels(labels, class_num):
  # labels.shape = (batch, max_mat_nums - 1)
  multi_hot = list()
  for label in labels:
    mask = label >= 0
    cls_set = label[mask]
    label = np.zeros((class_num,))
    label[cls_set] = np.ones((cls_set.shape[0],))
    multi_hot.append(label)
  multi_hot = np.stack(multi_hot, axis = 0)
  return multi_hot
