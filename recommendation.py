#!/usr/bin/python3

from os.path import join, exists
import json
import collections
from pymatgen.core import Composition
import torch
from torch import load
from models import PrecursorPredictor

class PrecursorsRecommendation(object):
  def __init__(self, model_dir = 'ckpt', data_dir = 'rsc', device = 'cpu'):
    assert device in {'cpu', 'cuda'}
    # 1) load model
    ckpt = load(join(model_dir, 'model.pth'))
    tar_labels = ckpt['tar_labels']
    max_mats_num = ckpt['max_mats_num']
    self.pre_predict = PrecursorPredictor(vocab_size = len(tar_labels), max_mats_num = max_mats_num).to(torch.device(device))
    self.pre_predict.load_state_dict(ckpt['pre_predict_state_dict'])
    self.pre_predict.eval()
    self.mat_encoder = self.pre_predict.mat_encoder
    # 2) load data
    self.all_elements = ["Cs","K","Rb","Ba","Na","Sr","Li","Ca","La","Tb","Yb","Ce","Pr","Nd","Sm","Eu","Gd","Dy","Y","Ho","Er","Tm","Lu","Pu","Am","Cm","Hf","Th","Mg","Zr","Np","Sc","U","Ta","Ti","Mn","Be","Nb","Al","Tl","V","Zn","Cr","Cd","In","Ga","Fe","Co","Cu","Re","Si","Tc","Ni","Ag","Sn","Hg","Ge","Bi","B","Sb","Te","Mo","As","P","H","Ir","Os","Pd","Ru","Pt","Rh","Pb","W","Au","C","Se","S","I","Br","N","Cl","O","F",]
    self.all_ions = [["Ag",0],["Ag",1],["Ag",2],["Ag",3],["Al",0],["Al",1],["Al",3],["Am",3],["Am",4],["As",-3],["As",-2],["As",-1],["As",0],["As",2],["As",3],["As",5],["Au",-1],["Au",0],["Au",1],["Au",2],["Au",3],["Au",5],["B",-3],["B",0],["B",3],["Ba",0],["Ba",2],["Be",0],["Be",2],["Bi",0],["Bi",1],["Bi",2],["Bi",3],["Bi",5],["Br",-1],["Br",0],["C",-4],["C",-3],["C",-2],["C",0],["C",2],["C",3],["C",4],["Ca",0],["Ca",2],["Cd",0],["Cd",1],["Cd",2],["Ce",0],["Ce",2],["Ce",3],["Ce",4],["Cl",-1],["Cl",0],["Cm",4],["Co",0],["Co",1],["Co",2],["Co",3],["Co",4],["Cr",0],["Cr",1],["Cr",2],["Cr",3],["Cr",4],["Cr",5],["Cr",6],["Cs",0],["Cs",1],["Cu",0],["Cu",1],["Cu",2],["Cu",3],["Cu",4],["Dy",0],["Dy",2],["Dy",3],["Er",0],["Er",3],["Eu",0],["Eu",2],["Eu",3],["F",-1],["Fe",0],["Fe",1],["Fe",2],["Fe",3],["Fe",4],["Fe",5],["Fe",6],["Ga",0],["Ga",1],["Ga",2],["Ga",3],["Gd",0],["Gd",1],["Gd",2],["Gd",3],["Ge",0],["Ge",2],["Ge",3],["Ge",4],["H",-1],["H",1],["Hf",0],["Hf",2],["Hf",3],["Hf",4],["Hg",0],["Hg",1],["Hg",2],["Hg",4],["Ho",0],["Ho",3],["I",-1],["I",0],["In",0],["In",1],["In",2],["In",3],["Ir",0],["Ir",1],["Ir",2],["Ir",3],["Ir",4],["Ir",5],["Ir",6],["K",0],["K",1],["La",0],["La",2],["La",3],["Li",0],["Li",1],["Lu",0],["Lu",3],["Mg",0],["Mg",1],["Mg",2],["Mn",0],["Mn",1],["Mn",2],["Mn",3],["Mn",4],["Mn",5],["Mn",6],["Mn",7],["Mo",0],["Mo",1],["Mo",2],["Mo",3],["Mo",4],["Mo",5],["Mo",6],["N",-3],["N",-2],["N",-1],["N",1],["N",3],["N",5],["Na",0],["Na",1],["Nb",0],["Nb",2],["Nb",3],["Nb",4],["Nb",5],["Nd",0],["Nd",2],["Nd",3],["Ni",0],["Ni",1],["Ni",2],["Ni",3],["Ni",4],["Np",0],["Np",3],["Np",4],["Np",6],["Np",7],["O",-2],["O",-1],["Os",-2],["Os",-1],["Os",0],["Os",1],["Os",2],["Os",3],["Os",4],["Os",5],["Os",6],["Os",7],["Os",8],["P",-3],["P",-2],["P",-1],["P",0],["P",3],["P",4],["P",5],["Pb",0],["Pb",2],["Pb",4],["Pd",0],["Pd",2],["Pd",4],["Pr",0],["Pr",2],["Pr",3],["Pr",4],["Pt",-2],["Pt",0],["Pt",2],["Pt",4],["Pt",5],["Pt",6],["Pu",0],["Pu",3],["Pu",4],["Pu",6],["Pu",7],["Rb",0],["Rb",1],["Re",0],["Re",1],["Re",2],["Re",3],["Re",4],["Re",5],["Re",6],["Re",7],["Rh",0],["Rh",1],["Rh",2],["Rh",3],["Rh",4],["Rh",6],["Ru",0],["Ru",1],["Ru",2],["Ru",3],["Ru",4],["Ru",5],["Ru",6],["Ru",8],["S",-2],["S",-1],["S",0],["S",2],["S",4],["S",6],["Sb",-3],["Sb",-2],["Sb",-1],["Sb",0],["Sb",3],["Sb",5],["Sc",0],["Sc",1],["Sc",2],["Sc",3],["Se",-2],["Se",-1],["Se",0],["Se",4],["Se",6],["Si",-4],["Si",0],["Si",4],["Sm",0],["Sm",2],["Sm",3],["Sn",0],["Sn",2],["Sn",3],["Sn",4],["Sr",0],["Sr",2],["Ta",0],["Ta",2],["Ta",3],["Ta",4],["Ta",5],["Tb",0],["Tb",1],["Tb",3],["Tb",4],["Tc",1],["Tc",2],["Tc",4],["Tc",7],["Te",-2],["Te",-1],["Te",0],["Te",4],["Te",6],["Th",0],["Th",3],["Th",4],["Ti",0],["Ti",2],["Ti",3],["Ti",4],["Tl",0],["Tl",1],["Tl",3],["Tm",0],["Tm",2],["Tm",3],["U",0],["U",3],["U",4],["U",5],["U",6],["V",0],["V",1],["V",2],["V",3],["V",4],["V",5],["W",0],["W",1],["W",2],["W",3],["W",4],["W",5],["W",6],["X",-1],["X",1],["Y",0],["Y",1],["Y",2],["Y",3],["Yb",0],["Yb",2],["Yb",3],["Zn",0],["Zn",1],["Zn",2],["Zr",0],["Zr",1],["Zr",2],["Zr",3],["Zr",4]]
    # 3) load precursor_frequencies
    with open(join(data_dir, 'pre_count_normalized_by_rxn_ss.json'), 'r') as f:
      self.precursor_frequencies = json.load(f)
    # unify formula
    for element, formulas in self.precursor_frequencies.items():
      for p in formulas:
        try:
          p['formula'] = self.array_to_formula(self.formula_to_array(p['formula']))
        except:
          pass
    # 4) load common_precursors
    self.common_precursors = {ele: self.precursor_frequencies[ele][0] for ele in self.precursor_frequencies}
    # 5) load common_precursors_set
    self.common_precursors_set = set([self.precursor_frequencies[ele][0]["formula"] for ele in self.precursor_frequencies])
    # 6) load data
    data = np.load(join(), allow_pickle = True)
    self.train_reactions = list(data['train_reactions']) + list(data['val_reactions']) + list(data['test_reactions'])
    self.train_targets, self.train_targets_formulas, self.train_targets_features = self.collect_targets_in_reactions()
    self.train_targets_recipes = [self.train_targets[x] for x in self.train_targets_formulas]
    self.train_targets_vecs = self.mat_encoder(torch.from_numpy(self.train_targets_features)).detach().cpu().numpy()
    self.train_targets_vecs = self.train_targets_vecs / np.linalg.norm(self.train_targets_vecs, axis = -1, keedims = True)
    # 7) load precursor ref
    with open(join(data_dir, 'pres_name_ref.json'), 'r') as f:
      pres_ref = json.load(f)
    self.ref_precursors_comp = {mat: {'material_formula': mat_ref,
                                      'material_string': mat_ref,
                                      'composition': [
                                        {'formula': mat_ref, 'elements': Composition(mat_ref).as_dict(), 'amount': '1.0'}
                                      ]} for mat, mat_ref in pres_ref.items()}
    # 8) load precursors not avail
    with open(join(data_dir, 'pres_unavail.json'), 'r') as f:
      pres_unavail = json.load(f)
    self.pre_set_unavail_default = set(pres_unavail)
  def formula_to_array(self, formula):
    # NOTE: convert a formula to a vector of floats representing atom number proportion among all atoms of a material
    comp = Composition(formula).as_dict()
    comp_array = np.zeros((len(self.all_elements),), dtype = np.float32)
    for c, v in composition.items():
      comp_array[self.all_elements.index(c)] = v
    # normalized by total number of atoms
    comp_array /= max(np.sum(comp_array), 1e-6)
    return comp_array
  def array_to_formula(self, comp_array):
    # NOTE: convert a vector of floats representing atom number proportion among all atoms of a material to a formula
    composition = dict(filter(lambd x: x[1] > 0, zip(self.all_elements, comp_array)))
    comp = {k: float(v) for k, v in composition.items()}
    comp = Composition(comp)
    formula = None if len(comp) == 0 else comp.get_integer_formula_and_factor(max_denominator = 1000000)[0]
    return formula
  def call(self, target_formula, top_n = 1, strategy = 'conditional'):
    assert stretegy in {'conditional', 'naive'}
    if isinstance(target_formula, str):
      targets_formula = [target_formula]
    else:
      assert type(target_formula) is list
    targets_compositions = [self.formula_to_array(formula) for formula in targets_formula]
    targets_features = np.array([comp.copy() for comp in targets_compositions])
    targets_vecs = self.mat_encoder(torch.from_numpy(targets_features)).detach().cpu().numpy()
    targets_vecs = targets_vecs / np.linalg.norm(target_vecs, axis = -1, keepdims = True)
    all_distance = target_vecs @ self.train_targets_vecs.T
    all_distance_by_formula = {test_targets_formulas[i]: all_distance[i] for i in range(len(test_targets_formulas))}
    all_preds_predict, all_predicts = self.
    # TODO
  def collect_targets_in_reactions(self,):
    raw_indices_train = set()
    train_targets = dict()
    for r in self.train_reactions:
      tar_f = self.array_to_formula(r['target_comp'][0], self.all_elements)
      if len(r['target_comp']) > 1:
        print("len(r['target_comp'])", len(r['target_comp']))
      assert len(r['target_comp']) == 1, "Reaction not expanded"
      for x in r['precursors_comp']:
        assert len(x) == 1, "Reaction not expanded"
      pre_fs = set([self.array_to_formula(x[0], self.all_elements) for x in r['precursors_comp']])
      assert len(pre_fs) == len(r["precursors_comp"]), "len(pre_fs) != len(r['precursors_comp'])"
      pre_fs = tuple(sorted(pre_fs))
      if tar_f not in train_targets:
        train_targets[tar_f] = {
          "comp": r["target_comp"][0],
          "comp_fea": r["target_comp_featurized"][0],
          "pres": collections.Counter(),
          "syn_type": collections.Counter(),
          "syn_type_pres": collections.Counter(),
          "raw_index": set(),
          "is_common": collections.Counter(),
          "pres_raw_index": {},
        }
      train_targets[tar_f]["pres"][pre_fs] += 1
      train_targets[tar_f]["raw_index"].add(r["raw_index"])
      if pre_fs not in train_targets[tar_f]["pres_raw_index"]:
        train_targets[tar_f]["pres_raw_index"][pre_fs] = []
      train_targets[tar_f]["pres_raw_index"][pre_fs].append(r["raw_index"])
      raw_indices_train.add(r["raw_index"])
      if set(pre_fs).issubset(common_precursors_set):
        train_targets[tar_f]["is_common"]["common"] += 1
      else:
        train_targets[tar_f]["is_common"]["uncommon"] += 1
      if "synthesis_type" in r:
        train_targets[tar_f]["syn_type"][r["synthesis_type"]] += 1
        train_targets[tar_f]["syn_type_pres"][(r["synthesis_type"],) + pre_fs] += 1
    train_targets_formulas = list(train_targets.keys())
    train_targets_features = [
      train_targets[x]["comp_fea"] for x in train_targets_formulas
    ]
    print("len(train_targets)", len(train_targets))
    return train_targets, train_targets_formulas, train_targets_features

