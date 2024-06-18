#!/usr/bin/python3

from os import mkdir, remove
from os.path import join, exists, expanduser
import json
import collections
from pymatgen.core import Composition
from gdown import download
from zipfile import ZipFile
import numpy as np
import torch
from torch import load
from .models import PrecursorPredictor
from .utils import get_composition_string

class PrecursorsRecommendation(object):
  def __init__(self, device = 'cpu'):
    assert device in {'cpu', 'cuda'}
    if not exists(join(expanduser('~'), '.react_path')): mkdir(join(expanduser('~'), '.react_path'))
    if not exists(join(expanduser('~'), '.react_path', 'rsc')):
      download(id = '1ack7mcyHtUVMe99kRARvdDV8UhweElJ4', output = join(expanduser('~'), '.react_path', 'rsc.zip'))
      with ZipFile(join(expanduser('~'), '.react_path', 'rsc.zip'), 'r') as f:
        f.extractall(join(expanduser('~'), '.react_path'))
      remove(join(expanduser('~'), '.react_path', 'rsc.zip'))
    if not exists(join(expanduser('~'), '.react_path', 'reaction_path_ckpt')):
      download(id = '1dDiCcWNEbsnPyKrZYXsYiOsWiLAsmii3', output = join(expanduser('~'), '.react_path', 'ckpt.zip'))
      with ZipFile(join(expanduser('~'), '.react_path', 'ckpt.zip'), 'r') as f:
        f.extractall(join(expanduser('~'), '.react_path'))
      remove(join(expanduser('~'), '.react_path', 'ckpt.zip'))
    model_dir = join(expanduser('~'), '.react_path', 'reaction_path_ckpt')
    data_dir = join(expanduser('~'), '.react_path', 'rsc')
    # 1) load model
    ckpt = load(join(model_dir, 'model.pth'), map_location = torch.device(device))
    self.tar_labels = ckpt['tar_labels']
    self.max_mats_num = ckpt['max_mats_num']
    self.num_reserved_ids = ckpt['num_reserved_ids']
    self.pre_predict = PrecursorPredictor(vocab_size = len(self.tar_labels), max_mats_num = self.max_mats_num).to(torch.device(device))
    self.pre_predict.load_state_dict(ckpt['pre_predict_state_dict'])
    self.pre_predict.eval()
    self.mat_encoder = self.pre_predict.mat_encoder
    # 2) load data
    self.all_elements = ["Cs","K","Rb","Ba","Na","Sr","Li","Ca","La","Tb","Yb","Ce","Pr","Nd","Sm","Eu","Gd","Dy","Y","Ho","Er","Tm","Lu","Pu","Am","Cm","Hf","Th","Mg","Zr","Np","Sc","U","Ta","Ti","Mn","Be","Nb","Al","Tl","V","Zn","Cr","Cd","In","Ga","Fe","Co","Cu","Re","Si","Tc","Ni","Ag","Sn","Hg","Ge","Bi","B","Sb","Te","Mo","As","P","H","Ir","Os","Pd","Ru","Pt","Rh","Pb","W","Au","C","Se","S","I","Br","N","Cl","O","F",]
    self.all_ions = [["Ag",0],["Ag",1],["Ag",2],["Ag",3],["Al",0],["Al",1],["Al",3],["Am",3],["Am",4],["As",-3],["As",-2],["As",-1],["As",0],["As",2],["As",3],["As",5],["Au",-1],["Au",0],["Au",1],["Au",2],["Au",3],["Au",5],["B",-3],["B",0],["B",3],["Ba",0],["Ba",2],["Be",0],["Be",2],["Bi",0],["Bi",1],["Bi",2],["Bi",3],["Bi",5],["Br",-1],["Br",0],["C",-4],["C",-3],["C",-2],["C",0],["C",2],["C",3],["C",4],["Ca",0],["Ca",2],["Cd",0],["Cd",1],["Cd",2],["Ce",0],["Ce",2],["Ce",3],["Ce",4],["Cl",-1],["Cl",0],["Cm",4],["Co",0],["Co",1],["Co",2],["Co",3],["Co",4],["Cr",0],["Cr",1],["Cr",2],["Cr",3],["Cr",4],["Cr",5],["Cr",6],["Cs",0],["Cs",1],["Cu",0],["Cu",1],["Cu",2],["Cu",3],["Cu",4],["Dy",0],["Dy",2],["Dy",3],["Er",0],["Er",3],["Eu",0],["Eu",2],["Eu",3],["F",-1],["Fe",0],["Fe",1],["Fe",2],["Fe",3],["Fe",4],["Fe",5],["Fe",6],["Ga",0],["Ga",1],["Ga",2],["Ga",3],["Gd",0],["Gd",1],["Gd",2],["Gd",3],["Ge",0],["Ge",2],["Ge",3],["Ge",4],["H",-1],["H",1],["Hf",0],["Hf",2],["Hf",3],["Hf",4],["Hg",0],["Hg",1],["Hg",2],["Hg",4],["Ho",0],["Ho",3],["I",-1],["I",0],["In",0],["In",1],["In",2],["In",3],["Ir",0],["Ir",1],["Ir",2],["Ir",3],["Ir",4],["Ir",5],["Ir",6],["K",0],["K",1],["La",0],["La",2],["La",3],["Li",0],["Li",1],["Lu",0],["Lu",3],["Mg",0],["Mg",1],["Mg",2],["Mn",0],["Mn",1],["Mn",2],["Mn",3],["Mn",4],["Mn",5],["Mn",6],["Mn",7],["Mo",0],["Mo",1],["Mo",2],["Mo",3],["Mo",4],["Mo",5],["Mo",6],["N",-3],["N",-2],["N",-1],["N",1],["N",3],["N",5],["Na",0],["Na",1],["Nb",0],["Nb",2],["Nb",3],["Nb",4],["Nb",5],["Nd",0],["Nd",2],["Nd",3],["Ni",0],["Ni",1],["Ni",2],["Ni",3],["Ni",4],["Np",0],["Np",3],["Np",4],["Np",6],["Np",7],["O",-2],["O",-1],["Os",-2],["Os",-1],["Os",0],["Os",1],["Os",2],["Os",3],["Os",4],["Os",5],["Os",6],["Os",7],["Os",8],["P",-3],["P",-2],["P",-1],["P",0],["P",3],["P",4],["P",5],["Pb",0],["Pb",2],["Pb",4],["Pd",0],["Pd",2],["Pd",4],["Pr",0],["Pr",2],["Pr",3],["Pr",4],["Pt",-2],["Pt",0],["Pt",2],["Pt",4],["Pt",5],["Pt",6],["Pu",0],["Pu",3],["Pu",4],["Pu",6],["Pu",7],["Rb",0],["Rb",1],["Re",0],["Re",1],["Re",2],["Re",3],["Re",4],["Re",5],["Re",6],["Re",7],["Rh",0],["Rh",1],["Rh",2],["Rh",3],["Rh",4],["Rh",6],["Ru",0],["Ru",1],["Ru",2],["Ru",3],["Ru",4],["Ru",5],["Ru",6],["Ru",8],["S",-2],["S",-1],["S",0],["S",2],["S",4],["S",6],["Sb",-3],["Sb",-2],["Sb",-1],["Sb",0],["Sb",3],["Sb",5],["Sc",0],["Sc",1],["Sc",2],["Sc",3],["Se",-2],["Se",-1],["Se",0],["Se",4],["Se",6],["Si",-4],["Si",0],["Si",4],["Sm",0],["Sm",2],["Sm",3],["Sn",0],["Sn",2],["Sn",3],["Sn",4],["Sr",0],["Sr",2],["Ta",0],["Ta",2],["Ta",3],["Ta",4],["Ta",5],["Tb",0],["Tb",1],["Tb",3],["Tb",4],["Tc",1],["Tc",2],["Tc",4],["Tc",7],["Te",-2],["Te",-1],["Te",0],["Te",4],["Te",6],["Th",0],["Th",3],["Th",4],["Ti",0],["Ti",2],["Ti",3],["Ti",4],["Tl",0],["Tl",1],["Tl",3],["Tm",0],["Tm",2],["Tm",3],["U",0],["U",3],["U",4],["U",5],["U",6],["V",0],["V",1],["V",2],["V",3],["V",4],["V",5],["W",0],["W",1],["W",2],["W",3],["W",4],["W",5],["W",6],["X",-1],["X",1],["Y",0],["Y",1],["Y",2],["Y",3],["Yb",0],["Yb",2],["Yb",3],["Zn",0],["Zn",1],["Zn",2],["Zr",0],["Zr",1],["Zr",2],["Zr",3],["Zr",4]]
    # 3) load precursor_frequencies
    # load frequencies of materials of every element
    with open(join(data_dir, 'pre_count_normalized_by_rxn_ss.json'), 'r') as f:
      self.precursor_frequencies = json.load(f)
    for element, formulas in self.precursor_frequencies.items():
      for p in formulas:
        try:
          # convert formula to a canonical format
          p['formula'] = self.array_to_formula(self.formula_to_array(p['formula']))
        except:
          pass
    # 4) load common_precursors
    # load first formula frequency of each element
    self.common_precursors = {ele: formulas[0] for ele, formulas in self.precursor_frequencies.items()}
    # 5) load common_precursors_set
    # load first formula string of each element
    self.common_precursors_set = set([formulas[0]["formula"] for ele, formulas in self.precursor_frequencies.items()])
    # 6) load data
    self.train_targets, self.train_targets_formulas, self.train_targets_features = self.collect_targets_in_reactions(data_dir)
    self.train_targets_recipes = [self.train_targets[x] for x in self.train_targets_formulas]
    self.train_targets_vecs = self.mat_encoder(torch.from_numpy(np.stack(self.train_targets_features)))[0].detach().cpu().numpy()
    self.train_targets_vecs = self.train_targets_vecs / np.linalg.norm(self.train_targets_vecs, axis = -1, keepdims = True)
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
    for c, v in comp.items():
      comp_array[self.all_elements.index(c)] = v
    # normalized by total number of atoms
    comp_array /= max(np.sum(comp_array), 1e-6)
    return comp_array
  def array_to_formula(self, comp_array):
    # NOTE: convert a vector of floats representing atom number proportion among all atoms of a material to a formula
    composition = dict(filter(lambda x: x[1] > 0, zip(self.all_elements, comp_array)))
    comp = {k: float(v) for k, v in composition.items()}
    comp = Composition(comp)
    formula = None if len(comp) == 0 else comp.get_integer_formula_and_factor(max_denominator = 1000000)[0]
    return formula
  def call(self, target_formula, top_n = 1, strategy = 'conditional', precursors_not_available = "default"):
    assert strategy in {'conditional', 'naive'}
    if isinstance(target_formula, str):
      target_formula = [target_formula]
    else:
      assert type(target_formula) is list
    if precursors_not_available is None:
      precursors_not_available = set()
    elif precursors_not_available == "default":
      precursors_not_available = self.pre_set_unavail_default
    elif isinstance(precursors_not_available, str):
      raise NotImplementedError
    # 1) convert formula of targets to feature vectors with material encoder
    targets_compositions = [self.formula_to_array(formula) for formula in target_formula]
    targets_features = np.array([comp.copy() for comp in targets_compositions])
    targets_vecs = self.mat_encoder(torch.from_numpy(targets_features))[0].detach().cpu().numpy()
    # 2) calculate distances between feature vectors of target material and precursor feature vectors
    targets_vecs = targets_vecs / np.linalg.norm(targets_vecs, axis = -1, keepdims = True)
    all_distance = targets_vecs @ self.train_targets_vecs.T # all_distance.shape = (target_num, precursor_num)
    all_distance_by_formula = {target_formula[i]: all_distance[i] for i in range(len(target_formula))}
    # 3) predict all precursors with the precursors with closest distances
    all_preds_predict, all_predicts = self.recommend_precursors_by_similarity(
      test_targets_formulas = target_formula,
      test_targets_compositions = targets_compositions,
      test_targets_features = targets_features,
      all_distance = all_distance,
      top_n = top_n,
      validate_first_attempt = True,
      strategy = strategy,
      precursors_not_available = precursors_not_available
    )
    return all_predicts 
  def reformat_precursors(self, pres_candidates, ref_precursors_comp):
    reformated_pres_candidates = []
    for i in range(len(pres_candidates)):
      pres = pres_candidates[i]
      pres_info = {}
      for p in pres:
        if p in ref_precursors_comp:
          p_comp = ref_precursors_comp[p]
        else:
          p_comp = self.formula_to_array(p)
          ref_precursors_comp[p] = p_comp
        # TODO: can i also make self.all_elements as ndarray?
        p_eles = set(np.array(self.all_elements)[p_comp > 0])
        pres_info[p] = {
          "formula": p,
          "composition": p_comp,
          "elements": p_eles,
        }
      reformated_pres_candidates.append(pres_info)
    return reformated_pres_candidates, ref_precursors_comp
  def common_precursors_recommendation(self, eles_target, common_precursors, common_eles, validate_first_attempt = True, validate_reaction = True, target_formula = None, ref_materials_comp = None):
    common_pres = []
    for ele in eles_target:
      if ele in common_precursors:
        common_pres.append(common_precursors[ele])
    pres_eles = set(sum([x["elements"] for x in common_pres], []))
    pres_formulas = tuple(sorted(set([x["formula"] for x in common_pres])))
    if validate_first_attempt:
      if not (
        pres_eles.issubset(eles_target | common_eles)
        and eles_target.issubset(pres_eles | {"O","H",})
      ):
        pres_formulas = None
    return pres_formulas
  def recommend_precursors_by_similarity(self, test_targets_formulas, all_distance, test_targets_compositions = None, test_targets_features = None, top_n = 1, validate_first_attempt = False, path_log = "dist_reaction.txt", common_eles = ("C", "H", "O", "N"), strategy = "conditional", precursors_not_available = None,):
        all_pres_predict = []
        all_rxns_predict = []
        ref_precursors_comp = {}
        ref_materials_comp = {}
        common_eles = set(common_eles)
        if precursors_not_available is None:
            precursors_not_available = set()

        all_predicts = []
        print("len(test_targets_formulas)", len(test_targets_formulas))
        print("top_n", top_n)
        for x_index, x in enumerate(test_targets_formulas):
            if x_index % 10 == 0:
                print(
                    "x_index: {} out of {}".format(x_index, len(test_targets_formulas))
                )
            most_similar_y_index = np.argsort(all_distance[x_index, :])[::-1]

            all_predicts.append(
                {
                    "target_formula": x,
                    "precursors_predicts": [],
                }
            )

            eles_x = set([str(ele) for ele in Composition(x).elements])
            pres_candidates = []
            zero_composition = np.zeros(
                shape=(len(self.all_elements),),
                dtype=np.float32,
            )
            for y_index in most_similar_y_index[: 300 * top_n]:
                # 30*top_n is sufficient to get recommendations
                # no need to check all ref targets
                # all recipes for each target used by freq here
                pres_candidates.extend(
                    [
                        item[0]
                        for item in self.train_targets_recipes[y_index]["pres"].most_common()
                    ]
                )
            # reformat pres_candidates (formula -> eles)
            pres_candidates, ref_precursors_comp = self.reformat_precursors(
                pres_candidates=pres_candidates,
                ref_precursors_comp=ref_precursors_comp,
            )

            pres_multi_predicts = []

            # add common precursors
            if top_n > 1:
                pres_predict = self.common_precursors_recommendation(
                    eles_target=eles_x,
                    common_precursors=self.common_precursors,
                    common_eles=common_eles,
                    validate_first_attempt=validate_first_attempt,
                    # validate_reaction=validate_reaction,
                    target_formula=x,
                    ref_materials_comp=ref_materials_comp,
                )
                if pres_predict is not None:
                    pres_multi_predicts.append(pres_predict)
                    all_predicts[-1]["precursors_predicts"].append(pres_predict)

            pres_conditional_tried = set()
            for i in range(len(pres_candidates)):
                pres_predict = set()
                eles_covered = {
                    "O",
                    "H",
                }
                precursors_conditional = []

                pres = pres_candidates[i]
                for p in pres.values():
                    if p["formula"] in pres_predict:
                        continue
                    if p["elements"].issubset(eles_x | common_eles):
                        pres_predict.add(p["formula"])
                        eles_covered |= p["elements"]
                        precursors_conditional.append(p["composition"])

                # print('pres_predict 1', pres_predict)
                if not eles_x.issubset(eles_covered):
                    if tuple(sorted(pres_predict)) in pres_conditional_tried:
                        continue
                    pres_conditional_tried.add(tuple(sorted(pres_predict)))
                    for _ in range(len(pres_predict), self.max_mats_num - 1):
                        precursors_conditional.append(zero_composition)
                    target_compositions = np.expand_dims(test_targets_compositions[x_index], axis = 0)
                    target_features = np.expand_dims(test_targets_features[x_index], axis = 0)
                    pre_cond = get_composition_string(precursors_conditional)
                    pre_cond_label = np.array([(self.tar_labels.index(pre.item()) - self.num_reserved_ids if pre.item() in self.tar_labels else -1) for pre in pre_cond])
                    pre_cond_label = np.expand_dims(pre_cond_label, axis = 0) # (1, max_mat_nums - 1)
                    y_pred = self.pre_predict(torch.from_numpy(target_features), precursors_conditional_indices = torch.from_numpy(pre_cond_label)) # y_pred.shape = (batch, mat_count - num_reserved_ids)
                    y_pred = y_pred.detach().cpu().numpy()
                    pre_lists_pred = list()
                    for a_y in y_pred:
                        pre_list_pred = np.array(self.tar_labels[self.num_reserved_ids:])[a_y > 0.5]
                        pre_score_pred = a_y[a_y > 0.5]
                    pre_lists_pred.append([{'composition': np.array(comp.decode('utf-8').split(' ')).astype(np.float32), 'score': score} for (comp, score) in zip(pre_list_pred, pre_score_pred)])
                    pre_lists_pred[-1] = sorted(pre_lists_pred[-1], key = lambda x: x['score'], reverse = True)
                    pre_str_lists_pred = list()
                    for i, tar_comp in enumerate(target_compositions):
                        pre_str_list = [(self.array_to_formula(comp['composition']), comp['score']) for comp in pre_lists_pred[i]]
                        pre_str_lists_pred.append(pre_str_list)
                    # NOTE: here
                    for ele in eles_x - eles_covered:
                        if eles_x.issubset(eles_covered):
                            # done
                            break
                        for (p_comp_prob, p_f_prob) in zip(pre_lists_pred[0], pre_str_lists_pred[0]):
                            p_comp = p_comp_prob["composition"]
                            assert p_comp.shape[0] == len(self.all_elements)
                            p_f = p_f_prob[0]
                            p_eles = set(np.array(self.all_elements)[p_comp > 0])
                            if p_f in pres_predict:
                                continue
                            if p_f in precursors_not_available:
                                continue
                            if ele in p_eles and p_eles.issubset(
                                eles_x | common_eles
                            ):
                                pres_predict.add(p_f)
                                eles_covered |= p_eles
                                break

                if not eles_x.issubset(eles_covered):
                    continue

                pres_predict = tuple(sorted(pres_predict))
                # precursors recommended or not
                if pres_predict in pres_multi_predicts:
                    # is_recommended = True
                    continue
                pres_multi_predicts.append(pres_predict)
                all_predicts[-1]["precursors_predicts"].append(pres_predict)

                if len(pres_multi_predicts) >= top_n:
                    break

            all_pres_predict.append(pres_multi_predicts)
        return all_pres_predict, all_predicts
  def collect_targets_in_reactions(self, data_dir):
    data = np.load(join(data_dir, 'data_split.npz'), allow_pickle = True)
    train_reactions = list(data['train_reactions']) + list(data['val_reactions']) + list(data['test_reactions'])
    raw_indices_train = set()
    train_targets = dict()
    for r in train_reactions:
      # save formula of target to string tar_f
      tar_f = self.array_to_formula(r['target_comp'][0])
      if len(r['target_comp']) > 1:
        print("len(r['target_comp'])", len(r['target_comp']))
      assert len(r['target_comp']) == 1, "Reaction not expanded"
      # save formulas of precursors to tuple pre_fs
      for x in r['precursors_comp']:
        assert len(x) == 1, "Reaction not expanded"
      pre_fs = set([self.array_to_formula(x[0]) for x in r['precursors_comp']])
      assert len(pre_fs) == len(r["precursors_comp"]), "len(pre_fs) != len(r['precursors_comp'])"
      pre_fs = tuple(sorted(pre_fs))
      # insert target and related info into train_targets
      if tar_f not in train_targets:
        train_targets[tar_f] = {
          "comp": r["target_comp"][0],
          "comp_fea": r["target_comp_featurized"][0],
          "pres": collections.Counter(),
          "syn_type": collections.Counter(),
          "syn_type_pres": collections.Counter(),
          "raw_index": set(),
          "is_common": collections.Counter(),
          "pres_raw_index": dict(),
        }
      train_targets[tar_f]["pres"][pre_fs] += 1
      train_targets[tar_f]["raw_index"].add(r["raw_index"])
      if pre_fs not in train_targets[tar_f]["pres_raw_index"]:
        train_targets[tar_f]["pres_raw_index"][pre_fs] = []
      train_targets[tar_f]["pres_raw_index"][pre_fs].append(r["raw_index"])
      raw_indices_train.add(r["raw_index"])
      if set(pre_fs).issubset(self.common_precursors_set):
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

