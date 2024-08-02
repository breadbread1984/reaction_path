#!/usr/bin/python3

from reaction_path import PrecursorsRecommendation

def test():
  recommend = PrecursorsRecommendation()
  target_formula = [
    "SrZnSO"
  ]
  all_predicts = recommend.call(target_formula = target_formula, top_n = 10)
  assert all_predicts[0]['target_formula'] == 'SrZnSO' and len(all_predicts[0]['precursors_predicts']) != 0

