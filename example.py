#!/usr/bin/python3

from reaction_path import PrecursorsRecommendation

def main():
  recommend = PrecursorsRecommendation()
  target_formula = [
    "SrZnSO",
    "Na3TiV(PO4)3",
    "GdLu(MoO4)3",
    "BaYSi2O5N",
    "Cu3Yb(SeO3)2O2Cl",
    #"Li6PS5Cl",
    #"Li1.39S1.59P0.4Br0.109I0.101",
    #"Li5.4S4.45PCl1.7"
  ]
  all_predicts = recommend.call(target_formula = target_formula, top_n = 10)
  from pprint import pprint
  for target, precursors in zip(target_formula, all_predicts):
    print(precursors)

if __name__ == "__main__":
  main()

