#!/usr/bin/python3

from absl import flags, app
from models import MaterialDecoder, PrecursorPredictor
import torch
from torch import nn

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = None, help = 'path to dataset')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to ckpt')

def main(unused_argv):
  mat_encoder = MaterialEncoder()
  mat_decoder = MaterialDecoder()
  pre_predict = PrecursorPredictor(mat_encoder = mat_encoder)
  mse = nn.MSELoss()

if __name__ == "__main__":
  add_options()
  app.run(main)

