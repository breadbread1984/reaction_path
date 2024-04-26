#!/usr/bin/python3

from absl import flags, app
import torch
from torch import nn
from torch.utils.data import DataLoader
from models import MaterialDecoder, PrecursorPredictor
from datasets import MaterialDataset

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = None, help = 'path to dataset')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to ckpt')
  flags.DEFINE_integer('batch_size', default = 32, help = 'batch size')
  flags.DEFINE_integer('workers', default = 4, help = 'workers')

def main(unused_argv):
  pre_predict = PrecursorPredictor()
  mat_encoder = pre_predict.mat_encoder
  mat_decoder = MaterialDecoder()
  trainset = MaterialDataset(FLAGS.dataset, divide = 'train')
  evalset = MaterialDataset(FLAGS.dataset, divide = 'val')
  trainset_loader = DataLoader(trainset, batch_size = FLAGS.batch_size, shuffle = True, num_workers = FLAGS.workers)
  mse = nn.MSELoss()

if __name__ == "__main__":
  add_options()
  app.run(main)

