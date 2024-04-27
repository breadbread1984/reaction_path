#!/usr/bin/python3

from absl import flags, app
from os import mkdir
from os.path import exists, join
import torch
from torch import nn, save, load, no_grad, device
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from models import MaterialDecoder, PrecursorPredictor
from datasets import MaterialDataset
from utils import get_ele_counts

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = None, help = 'path to dataset')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to ckpt')
  flags.DEFINE_integer('batch_size', default = 32, help = 'batch size')
  flags.DEFINE_integer('workers', default = 4, help = 'workers')
  flags.DEFINE_float('lr', default = 1e-3, help = 'learning rate')
  flags.DEFINE_integer('epoch', default = 100, help = 'epochs')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')

def main(unused_argv):
  pre_predict = PrecursorPredictor()
  mat_encoder = pre_predict.mat_encoder
  mat_decoder = MaterialDecoder()
  trainset = MaterialDataset(FLAGS.dataset, divide = 'train')
  evalset = MaterialDataset(FLAGS.dataset, divide = 'val')
  ele_counts = get_ele_counts(trainset)
  ele_mask = ele_counts > 0
  trainset_loader = DataLoader(trainset, batch_size = FLAGS.batch_size, shuffle = True, num_workers = FLAGS.workers)
  evalset_loader = DataLoader(evalset, batch_size = FLAGS.batch_size, shuffle = True, num_workers = FLAGS.workers)
  optimizer = Adam(list(pre_predict.parameters()) + list(mat_decoder.parameters()), lr = FLAGS.lr)
  scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5 , T_mult = 2)
  mse = nn.MSELoss()
  if not exists(FLAGS.ckpt): mkdir(FLAGS.ckpt)
  tb_writer = SummaryWriter(log_dir = join(FLAGS.ckpt, 'summaries'))
  start_epoch = 0
  if exists(join(FLAGS.ckpt, 'model.pth')):
    ckpt = load(join(FLAGS.ckpt, 'model.pth'))
    pre_predict.load_state_dict(ckpt['pre_predict_state_dict'])
    mat_encoder = pre_predict.mat_encoder
    mat_decoder.load_state_dict(ckpt['mat_decoder_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler = ckpt['scheduler']
    start_epoch = ckpt['epoch']
  for epoch in range(start_epoch, FLAGS.epoch):
    pre_predict.train()
    mat_decoder.train()
    for step, sample in enumerate(trainset_loader):
      optimizer.zero_grad()
      materials_featurized = sample['reaction_featurized']
      x_mat = mat_encoder(materials_featurized)
      x_mat = mat_decoder(x_mat)

if __name__ == "__main__":
  add_options()
  app.run(main)

