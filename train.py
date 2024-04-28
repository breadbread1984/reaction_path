#!/usr/bin/python3

from absl import flags, app
from os import mkdir
from os.path import exists, join
import numpy as np
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
  flags.DEFINE_integer('batch_size', default = 8, help = 'batch size')
  flags.DEFINE_integer('workers', default = 4, help = 'workers')
  flags.DEFINE_float('lr', default = 5e-4, help = 'learning rate')
  flags.DEFINE_integer('epoch', default = 100, help = 'epochs')
  flags.DEFINE_integer('max_mats_num', default = 6, help = 'max number of materials in a sample')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')

def main(unused_argv):
  trainset = MaterialDataset(FLAGS.dataset, divide = 'train', max_mats_num = FLAGS.max_mats_num)
  evalset = MaterialDataset(FLAGS.dataset, divide = 'val', max_mats_num = FLAGS.max_mats_num)
  ele_counts, tar_labels = get_ele_counts(FLAGS.dataset)
  ele_mask = torch.from_numpy(ele_counts > 0).to(torch.float32).to(device(FLAGS.device)) # ele_mask.shape = (83,)
  vocab_size = len(tar_labels) # 10 reseved tokens + number of materials
  pre_predict = PrecursorPredictor(vocab_size = vocab_size, max_mats_num = max_mats_num)
  mat_encoder = pre_predict.mat_encoder
  mat_decoder = MaterialDecoder()
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
      mat = torch.from_numpy(np.concatenate(sample['reaction_featurized'], axis = 0)).to(device(FLAGS.device)) # mat.shape = (material num, 83)
      optimizer.zero_grad()
      # 1) mat encoder + decoder
      embed, mask = mat_encoder(mat) # embed.shape = (material num, 32) mask.shape = (material num)
      rebuild, _ = mat_decoder(embed) # rebuild.shape = (material num, 83)
      mat_decoder_loss = torch.sum(((mat - rebuild) * ele_mask) ** 2, dim = -1) # loss.shape = (material num,)
      mat_decoder_loss = mat_decoder_loss * mask.to(torch.float32) # mat_decoder_loss.shape = (material num)
      # 2) precursor prediction
      targets = torch.unsqueeze(mat[0,:], dim = 0) # targets.shape = (1, 83)

if __name__ == "__main__":
  add_options()
  app.run(main)

