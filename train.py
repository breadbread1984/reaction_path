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
from utils import get_ele_counts, get_composition_string, generate_labels

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = None, help = 'path to dataset')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to ckpt')
  flags.DEFINE_integer('batch_size', default = 8, help = 'batch size')
  flags.DEFINE_integer('workers', default = 4, help = 'workers')
  flags.DEFINE_float('lr', default = 5e-4, help = 'learning rate')
  flags.DEFINE_integer('epoch', default = 100, help = 'epochs')
  flags.DEFINE_integer('max_mats_num', default = 6, help = 'max number of materials in a sample')
  flags.DEFINE_integer('num_reserved_ids', default = 10, help = 'reserved token number')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')
  flags.DEFINE_integer('save_freq', default = 1000, help = 'checkpoint save frequency')

def main(unused_argv):
  trainset = MaterialDataset(FLAGS.dataset, divide = 'train', max_mats_num = FLAGS.max_mats_num)
  evalset = MaterialDataset(FLAGS.dataset, divide = 'val', max_mats_num = FLAGS.max_mats_num)
  ele_counts, tar_labels = get_ele_counts(FLAGS.dataset, num_reserved_ids = FLAGS.num_reserved_ids)
  ele_mask = torch.from_numpy(ele_counts > 0).to(torch.float32).to(device(FLAGS.device)) # ele_mask.shape = (83,)
  vocab_size = len(tar_labels) # 10 reseved tokens + number of materials
  pre_predict = PrecursorPredictor(vocab_size = vocab_size, max_mats_num = FLAGS.max_mats_num).to(device(FLAGS.device))
  mat_encoder = pre_predict.mat_encoder
  mat_decoder = MaterialDecoder().to(device(FLAGS.device))
  trainset_loader = DataLoader(trainset, batch_size = FLAGS.batch_size, shuffle = True, num_workers = FLAGS.workers)
  evalset_loader = DataLoader(evalset, batch_size = FLAGS.batch_size, shuffle = True, num_workers = FLAGS.workers)
  optimizer = Adam(list(pre_predict.parameters()) + list(mat_decoder.parameters()), lr = FLAGS.lr)
  scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5 , T_mult = 2)
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
      reactions = sample['reaction'].to(device(FLAGS.device)) # reaction.shape = (batch, material num, 83)
      reaction_featurized = sample['reaction_featurized'].to(device(FLAGS.device)) # reaction_featurized.shape = (batch, material num, 83)
      precursors_conditional = sample['precursors_conditional'].to(device(FLAGS.device)) # precursors_conditional.shape = (batch, max_mat_nums - 1, 83)
      optimizer.zero_grad()
      # 1) mat encoder + decoder
      mat = torch.flatten(reaction_featurized, start_dim = 0, end_dim = 1) # mat.shape = (batch * material num, 83)
      embed, mask = mat_encoder(mat) # embed.shape = (material num, 32) mask.shape = (material num)

      rebuild, _ = mat_decoder(embed) # rebuild.shape = (material num, 83)
      
      mat_decoder_loss = torch.sum(((mat - rebuild) * ele_mask) ** 2, dim = -1) # loss.shape = (material num,)
      mat_decoder_loss = mat_decoder_loss * mask.to(torch.float32) # mat_decoder_loss.shape = (material num)
      mat_decoder_loss = torch.mean(mat_decoder_loss)
      # 2) precursor prediction
      targets = reaction_featurized[:,0,:].to(device(FLAGS.device)) # targets.shape = (batch, 83)
      pre_cond = precursors_conditional.detach().cpu().numpy() # pre_cond.shape = (batch, max_mat_nums - 1, 83)
      shape = pre_cond.shape
      pre_cond = np.reshape(pre_cond, (shape[0] * shape[1], shape[2])) # pre_cond.shape = (batch * (max_mat_nums - 1), 83)
      pre_cond = get_composition_string(pre_cond) # pre_cond_labels.shape = (batch * (max_mat_nums - 1),)
      pre_cond_indices = np.array([(tar_labels.index(pre.item()) - FLAGS.num_reserved_ids if pre.item() in tar_labels else -1) for pre in pre_cond])
      pre_cond_indices = np.reshape(pre_cond_indices, (shape[0], shape[1])) # pre_cond_indices.shape = (batch, max_mat_nums - 1)
      pre_cond_indices = torch.from_numpy(pre_cond_indices).to(device(FLAGS.device)).to(torch.int32) # pre_cond_indices.shape = (batch, max_mat_nums - 1)

      y_pred = pre_predict(targets, precursors_conditional_indices = pre_cond_indices) # y_pred.shape = (batch, ele_count - num_reserved_ids)

      precursors = reactions[:,1:,:].detach().cpu().numpy() # precursors.shape = (batch, max_mats_num - 1, 83)
      shape = precursors.shape
      precursors = np.reshape(precursors, (shape[0] * shape[1], shape[2])) # precursors.shape = (batch * (max_mat_num - 1), 83)
      precursors = get_composition_string(precursors) # precursors.shape = (batch * (max_mat_nums - 1))
      y_real = np.array([(tar_labels.index(pre.item()) - FLAGS.num_reserved_ids if pre.item() in tar_labels else -1) for pre in precursors])
      y_real = np.reshape(y_real, (shape[0], shape[1])) # y_real.shape = (batch, max_mat_nums - 1)
      y_real = torch.from_numpy(generate_labels(y_real, len(tar_labels) - FLAGS.num_reserved_ids)).to(torch.device(FLAGS.device)) # y_real.shape = (batch, ele count - num_reserved_ids)
      pre_loss = torch.sum(y_real * (-torch.log(y_pred)) + (1 - y_real) * (-torch.log(1 - y_pred)), dim = -1)
      pre_loss = torch.mean(pre_loss, dim = 0)

      # 3) total loss
      loss = mat_decoder_loss * 0.1 + pre_loss * 1.0
      loss.backward()
      optimizer.step()
      global_steps = epoch * len(trainset_loader) + step
      if global_steps % 100 == 0:
        print('Step #%d Epoch #%d: loss %f rebuild loss %f precursors pred loss %f' % (global_steps, epoch, loss, mat_decoder_loss, pre_loss))
        tb_writer.add_scalar('rebuild loss', mat_decoder_loss, global_steps)
        tb_writer.add_scalar('precursors pred loss', pre_loss, global_steps)
        tb_writer.add_scalar('loss', loss, global_steps)
      if global_steps % FLAGS.save_freq == 0:
        ckpt = {
          'pre_predict_state_dict': pre_predict.state_dict(),
          'mat_decoder_state_dict': mat_decoder.state_dict(),
          'optimizer': optimizer.state_dict(),
          'scheduler': scheduler,
          'epoch': epoch}
        save(ckpt, join(FLAGS.ckpt, 'model.pth'))
    scheduler.step()
    pre_predict.eval()
    mat_decoder.eval()
    for step, sample in enumerate(evalset_loader):
      reactions = sample['reaction'].to(device(FLAGS.device))
      reaction_featurized = sample['reaction_featurized'].to(device(FLAGS.device))
      precursors_conditional = sample['precursors_conditional'].to(device(FLAGS.device))
      # 1) mat encoder + decoder
      mat = torch.flatten(reaction_featurized, start_dim = 0, end_dim = 1)
      embed, mask = mat_encoder(mat)
      rebuild, _ = mat_decoder(embed)
      mat_decoder_loss = torch.sum(((mat - rebuild) * ele_mask) ** 2, dim = -1)
      mat_decoder_loss = mat_decoder_loss * mask.to(torch.float32)
      mat_decoder_loss = torch.mean(mat_decoder_loss)
      # 2) precursor prediction
      targets = reaction_featurized[:,0,:].to(device(FLAGS.device))
      pre_cond = precursors_conditional.detach().cpu().numpy()
      shape = pre_cond.shape
      pre_cond = np.reshape(pre_cond, (shape[0] * shape[1], shape[2]))
      pre_cond = get_composition_string(pre_cond)
      pre_cond_indices = np.array([(tar_labels.index(pre.item()) - FLAGS.num_reserved_ids if pre.item() in tar_labels else -1) for pre in pre_cond])
      pre_cond_indices = np.reshape(pre_cond_indices, (shape[0], shape[1]))
      pre_cond_indices = torch.from_numpy(pre_cond_indices).to(device(FLAGS.device)).to(torch.int32)
      y_pred = pre_predict(targets, precursors_conditional_indices = pre_cond_indices)
      precursors = reactions[:,1:,:].detach().cpu().numpy()
      shape = precursors.shape
      precursors = np.reshape(precursors, (shape[0] * shape[1], shape[2]))
      precursors = get_composition_string(precursors)
      y_real = np.array([(tar_labels.index(pre.item()) - FLAGS.num_reserved_ids if pre.item() in tar_labels else -1) for pre in precursors])
      y_real = np.reshape(y_real, (shape[0], shape[1]))
      y_real = torch.from_numpy(generate_labels(y_real, len(tar_labels) - FLAGS.num_reserved_ids)).to(torch.device(FLAGS.device))
      pre_loss = torch.sum(y_real * (-torch.log(y_pred)) + (1 - y_real) * (-torch.log(1 - y_pred)), dim = -1)
      pre_loss = torch.mean(pre_loss, dim = 0)
      print('evaluate: loss %f rebuild loss %f precursors pred loss %f' % (loss, mat_decoder_loss, pre_loss))

if __name__ == "__main__":
  add_options()
  app.run(main)

