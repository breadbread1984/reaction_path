#!/usr/bin/python3

from absl import flags, app
from models import MaterialDecoder, PrecursorPredictor

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('task', enum_values = {'mat', 'reaction_pre'}, default = None, help = 'training task')
  flags.DEFINE_string('dataset', default = None, help = 'path to dataset')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to ckpt')

def train_mat():
  pass

def train_reaction_pre():
  pass

def main(unused_argv):
  if FLAGS.task == 'mat':
    train_mat()
  elif FLAGS.task == 'reaction_pre':
    train_reaction_pre()

if __name__ == "__main__":
  add_options()
  app.run(main)

