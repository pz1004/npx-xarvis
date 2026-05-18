from pathlib import *
import math
import copy
import os
import argparse

import torch
import torch.nn as nn
import snntorch

from npx_define import *
from npx_neuron_type import *
from npx_data_manager import *

class NpxAppCfgGenerator():
  def __init__(self):
    pass

  def generate_predifined_app(self, app_name:str, neuron_type_str:str, dataset_path:Path):
    self.app_name = f'{app_name}_{neuron_type_str}'
    npx_data_manager = NpxDataManager(dataset_name=self.dataset_name, dataset_path=dataset_path, kfold=5)

    # print(self.app_cfg_path)
    self.cfg_parser = NpxCfgParser()
    self.cfg_parser.add_section('network')
    self.cfg_parser.add_option(-1, 'dataset', self.dataset_name)
    self.cfg_parser.add_option(-1, 'neuron_type', neuron_type_str)
    self.cfg_parser.add_option(-1, 'height', npx_data_manager.input_size[1])
    self.cfg_parser.add_option(-1, 'width', npx_data_manager.input_size[2])
    self.cfg_parser.add_option(-1, 'channels', npx_data_manager.input_size[0])
    self.cfg_parser.add_option(-1, 'timesteps', 32)
    self.cfg_parser.add_option(-1, 'classes', npx_data_manager.num_classes)

    # print(self.cfg_parser.layer_info_list)

    if self.app_name.startswith('mnist_l1f'):
      self.gen_fc_section(in_features=14*14, out_features=10, 
                     input_type='spike', output_type='value')

    elif self.app_name.startswith('mnist_l2cf'): # 256x64
      filter_size = 5
      self.gen_conv_section(in_channels=1, out_channels=2, kernel_size=filter_size,
                      input_type='spike', output_type='spike')
      self.gen_fc_section(in_features=10*10*2, out_features=10, 
                     input_type='spike', output_type='value')

    elif self.app_name.startswith('mnist_l2ff'): # 256x64
      self.gen_fc_section(in_features=14*14, out_features=256, 
                     input_type='spike', output_type='spike')
      self.gen_fc_section(in_features=256, out_features=10, 
                     input_type='spike', output_type='value')

    elif self.app_name.startswith('mnist_l3fff'): # 256x64
      self.gen_fc_section(in_features=14*14, out_features=256, 
                     input_type='spike', output_type='spike')
      self.gen_fc_section(in_features=256, out_features=256, 
                     input_type='spike', output_type='spike')
      self.gen_fc_section(in_features=256, out_features=10, 
                     input_type='spike', output_type='value')

    elif self.app_name.startswith('kmnist_l3fff'): # 256x64
      self.gen_fc_section(in_features=14*14, out_features=256,
                     input_type='value', output_type='spike')
      self.gen_fc_section(in_features=256, out_features=256,
                     input_type='spike', output_type='spike')
      self.gen_fc_section(in_features=256, out_features=10,
                     input_type='spike', output_type='value')

    elif self.app_name.startswith('mnist_l3cff'): # 256x64
      filter_size = 5
      self.gen_conv_section(in_channels=1, out_channels=2, kernel_size=filter_size,
                      input_type='spike', output_type='spike')
      self.gen_fc_section(in_features=10*10*2, out_features=256, 
                     input_type='spike', output_type='spike')
      self.gen_fc_section(in_features=256, out_features=10, 
                     input_type='spike', output_type='value')

    elif self.app_name.startswith('mnist_l3ccf'): # 256x64
      filter_size = 5
      self.gen_conv_section(in_channels=1, out_channels=2, kernel_size=filter_size,
                      input_type='spike', output_type='spike')
      self.gen_conv_section(in_channels=2, out_channels=7, kernel_size=filter_size,
                      input_type='spike', output_type='spike')
      self.gen_fc_section(in_features=6*6*7, out_features=10, 
                     input_type='spike', output_type='value')

    elif self.app_name.startswith('fmnist_l2cf'):
      filter_size = 5
      self.gen_conv_section(in_channels=1, out_channels=2, kernel_size=filter_size,
                      input_type='spike', output_type='spike')
      self.gen_fc_section(in_features=10*10*2, out_features=10, 
                     input_type='spike', output_type='value')

    elif self.app_name.startswith('fmnist_l3cff'):
      filter_size = 5
      self.gen_conv_section(in_channels=1, out_channels=2, kernel_size=filter_size,
                      input_type='spike', output_type='spike')
      self.gen_fc_section(in_features=10*10*2, out_features=256, 
                     input_type='spike', output_type='spike')
      self.gen_fc_section(in_features=256, out_features=10, 
                     input_type='spike', output_type='value')

    elif self.app_name.startswith('fmnist_l3ccf'):
      filter_size = 5
      self.gen_conv_section(in_channels=1, out_channels=2, kernel_size=filter_size,
                      input_type='spike', output_type='spike')
      self.gen_conv_section(in_channels=2, out_channels=7, kernel_size=filter_size,
                      input_type='spike', output_type='spike')
      self.gen_fc_section(in_features=6*6*7, out_features=10, 
                     input_type='spike', output_type='value')

    elif self.app_name.startswith('cifar10_l5cccff'): 
      filter_size = 3
      self.gen_conv_section(in_channels=3, out_channels=8, kernel_size=filter_size,
                      input_type='spike', output_type='spike')
      self.gen_conv_section(in_channels=8, out_channels=16, kernel_size=filter_size,
                      input_type='spike', output_type='spike')
      self.gen_conv_section(in_channels=16, out_channels=32, kernel_size=filter_size,
                      input_type='spike', output_type='spike')
      self.gen_fc_section(in_features=26*26*32, out_features=256, 
                     input_type='spike', output_type='spike')
      self.gen_fc_section(in_features=256, out_features=10, 
                     input_type='spike', output_type='value')

    elif self.app_name.startswith('gtsrb_l5cccff'): 
      filter_size = 5
      self.gen_conv_section(in_channels=3, out_channels=10, kernel_size=filter_size,
                      input_type='spike', output_type='spike')
      self.gen_conv_section(in_channels=10, out_channels=15, kernel_size=filter_size,
                      input_type='spike', output_type='spike')
      self.gen_conv_section(in_channels=15, out_channels=25, kernel_size=filter_size,
                      input_type='spike', output_type='spike')
      self.gen_fc_section(in_features=20*20*25, out_features=350, 
                     input_type='spike', output_type='spike')
      self.gen_fc_section(in_features=350, out_features=43, 
                     input_type='spike', output_type='value')

    else:
      assert 0, self.app_name
    
  @property
  def dataset_name(self):
    return self.app_name.split('_')[0]
  
  @property
  def app_cfg_name(self):
    return f'{self.app_name}.cfg'

  def gen_fc_section(self, in_features:int, out_features:int, 
                     input_type='spike', output_type='spike'):
    self.cfg_parser.add_section('fc')
    self.cfg_parser.add_option(-1, 'in_features', in_features)
    self.cfg_parser.add_option(-1, 'out_features', out_features)
    #self.cfg_parser.add_option(-1, 'input_type', input_type)
    #self.cfg_parser.add_option(-1, 'output_type', output_type)

  def gen_conv_section(self, in_channels:int, out_channels:int, 
                      kernel_size=3, stride=1, padding=0,
                      input_type='spike', output_type='spike'):
    self.cfg_parser.add_section('conv')
    self.cfg_parser.add_option(-1, 'in_channels', in_channels)
    self.cfg_parser.add_option(-1, 'out_channels', out_channels)
    self.cfg_parser.add_option(-1, 'kernel_size', kernel_size)
    self.cfg_parser.add_option(-1, 'stride', stride)
    self.cfg_parser.add_option(-1, 'padding', padding)
    #self.cfg_parser.add_option(-1, 'input_type', input_type)
    #self.cfg_parser.add_option(-1, 'output_type', output_type)

  def import_module(self, npx_module):
    self.app_name = npx_module.app_name
    self.cfg_parser = copy.deepcopy(npx_module.cfg_parser)

  def __str__(self) -> str:
    assert self.cfg_parser
    return str(self.cfg_parser)
  
  def __repr__(self) -> str:
    assert self.cfg_parser
    return repr(self.cfg_parser)
  
  def write_file(self, path:Path):
    if path.is_dir():
      app_cfg_path = path / self.app_cfg_name
    else:
      app_cfg_path = path
    app_cfg_path.write_text(str(self))

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='NPX Framework')
  parser.add_argument('-app', '-a', nargs='+', help='example app name', default=[])
  parser.add_argument('-neuron', '-n', nargs='+', help='types of neuron', default=['q8ssf'])
  parser.add_argument('-dataset', '-d', help='dataset directory')
  parser.add_argument('-output', '-o', help='app cfg directory', default='./generated_cfg')

  # check args
  args = parser.parse_args()
  assert args.app
  assert args.neuron
  assert args.dataset
  assert args.output

  exam_app_name_list = args.app

  neuron_list = []
  for neuron_set in args.neuron:
    if '-' in neuron_set:
      train_neuron_str, test_neuron_str = neuron_set.split('-')
    else:
      train_neuron_str = neuron_set
      test_neuron_str = neuron_set
    neuron_list.append((train_neuron_str,test_neuron_str))
  dataset_path = Path(args.dataset).absolute()
  if not dataset_path.is_dir():
    dataset_path.mkdir(parents=True)
  app_cfg_dir_path = Path(args.output).absolute()
  if not app_cfg_dir_path.is_dir():
    app_cfg_dir_path.mkdir(parents=True)

  #print(exam_app_name_list)
  #print(neuron_list)
  for app_name in exam_app_name_list:
    for train_neuron_str, test_neuron_str in neuron_list:
      app_cfg_generator = NpxAppCfgGenerator()
      app_cfg_generator.generate_predifined_app(app_name=app_name, neuron_type_str=train_neuron_str,
                                dataset_path=dataset_path)
      print(app_cfg_generator)
      app_cfg_generator.write_file(app_cfg_dir_path)
