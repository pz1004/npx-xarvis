from pathlib import *
import math
import copy
import os

import torch
import torch.nn as nn
import snntorch
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt
from snntorch import spikegen

from npx_define import *
from npx_neuron_type import *
from npx_cfg_parser import *

import npx_app_cfg_generator

class PotentialResult():
  def __init__(self, pacc:int=0, nacc:int=0, neuron_type:NpxNeuronType=None):
    self.neuron_type = neuron_type
    self.pacc = pacc
    self.nacc = nacc

  def update(self, result):
    self.pacc = max(self.pacc,result.pacc)
    self.nacc = max(self.nacc,result.nacc)
    
  # self.training is changed by train() and eval()

  @property
  def max(self):
    return max(self.pacc,self.nacc)

  def __repr__(self):
    assert self.neuron_type
    return str((self.pacc,self.nacc,math.ceil(self.max/self.neuron_type.qscale)))
  
class NpxModule(nn.Module):
  def __init__(self, app_cfg_path:Path, neuron_type_class=NpxNeuronType):
    super().__init__()
    self.neuron_type_class = neuron_type_class
    self.app_cfg_path = app_cfg_path
    if self.app_cfg_path and self.app_cfg_path.is_file():
      self.cfg_parser = NpxCfgParser()
      self.cfg_parser.parse_file(self.app_cfg_path)
      
      self.layer_sequence = []
      self.gen_layer_sequence(self.cfg_parser.layer_info_list)
      # print(net_option, layer_option_list)
    self.is_network_quantized = False
  
  def global_config(self, option_name:str):
    return self.cfg_parser.global_info.get(option_name)
      
  @property
  def app_name(self):
    return self.app_cfg_path.stem
  
  @property
  def dataset_name(self):
    return self.dataset
      
  @property
  def num_layer(self):
    return len(self.cfg_parser.layer_info_list)
  
  @property
  def input_size(self):
    return self.cfg_parser.global_info['input_size']
  
  @property
  def timesteps(self):
    return self.cfg_parser.preprocess_info['timesteps']

  def backup_epoch_cfg(self, cfg_path:Path, overwrite:bool=False):
    assert overwrite or (not cfg_path.is_file()), cfg_path
    app_cfg_generator = npx_app_cfg_generator.NpxAppCfgGenerator()
    app_cfg_generator.import_module(self)
    app_cfg_generator.write_file(cfg_path)
  
  def backup_raw_cfg(self, cfg_path:Path):
    contents = self.app_cfg_path.read_text()
    if cfg_path.is_file():
      previous_contents = cfg_path.read_text()
      assert contents==previous_contents
    cfg_path.write_text(contents)

  def backup_cfg(self, npx_define:NpxDefine, epoch_index:int):
    self.backup_raw_cfg(npx_define.get_parameter_raw_cfg_path())
    self.backup_epoch_cfg(npx_define.get_parameter_epoch_cfg_path(epoch_index),True)

  def forward(self, x:Tensor):
    last_tensor = x
    for i, layer in enumerate(self.layer_sequence):
      if (type(layer)==nn.Linear) or (type(layer)==nn.Conv2d):
        last_tensor = self.forward_layer(i, layer, last_tensor)
      elif type(layer)==snntorch.Leaky:
        last_tensor = self.forward_neuron(i, layer, last_tensor)
      else:
        last_tensor = layer(last_tensor)

    return last_tensor

  def forward_layer(self, i:int, layer, x:Tensor):
    if self.training and layer.neuron_type:
      original_tensor = copy.deepcopy(layer.weight.data)
      layer.neuron_type.synch_with_threshold(layer.neuron.threshold)
      layer.neuron_type.update_mapped_fvalue(layer.weight.data)
      qtensor = layer.neuron_type.quantize_tensor(layer.weight.data, bounded=True)
      layer.weight.data = layer.neuron_type.dequantize_tensor(qtensor)
    current = layer(x)
    if self.training and layer.neuron_type:
      layer.weight.data = original_tensor
      layer.neuron_type.clamp_weight_(layer.weight.data, self.is_network_quantized)
    return current
      
  def forward_neuron(self, i:int, neuron, x:Tensor):
    #if self.training and self.can_learn_neural_threshold and self.does_neuron_learn_threshold(neuron):
    current = neuron(x)
    neuron_type:NpxNeuronType = neuron.neuron_type
    if neuron_type:
      neuron_type.clamp_mem_(neuron.mem, self.is_network_quantized)
      if neuron_type.learn_beta:
        neuron.beta.data.fill_(neuron_type.quantize_beta(neuron.beta.data.float()))
    return current
      
  def print_parameter(self):
    for layer in self.layer_sequence:
      if (type(layer)==nn.Linear) or (type(layer)==nn.Conv2d):
        print(layer.weight)
      elif type(layer)==snntorch.Leaky:
        print(layer.threshold)

  def quantize_network(self):
    assert not self.training
    self.is_network_quantized = True
    for layer in self.layer_sequence:
      if (type(layer)==nn.Linear) or (type(layer)==nn.Conv2d):
        qtensor = layer.neuron_type.quantize_tensor(layer.weight.data, bounded=True)
        layer.weight.data = qtensor.tensor.float()
      elif type(layer)==snntorch.Leaky:
        qtensor = layer.neuron_type.quantize_tensor(layer.threshold, bounded=False)
        layer.threshold = type(layer.threshold)(qtensor.tensor.float())

  def write_parameter(self, path:Path):
    assert path.parent.is_dir(), path
    line_list = []
    for layer in self.layer_sequence:
      if (type(layer)==nn.Linear) or (type(layer)==nn.Conv2d):
        line_list.append(str(layer.weight.tolist()))
      elif type(layer)==snntorch.Leaky:
        line_list.append(str(layer.threshold.tolist()))
        line_list.append(str(layer.beta.tolist()))
    path.write_text('\n'.join(line_list))
  
  def dicide_option_value(self, layer_option:dict, option_name:str, default_value):
    global_value = self.global_config(option_name)
    local_value = layer_option.get(option_name)
    if local_value!=None:
      result = local_value
    elif global_value!=None:
      result = global_value
    else:
      result = default_value
    assert result!=None
    layer_option[option_name] = result
    return result

  def gen_layer_sequence(self, layer_option_list):
    #num_layer = len(layer_option_list)
    not_assigned_layer_list = []
    for i, layer_option in enumerate(layer_option_list):
      #if i == (num_layer-1):
      #  neuron_output = True
      #else:
      #  neuron_output = False
      
      if layer_option.name == 'Linear':
        # synapse option
        in_features = layer_option.setdefault('in_features', 1)
        out_features = layer_option.setdefault('out_features', 1)
        # print(in_features, out_features)

        layer = nn.Linear(in_features, out_features, bias=False)
        not_assigned_layer_list.append((layer, layer_option))
        
      elif layer_option.name == 'Conv2d':
        # synapse option
        in_channels = layer_option.setdefault('in_channels', 1)
        out_channels = layer_option.setdefault('out_channels', 1)
        kernel_size = layer_option.setdefault('kernel_size', 3)
        stride = layer_option.setdefault('stride', 1)
        padding = layer_option.setdefault('padding', 0)
        # print(in_channels, out_channels, kernel_size, stride, padding)

        layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        not_assigned_layer_list.append((layer, layer_option))

      elif layer_option.name == 'MaxPool2d':
        kernel_size = layer_option.setdefault('kernel_size', 1)
        stride = layer_option.setdefault('stride', kernel_size)
        padding = layer_option.setdefault('padding', 0)

        layer = nn.MaxPool2d(kernel_size, stride, padding)
        not_assigned_layer_list.append((layer, layer_option))
          
      elif layer_option.name == 'AvgPool2d':
        kernel_size = layer_option.setdefault('kernel_size', 1)
        stride = layer_option.setdefault('stride', kernel_size)
        padding = layer_option.setdefault('padding', 0)

        layer = nn.AvgPool2d(kernel_size, stride, padding)
        not_assigned_layer_list.append((layer, layer_option))

      elif layer_option.name == 'Flatten':
        layer = nn.Flatten()
        not_assigned_layer_list.append((layer, layer_option))

      elif layer_option.name == 'Leaky':
        #layer = self.make_neuron(layer_option, neuron_output)
        layer = self.make_neuron(layer_option, False)
        assert layer.neuron_type
        for previous_layer, previous_layer_option in not_assigned_layer_list:
          previous_layer.neuron = layer
          previous_layer.neuron_type = layer.neuron_type
          assert 'neuron_type' not in previous_layer_option
          previous_layer_option['neuron_type'] = layer.neuron_type.name
        not_assigned_layer_list = []
      else:
        assert 0

      self.add_module('layer' + str(i), layer)
      self.layer_sequence.append(layer)
    assert len(not_assigned_layer_list)==0

  def make_neuron(self, layer_option, neuron_output):
    neuron_type_str = self.dicide_option_value(layer_option, 'neuron_type', 'q8ssf')
    neuron_type = self.neuron_type_class(neuron_type_str)
    
    mapped_fvalue = self.dicide_option_value(layer_option, 'mapped_fvalue', neuron_type.mapped_fvalue)
    neuron_type.mapped_fvalue = mapped_fvalue
    layer_option['mapped_fvalue'] = neuron_type.mapped_fvalue
    
    beta = self.dicide_option_value(layer_option, 'beta', 1.0)
    beta = neuron_type.quantize_beta(beta)
    layer_option['beta'] = beta
    
    if neuron_type.can_learn_beta:
      learn_beta = self.dicide_option_value(layer_option, 'learn_beta', False)
    else:
      layer_option['learn_beta'] = False
    neuron_type.learn_beta = learn_beta
    
    reset_mechanism = self.dicide_option_value(layer_option, 'reset_mechanism', 'subtract')    
    reset_delay = self.dicide_option_value(layer_option, 'reset_delay', True)
    
    threshold = self.dicide_option_value(layer_option, 'threshold', 1.0)
    if neuron_type.can_learn_threshold:
      learn_threshold = self.dicide_option_value(layer_option, 'learn_threshold', False)
    else:
      layer_option['learn_threshold'] = False
      learn_threshold = False
    neuron_type.learn_threshold = learn_threshold
      
    #spike_grad = surrogate.fast_sigmoid(slope=25)
    spike_grad = None
    neuron = snntorch.Leaky(beta=beta, learn_beta=learn_beta, spike_grad=spike_grad, threshold=threshold, learn_threshold=learn_threshold,
                            init_hidden=True, reset_delay=reset_delay, reset_mechanism=reset_mechanism, output=neuron_output)
    neuron.neuron_type = neuron_type
    
    return neuron
