from pathlib import *
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset

import tonic

from collections import Counter
from requests import get
from npx_cfg_parser import *
from npx_define import *

from npx_speechcommands import *

def download(url:str, root:Path,file_name = None):
  if not file_name:
    file_name = url.split('/')[-1]

  path = root / file_name
  with open(path, "wb") as file:
          response = get(url)
          file.write(response.content)

def intstr_to_tuple(intstr):
  intstr_list = intstr.split(',')
  int_list = []
  for i in range(len(intstr_list)):
    int_list.append(int(intstr_list[i]))
  return tuple(int_list)

class NpxDataManager():
  def __init__(self, npx_define:NpxDefine, dataset_path:Path, kfold:int=None):

    self.name = npx_define.cfg_parser.preprocess_info['input']
    is_open_dataset = False
    if self.name.endswith('_opendataset'):
      self.name = self.name[:-len('_opendataset')]
      opendatatset_list = ['mnist', 'kmnist', 'fmnist', 'cifar10', 'gtsrb', 'dvsgesture', 'speechcommands']
      assert self.name in opendatatset_list, self.name
      is_open_dataset = True
    elif self.name.endswith('_dataset'):
      self.name = self.name[:-len('_dataset')]
    
    self.download_path = dataset_path / self.name
    if is_open_dataset:
      self.download_path.mkdir(parents=True, exist_ok=True)
    else:
      assert self.download_path.is_dir(), self.download_path
    
    if kfold==None:
      kfold = 5
    assert kfold>=4, kfold
    self.kfold = kfold
    self.fair_distribution = False

    if self.name=='mnist':
      self.raw_data_format = DataFormat.MATRIX3D
      self.data_format = DataFormat.MATRIX3D
      self.step_generation = npx_define.cfg_parser.preprocess_info.setdefault('step_generation','direct')
      self.timesteps = npx_define.cfg_parser.preprocess_info.setdefault('timesteps',4)
      #value = NpxCfgParser.find_option_value(npx_define.cfg_parser.preprocess_info, 'resize', '14,14')
      #self.resize = intstr_to_tuple(value)
      transform = transforms.Compose([
        #transforms.Resize((14, 14)),
        #transforms.Resize(self.resize),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))])
      dataset_train_and_val = datasets.MNIST(root=self.download_path, train=True, download=True, transform=transform)
      self.dataset_test = datasets.MNIST(root=self.download_path, train=False, download=True, transform=transform)
    elif self.name=='kmnist':
      self.raw_data_format = DataFormat.MATRIX3D
      self.data_format = DataFormat.MATRIX3D
      self.step_generation = npx_define.cfg_parser.preprocess_info.setdefault('step_generation','direct')
      self.timesteps = npx_define.cfg_parser.preprocess_info.setdefault('timesteps',4)
      #value = NpxCfgParser.find_option_value(npx_define.cfg_parser.preprocess_info, 'resize', '14,14')
      #self.resize = intstr_to_tuple(value)
      transform = transforms.Compose([
        #transforms.Resize((14, 14)),
        #transforms.Resize(self.resize),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))])
      dataset_train_and_val = datasets.KMNIST(root=self.download_path, train=True, download=True, transform=transform)
      self.dataset_test = datasets.KMNIST(root=self.download_path, train=False, download=True, transform=transform)
    elif self.name=='fmnist':
      self.raw_data_format = DataFormat.MATRIX3D
      self.data_format = DataFormat.MATRIX3D
      self.step_generation = npx_define.cfg_parser.preprocess_info.setdefault('step_generation','direct')
      self.timesteps = npx_define.cfg_parser.preprocess_info.setdefault('timesteps',4)
      #value = NpxCfgParser.find_option_value(npx_define.cfg_parser.preprocess_info, 'resize', '14,14')
      #self.resize = intstr_to_tuple(value)
      transform = transforms.Compose([
        #transforms.Resize((14, 14)),
        #transforms.Resize(self.resize),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))])
      download('https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz', self.download_path / 'FashionMNIST/raw' )
      download('https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz', self.download_path / 'FashionMNIST/raw' )
      download('https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz', self.download_path / 'FashionMNIST/raw' )
      download('https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz', self.download_path / 'FashionMNIST/raw' )
      dataset_train_and_val = datasets.FashionMNIST(root=self.download_path, train=True, download=True, transform=transform)
      self.dataset_test = datasets.FashionMNIST(root=self.download_path, train=False, download=True, transform=transform)
    elif self.name=='cifar10':
      self.raw_data_format = DataFormat.MATRIX3D
      self.data_format = DataFormat.MATRIX3D
      self.step_generation = npx_define.cfg_parser.preprocess_info.setdefault('step_generation','direct')
      self.timesteps = npx_define.cfg_parser.preprocess_info.setdefault('timesteps',4)
      #value = NpxCfgParser.find_option_value(npx_define.cfg_parser.preprocess_info, 'resize', '32,32')
      #self.resize = intstr_to_tuple(value)
      transform = transforms.Compose([
        #transforms.Resize((32, 32)),
        #transforms.Resize(self.resize),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (1, 1, 1))])
      dataset_train_and_val = datasets.CIFAR10(root=self.download_path, train=True, download=True, transform=transform)
      self.dataset_test = datasets.CIFAR10(root=self.download_path, train=False, download=True, transform=transform)
    elif self.name=='gtsrb':
      self.raw_data_format = DataFormat.MATRIX3D
      self.data_format = DataFormat.MATRIX3D
      self.step_generation = npx_define.cfg_parser.preprocess_info.setdefault('step_generation','direct')
      self.timesteps = npx_define.cfg_parser.preprocess_info.setdefault('timesteps',4)
      #value = NpxCfgParser.find_option_value(npx_define.cfg_parser.preprocess_info, 'resize', '32,32')
      #self.resize = intstr_to_tuple(value)
      transform = transforms.Compose([
        transforms.Resize((32, 32)),
        #transforms.Resize(self.resize),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (1, 1, 1))])
      dataset_train_and_val = datasets.GTSRB(root=self.download_path, split='train', download=True, transform=transform)
      self.dataset_test = datasets.GTSRB(root=self.download_path, split='test', download=True, transform=transform)
    elif self.name=='dvsgesture':
      self.raw_data_format = DataFormat.DVS
      self.data_format = DataFormat.MATRIX4D
      self.sensor_size = tonic.datasets.DVSGesture.sensor_size
      #denoise_transform = tonic.transforms.Denoise(filter_time=10000)
      #value = NpxCfgParser.find_option_value(npx_define.cfg_parser.preprocess_info, 'resize', '128,128')
      #self.resize = intstr_to_tuple(value)
      #self.resize = (2,) + self.resize
      self.timesteps = npx_define.cfg_parser.preprocess_info.setdefault('timesteps',32)
      frame_transform = tonic.transforms.ToFrame(sensor_size=self.sensor_size, n_time_bins=self.timesteps)
      all_transform = transforms.Compose([
        frame_transform])
      train_set = tonic.datasets.DVSGesture(save_to=self.download_path, transform=all_transform, train=True)
      test_set = tonic.datasets.DVSGesture(save_to=self.download_path, transform=all_transform, train=False)
      dataset_train_and_val = tonic.DiskCachedDataset(train_set, cache_path=self.download_path/'cache/dvsgesture/train')
      self.dataset_test = tonic.DiskCachedDataset(test_set, cache_path=self.download_path/'cache/dvsgesture/test')
      self.dataset_test_raw = tonic.datasets.DVSGesture(save_to=self.download_path, train=False)
    elif self.name=='speechcommands':
      self.raw_data_format = DataFormat.WAVEFORM
      self.data_format = DataFormat.MATRIX3D
      self.timesteps = npx_define.cfg_parser.preprocess_info.setdefault('timesteps',4)
      self.feature = npx_define.cfg_parser.preprocess_info.setdefault('feature', None)
      self.transform = None
      self.sample_rate = 16000
      if self.feature=='mel_spectrogram':
        self.num_samples = npx_define.cfg_parser.preprocess_info.setdefault('mel_spectrogram.num_samples',16000)
        self.sample_rate = npx_define.cfg_parser.preprocess_info.setdefault('mel_spectrogram.sample_rate',16000)
        self.n_fft = npx_define.cfg_parser.preprocess_info.setdefault('mel_spectrogram.n_fft',512)
        self.win_length = npx_define.cfg_parser.preprocess_info.setdefault('mel_spectrogram.win_length',400)
        self.hop_length = npx_define.cfg_parser.preprocess_info.setdefault('mel_spectrogram.hop_length',160)
        self.n_mels = npx_define.cfg_parser.preprocess_info.setdefault('mel_spectrogram.n_mels',40)
        self.transform = NpxMelSpectrogram(
              num_samples=self.num_samples,
              sample_rate=self.sample_rate,
              n_fft=self.n_fft,
              win_length=self.win_length,
              hop_length=self.hop_length,
              n_mels=self.n_mels
        )
      train_dataset = NpxSpeechCommandsPreprocess(root=self.download_path, subset='training', transform=self.transform, target_sr=self.sample_rate)
      val_dataset   = NpxSpeechCommandsPreprocess(root=self.download_path, subset='validation', transform=self.transform, target_sr=self.sample_rate)
      dataset_train_and_val = train_dataset + val_dataset
      self.dataset_test = NpxSpeechCommandsPreprocess(root=self.download_path, subset='testing', transform=self.transform, target_sr=self.sample_rate)
      self.dataset_test_raw = NpxSpeechCommandsPreprocess(root=self.download_path, subset='testing', transform=None, target_sr=self.sample_rate)
    else:
      print(f'Custom Dataset: {self.name}')
      assert self.download_path.is_dir(), f"Dataset does not exist in the path: {self.download_path}"
      self.input_type = npx_define.cfg_parser.preprocess_info.setdefault('input_type', 'image')
      if self.input_type == 'waveform':
        self.raw_data_format = DataFormat.WAVEFORM
      else:
        self.raw_data_format = DataFormat.MATRIX3D
      # if self.input_type == 'waveform' ??? dododo
      self.data_format = DataFormat.MATRIX3D
      self.step_generation = npx_define.cfg_parser.preprocess_info.setdefault('step_generation','direct')
      self.timesteps = npx_define.cfg_parser.preprocess_info.setdefault('timesteps',4)
      #value = NpxCfgParser.find_option_value(npx_define.cfg_parser.preprocess_info, 'resize', '14,14')
      #self.resize = intstr_to_tuple(value)
      transform = transforms.Compose([
        #transforms.Resize(self.resize),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))])
        #transforms.PILToTensor(),
        #transforms.Lambda(lambda t: t.to(torch.float32))])
      dataset_train_and_val = datasets.MNIST(root=self.download_path, train=True, download=False, transform=transform)
      self.dataset_test = datasets.MNIST(root=self.download_path, train=False, download=False, transform=transform)
    
    if self.fair_distribution:
      labeled_dataset_dict = {}
      for data, label in dataset_train_and_val:
        labeled_dataset = labeled_dataset_dict.get(label)
        if labeled_dataset:
          labeled_dataset.append((data, label))
        else:
          labeled_dataset_dict[label] = [(data, label)]
      self.dataset_both_list = None
      for labeled_dataset in labeled_dataset_dict.values():
        chunk_size = int(len(labeled_dataset)/self.kfold)
        chunk_size_list = []
        for i in range(0, kfold-1):
          chunk_size_list.append(chunk_size)
        last_chunk_size = len(labeled_dataset) - (chunk_size*(kfold-1))
        chunk_size_list.append(last_chunk_size)
        splited_labeled_dataset = torch.utils.data.random_split(labeled_dataset, chunk_size_list)
        if not self.dataset_both_list:
          self.dataset_both_list = splited_labeled_dataset
        else:
          for i in range(len(self.dataset_both_list)):
            self.dataset_both_list[i] += splited_labeled_dataset[i]
    else:
      chunk_size = int(len(dataset_train_and_val)/self.kfold)
      chunk_size_list = []
      for i in range(0, kfold-1):
        chunk_size_list.append(chunk_size)
      last_chunk_size = len(dataset_train_and_val) - (chunk_size*(kfold-1))
      chunk_size_list.append(last_chunk_size)
      self.dataset_both_list = torch.utils.data.random_split(dataset_train_and_val, chunk_size_list)

  def setup_loader(self, kfold_index:int, batch_size:int=100):
    assert kfold_index <= len(self.dataset_both_list), len(self.dataset_both_list)
    self.dataset_val = self.dataset_both_list[kfold_index]
    dataset_train_subset_list = self.dataset_both_list[:kfold_index] + self.dataset_both_list[kfold_index+1:]
    self.dataset_train = ConcatDataset(dataset_train_subset_list)
    print('The number of train dataset:', len(self.dataset_train))
    print('The number of valid dataset:', len(self.dataset_val))
    if self.name=='dvsgesture':
      collate_fn = tonic.collation.PadTensors(batch_first=False)
    else:
      collate_fn = None
    self.loader_list = (DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn),
                        DataLoader(self.dataset_val, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn),
                        DataLoader(self.dataset_test, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn))

  @property
  def train_loader(self):
    return self.loader_list[0]
  @property
  def val_loader(self):
    return self.loader_list[1]
  @property
  def test_loader(self):
    return self.loader_list[2]
  
  @property
  def input_size(self):
    return self.dataset_test[0][0].shape
  
  @property
  def num_classes(self):
    return len(dict(Counter(sample_tup[1] for sample_tup in self.dataset_test)))
