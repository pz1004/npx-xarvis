import os
import argparse
import time
import shutil
import random
import copy

import numpy as np
from pathlib import *
from tqdm.auto import tqdm
from collections import namedtuple

from npx_define import *
from npx_data_manager import *
from npx_module import *
from npx_converter import write_data_aligned_by_4bytes

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import tonic
from npx_to_frame import *

def get_sample(dataset:Dataset, num_sample:int):
  num_data = len(dataset) 
  random_sample_idx_list = random.sample(range(num_data), num_sample)
  #random_sample_idx_list = [0]
  sample_list = []
  for idx in random_sample_idx_list:
    data, target = dataset[idx]
    sample_list.append((data, target))
    #print(data, target)

  return sample_list 

def convert_dvs_dtype(events:np.ndarray) -> torch.Tensor:
  dtype = np.int32
  type_converted_events = events.astype(np.dtype([("x", dtype), ("y", dtype), ("p", dtype), ("t", dtype)])).view((dtype, 4))
  return torch.Tensor(type_converted_events).to(torch.int32)
  #return events.astype(np.dtype([("x", np.int16), ("y", np.int16), ("p", np.int8), ("t", np.int32)]))

def trasform_to_frame(events, sensor_size, n_time_bins: int, overlap: float=0.0, use_tonic=False):
  if use_tonic:
    frames = tonic.functional.to_frame_numpy(events, sensor_size=sensor_size, n_time_bins=n_time_bins, overlap=overlap)
    frames = torch.Tensor(frames)
    frames = torch.unsqueeze(frames, dim=1)
  else:
    data = convert_dvs_dtype(events)
    frames = npx_to_frame(data, sensor_size, n_time_bins, overlap)
    frames = frames.to(device)
  return frames

def list_to_string(lst):
    if isinstance(lst[0], list):  # ���� ������ ���������� ����
        return '\n'.join(list_to_string(sublist) for sublist in lst)
    else:  # ������ ����
        return ' '.join(map(str, lst))

def save_sample(npx_define:NpxDefine, data:torch.Tensor, target:np.ndarray, i:int, data_format:DataFormat, data_type:torch.dtype):
    riscv_sample_bin_path = npx_define.get_riscv_sample_bin_path(i=i, data_format=data_format)
    with open(riscv_sample_bin_path, "wb") as data_file:
      write_data_aligned_by_4bytes(data_file, data, data_type)
      data_file.write(target)
    riscv_sample_text_path = riscv_sample_bin_path.parent / f'{riscv_sample_bin_path.stem}.txt'
    line_list = []
    line_list.append(list_to_string(data.tolist()))
    line_list.append(str(target.tolist()))
    riscv_sample_text_path.write_text('\n'.join(line_list))

def _generate_testvector_for_dvs_input(npx_module:NpxModule, npx_define:NpxDefine, npx_data_manager:NpxDataManager, num_sample:int, sample_only:bool):
  sample_list = get_sample(dataset=npx_data_manager.dataset_test_raw, num_sample=num_sample)

  for i, (data, target) in enumerate(sample_list):
    frames = trasform_to_frame(data, npx_data_manager.sensor_size, npx_data_manager.timesteps, use_tonic=False)
    target = torch.Tensor([target]).to(torch.int32).numpy()

    # save dvs data sample
    dvs_data = convert_dvs_dtype(data)
    save_sample(npx_define, dvs_data, target, i, DataFormat.DVS, torch.int32)

    # save frames data sample
    #save_sample(npx_define, frames, target, i, DataFormat.MATRIX4D, torch.uint8)

    # resize
    if npx_module.input_size != frames.shape[-2:]:
      size = (frames.shape[-3],) + npx_module.input_size
      resized_frames = nn.functional.interpolate(frames, size=size)
      #frames = nn.functional.interpolate(frames, size=npx_module.input_size, mode='bilinear')
    else:
      resized_frames = frames
    #save_sample(npx_define, resized_frames, target, i, DataFormat.MATRIX4D, torch.uint8)    

    # save testvector
    if sample_only:
      riscv_testvector_bin_path = None
    else:
      riscv_testvector_bin_path = npx_define.get_riscv_layeroutput_bin_path(i=i)
    #if not riscv_testvector_bin_path.is_file():
    if True:
      assert(npx_define.timesteps == npx_data_manager.timesteps)
      spk_rec = manual_forward_pass(npx_module, resized_frames, tv_bin_path=riscv_testvector_bin_path)
      print('output spikes: ', spk_rec.data)
      print('output spikes: ', spk_rec.sum(0).data)
      inference_class_id = spk_rec.sum(0).argmax()
      print('class id from inference: ', int(inference_class_id))
      print('class id from dataset: ', int(target[0]))

def _generate_testvector_for_matrix3d_input(npx_module:NpxModule, npx_define:NpxDefine, npx_data_manager:NpxDataManager, num_sample:int, sample_only:bool):
  sample_list = get_sample(dataset=npx_data_manager.dataset_test, num_sample=num_sample)
  print('sample:')
  print(sample_list)

  spike_input = False
  for i, (data, target) in enumerate(sample_list):
    data = torch.unsqueeze(data, dim=0).to(device)
    target = torch.Tensor([target]).to(torch.int32).numpy()
    #target = target.to(torch.int32).numpy()

    # save value(raw data) sample
    raw_data = (data*255).round()
    save_sample(npx_define, raw_data, target, i, DataFormat.MATRIX3D, torch.uint8)

    num_steps = npx_define.timesteps
    if spike_input:
      if npx_module.input_size != data.shape[-2:]:
        resized_data = nn.functional.interpolate(data, size=npx_module.input_size)
      else:
        resized_data = data
      input_data = spikegen.rate(resized_data, num_steps=num_steps)
      save_sample(npx_define, input_data, target, i, DataFormat.MATRIX4D, torch.int8)
    else :
      if i==0:
        scale_threshold_for_first_leaky_layer(npx_module, 255)
      if npx_module.input_size != raw_data.shape[-2:]:
        resized_data = nn.functional.interpolate(raw_data, size=npx_module.input_size)
      else:
        resized_data = raw_data
      input_data = resized_data.repeat(tuple([num_steps] + torch.ones(len(resized_data.size()), dtype=int).tolist()))

    # save spike(rate coded data) sample and testvector
    if sample_only:
      riscv_testvector_bin_path = None
    else:
      riscv_testvector_bin_path = npx_define.get_riscv_layeroutput_bin_path(i=i)
    #if not riscv_testvector_bin_path.is_file():
    if True:
      spk_rec = manual_forward_pass(npx_module, input_data, tv_bin_path=riscv_testvector_bin_path)
      print('output spikes: ', spk_rec.data)
      print('accumulated output spikes: ', spk_rec.sum(0).data)
      inference_class_id = spk_rec.sum(0).argmax()
      print('class id from inference: ', int(inference_class_id))
      print('class id from dataset: ', int(target[0]))

def _generate_testvector_for_waveform_input(npx_module:NpxModule, npx_define:NpxDefine, npx_data_manager:NpxDataManager, num_sample:int, sample_only:bool):
  sample_list = get_sample(dataset=npx_data_manager.dataset_test_raw, num_sample=num_sample)

  num_steps = npx_define.timesteps
  for i, (waveform, target) in enumerate(sample_list):
    target = torch.Tensor([target]).to(torch.int32).numpy()

    # save waveform data sample
    raw_waveform = (waveform * 32768).clamp(-32768, 32767)
    save_sample(npx_define, raw_waveform, target, i, DataFormat.WAVEFORM, torch.int16)

    if npx_data_manager.feature=='mel_spectrogram':
      sample_rate = npx_define.cfg_parser.preprocess_info['mel_spectrogram.sample_rate']
      n_fft = npx_define.cfg_parser.preprocess_info['mel_spectrogram.n_fft']
      win_length = npx_define.cfg_parser.preprocess_info['mel_spectrogram.win_length']
      hop_length = npx_define.cfg_parser.preprocess_info['mel_spectrogram.hop_length']
      n_mels = npx_define.cfg_parser.preprocess_info['mel_spectrogram.n_mels']
      mel_spectrogram = NpxMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels
      )
      preprocessed_data = mel_spectrogram(waveform.squeeze(0)).unsqueeze(0)
    else:
      preprocessed_data = waveform.unsqueeze(0)
    print(preprocessed_data)
    input_data = preprocessed_data.repeat(tuple([num_steps] + torch.ones(len(preprocessed_data.size()), dtype=int).tolist()))

    # save testvector
    if sample_only:
      riscv_testvector_bin_path = None
    else:
      riscv_testvector_bin_path = npx_define.get_riscv_layeroutput_bin_path(i=i)
    #if not riscv_testvector_bin_path.is_file():
    if True:
      assert(npx_define.timesteps == npx_data_manager.timesteps)
      spk_rec = manual_forward_pass(npx_module, input_data, tv_bin_path=riscv_testvector_bin_path)
      print('output spikes: ', spk_rec.data)
      print('output spikes: ', spk_rec.sum(0).data)
      inference_class_id = spk_rec.sum(0).argmax()
      print('class id from inference: ', int(inference_class_id))
      print('class id from dataset: ', int(target[0]))

def generate_testvector(npx_define:NpxDefine, npx_data_manager:NpxDataManager, num_sample:int, sample_only:bool):
  print('\n[TEST VECTOR]', npx_define.app_name)

  npx_module = NpxModule(app_cfg_path=npx_define.app_cfg_path).to(device)
  npx_module.eval()
  riscv_parameter_path = npx_define.get_riscv_parameter_path(is_quantized=True)
  assert riscv_parameter_path.exists(), riscv_parameter_path
  npx_module.load_state_dict(torch.load(riscv_parameter_path, weights_only=False)['npx_module'])
  
  npx_define.riscv_tv_path.mkdir(exist_ok=True, parents=True)

  if npx_data_manager.raw_data_format == DataFormat.MATRIX3D:
    _generate_testvector_for_matrix3d_input(npx_module, npx_define, npx_data_manager, num_sample, sample_only)
  elif npx_data_manager.raw_data_format == DataFormat.DVS:
    _generate_testvector_for_dvs_input(npx_module, npx_define, npx_data_manager, num_sample, sample_only)
  elif npx_data_manager.raw_data_format == DataFormat.WAVEFORM:
    _generate_testvector_for_waveform_input(npx_module, npx_define, npx_data_manager, num_sample, sample_only)
  else:
    assert 0, npx_data_manager.raw_data_format

debug_check_cpu_vs_gpu_result = False
debug_print_layer_outout = False

def scale_threshold_for_first_leaky_layer(npx_module:NpxModule, scale:float=1.0):
  for i, layer in enumerate(npx_module.layer_sequence):
    if type(layer)==snntorch.Leaky:
      with torch.no_grad():
        layer.threshold *= scale
      break

# for saving test vector
def manual_forward_pass(npx_module:NpxModule, data, tv_bin_path:Path=None):
  spk_rec = []
  utils.reset(npx_module)
  if debug_check_cpu_vs_gpu_result:
    cmp_npx_module = copy.deepcopy(npx_module).to('cpu')
  if tv_bin_path:
    tv_file = open(tv_bin_path, 'wb')
    tv_text_path = tv_bin_path.parent / f'{tv_bin_path.stem}.txt'
    line_list = []

  num_steps = npx_module.timesteps
  for step in range(num_steps):
    prev_layer_type = snntorch.Leaky
    last_tensor = data[step]
    cmp_tensor = data[step].to('cpu')

    for i, layer in enumerate(npx_module.layer_sequence):
      last_tensor = layer(last_tensor)
      if type(layer) == nn.AvgPool2d:
        #last_tensor = last_tensor.to(torch.int32).to(torch.float)
        tv_tensor = last_tensor*layer.kernel_size*layer.kernel_size
      else:
        tv_tensor = last_tensor

      if tv_bin_path:
        write_data_aligned_by_4bytes(tv_file, tv_tensor, torch.int32)
        line_list.append(str(tv_tensor.tolist()))

      prev_layer_type = type(layer)

      if debug_check_cpu_vs_gpu_result:
        cmp_layer = cmp_npx_module.layer_sequence[i]
        cmp_tensor = cmp_layer(cmp_tensor).round()
        if (cmp_tensor==last_tensor.to('cpu')).all():
          print('###### ok #########################################################')
        else:
          print('###### not ok #####################################################')
          print(cmp_tensor.flatten()[0:10])
          print(last_tensor.flatten().to('cpu')[0:10])
          print(cmp_tensor.dtype)
          print(last_tensor.dtype)
          print(cmp_tensor==last_tensor.to('cpu'))

      if debug_print_layer_outout:
        print('@@@ step', step,' layer', i, type(layer))
        if last_tensor.dim()>3:
          print(last_tensor[0][0])
        else :
          print(last_tensor[0][0:20])
        print(last_tensor.shape)
    #print('last Leaky mem', npx_module.layer_sequence[-1].mem)
    spk_rec.append(last_tensor)
  if tv_bin_path:
    tv_file.close()
    tv_text_path.write_text('\n'.join(line_list))
  return torch.stack(spk_rec)

use_cuda = True and torch.cuda.is_available()
#device = torch.device("cuda:0" if use_cuda else "cpu")
#device = torch.device("cuda:1" if use_cuda else "cpu")
device = torch.device("cpu")

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='NPX Framework')
  parser.add_argument('-cfg', '-c', nargs='+', help='app cfg file name')
  parser.add_argument('-cmd', nargs='+', help='command')
  parser.add_argument('-dataset', '-d', help='dataset directory')
  parser.add_argument('-output', '-o', help='output directory')
  parser.add_argument('-sample', '-s', help='number of sample', default=10)
  parser.add_argument('--sample_only', action='store_true', help='if stores sample only or with layer outputs')

  # check args
  args = parser.parse_args()
  assert args.cfg
  assert args.cmd
  assert args.output

  app_cfg_list = args.cfg
  cmd_list = args.cmd
  num_sample = int(args.sample)
  
  output_path = Path(args.output).absolute()
  assert output_path.is_dir(), output_path
  dataset_path = Path(args.dataset).absolute() if args.dataset else (output_path / 'dataset')
  assert dataset_path.is_dir(), dataset_path
  
  # cfg
  for app_cfg in app_cfg_list:
    app_cfg_path = Path(app_cfg)
    #print(app_cfg_path)
    npx_define = NpxDefine(app_cfg_path=app_cfg_path, output_path=output_path)
    npx_data_manager = NpxDataManager(npx_define=npx_define, dataset_path=dataset_path, kfold=npx_define.kfold)
    if 'testvector' in cmd_list:
      generate_testvector(npx_define=npx_define, npx_data_manager=npx_data_manager, num_sample=num_sample, sample_only=args.sample_only)
    else:
      assert 0, cmd_list
