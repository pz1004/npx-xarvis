import os
import argparse
from pathlib import *
import numpy as np
import pickle
import struct

from npx_define import *
from npx_module import *
  
def analyze_best_result(npx_define:NpxDefine):
  line_list = []
  for report_path in npx_define.report_dir_path.glob(npx_define.get_report_filename_pattern()):
    line_list += report_path.read_text().split('\n')
    
  best_val_accuracy = 0
  best_result = None
  for line in line_list:
    if not line:
      pass
    single_result = RecordResult(*line.split('|'))
    val_accuracy = float(single_result.val_accuracy_str)
    if val_accuracy > best_val_accuracy:
      best_val_accuracy  = val_accuracy
      best_result = single_result
  return best_result

def copy_best_parameter(npx_define:NpxDefine, best_result:RecordResult):
  npx_define.riscv_dir_path.mkdir(parents=True, exist_ok=True)
  # best info
  with open(npx_define.get_riscv_info_path(True),"wb") as f:
    pickle.dump(best_result, f)
  npx_define.get_riscv_info_text_path(True).write_text(str(best_result))
  # cfg
  best_cfg = npx_define.get_parameter_epoch_cfg_path(int(best_result.epoch_index_str))
  assert best_cfg.is_file(), best_cfg
  cfg_parser = NpxCfgParser(best_cfg)
  cfg_parser.elaborate_for_riscv()
  riscv_cfg_path = npx_define.get_riscv_network_path()
  cfg_parser.write_file(riscv_cfg_path)
  # parameter
  best_parameter_path = npx_define.get_parameter_path(int(best_result.repeat_index_str),int(best_result.epoch_index_str),True)
  assert best_parameter_path.is_file(), best_parameter_path
  riscv_parameter_path = npx_define.get_riscv_parameter_path(True)
  riscv_parameter_path.write_bytes(best_parameter_path.read_bytes())
  
  best_parameter_text_path = npx_define.get_parameter_text_path(int(best_result.repeat_index_str),int(best_result.epoch_index_str),True)
  if best_parameter_text_path.is_file():
    riscv_parameter_text_path = npx_define.get_riscv_parameter_text_path(True)
    npx_define.riscv_dir_path.mkdir(parents=True, exist_ok=True)
    riscv_parameter_text_path.write_bytes(best_parameter_text_path.read_bytes())

def generate_riscv_binary(npx_define:NpxDefine):
  riscv_parameter_path = npx_define.get_riscv_parameter_path(True)
  assert riscv_parameter_path.is_file(), riscv_parameter_path

  npx_module = NpxModule(app_cfg_path=npx_define.app_cfg_path)
  npx_module.load_state_dict(torch.load(riscv_parameter_path, weights_only=False)['npx_module'])

  riscv_parameter_bin_path = npx_define.get_riscv_parameter_bin_path(True)
  #print(riscv_parameter_bin_path)

  if riscv_parameter_bin_path.is_file():
    riscv_parameter_bin_path.unlink()
  write_parameter_to_binaryfile(npx_module=npx_module, bin_path=riscv_parameter_bin_path)

def write_parameter_to_binaryfile(npx_module:NpxModule, bin_path:Path):
  print('save riscv parameter bin file:', bin_path)
  # print(npx_module.state_dict())
  with open(bin_path, "wb") as bin_file:
    for i, layer in enumerate(npx_module.layer_sequence):
      if (type(layer)==nn.Linear) or (type(layer)==nn.Conv2d):
        weights = layer.weight.data.flatten()
        neuron_type:NpxNeuronType = layer.neuron_type
        if neuron_type.num_bits <= 8:
          write_data_aligned_by_4bytes(bin_file, weights, torch.int8)
        elif neuron_type.num_bits <= 16:
          write_data_aligned_by_4bytes(bin_file, weights, torch.int16)
        elif neuron_type.num_bits <= 32:
          write_data_aligned_by_4bytes(bin_file, weights, torch.int32)
        else:
          assert 0, neuron_type.num_bits
      elif type(layer)==snntorch.Leaky:
        threshold = layer.threshold
        write_data_aligned_by_4bytes(bin_file, threshold, torch.int32)
        beta = layer.beta
        write_data_aligned_by_4bytes(bin_file, beta, torch.float32)

def write_data_aligned_by_4bytes(file_io, data:torch.Tensor, data_type:torch.dtype):
  if data_type==torch.float32:
    with torch.no_grad():
      #data = [struct.unpack('!I', struct.pack('!f', val))[0] for val in data.numpy().reshape(-1)]
      data = np.array([struct.unpack('!I', struct.pack('!f', val))[0] for val in data.numpy().reshape(-1)], dtype=np.uint32)
  else:
    data = data.to('cpu').to(data_type).numpy().reshape(-1)
    #data = data.to('cpu').round().to(data_type).numpy().reshape(-1)
    lenth = data.shape[0]
    fill_len = 0
    if (data_type == torch.int8) | (data_type == torch.uint8) :
      if (lenth%4) > 0 :
        fill_len = 4 - (lenth%4)
    elif (data_type == torch.int16) :
      if (lenth%2) > 0 :
        fill_len = 2 - (lenth%2)
    elif (data_type == torch.int32) :
      fill_len = 0
    else :
      print(f'unsupported type {data_type} in write_data_aligned_by_4bytes')
      return
    if fill_len > 0:
      fill_data = np.zeros(fill_len, dtype=data.dtype)
      data = np.append(data, fill_data)
  file_io.write(data)
  
if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='NPX Framework')
  parser.add_argument('-cfg', '-c', nargs='+', help='app cfg file name')
  parser.add_argument('-output', '-o', help='output directory')

  # check args
  args = parser.parse_args()
  assert args.cfg
  assert args.output

  app_cfg_list = args.cfg
  output_path = Path(args.output).absolute()
  assert output_path.is_dir(), output_path
  
  # cfg
  for app_cfg in app_cfg_list:
    app_cfg_path = Path(app_cfg)
    assert app_cfg_path.is_file(), app_cfg_path
    
    npx_define = NpxDefine(app_cfg_path=app_cfg_path, output_path=output_path)    
    best_result = analyze_best_result(npx_define)
    copy_best_parameter(npx_define,best_result)

    print()
    print(best_result)
    print()

    generate_riscv_binary(npx_define)
    
