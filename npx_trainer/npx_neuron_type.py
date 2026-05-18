from collections import namedtuple

from torch import Tensor

QTensor = namedtuple('QTensor', ['tensor', 'dqfactor', 'zero_point'])

class NpxNeuronType():
  def __init__(self, type_by_str:str=None):
    assert type_by_str!=None, type_by_str
    assert type_by_str[0]=='q', type_by_str
    
    self.is_quantized = True
    
    non_decimal_index = 1
    while 1:
      if not type_by_str[non_decimal_index].isdecimal():
        break
      non_decimal_index += 1
      
    bit_str = type_by_str[1:non_decimal_index]
    self.num_bits = int(bit_str)
    
    feature_str = type_by_str[non_decimal_index:]
    assert len(feature_str)==3, feature_str
    if feature_str[0]=='s':
      self.is_signed_weight = True
    elif feature_str[0]=='u':
      self.is_signed_weight = False
    else:
      assert 0
    if feature_str[1]=='s':
      self.is_signed_potential = True
    elif feature_str[1]=='u':
      self.is_signed_potential = False
    else:
      assert 0
    
    if feature_str[2]=='f':
      self.is_infinite_potential = False
      self.is_mapped_fvalue_updatable  = False
    elif feature_str[2]=='i':
      self.is_infinite_potential = True
      self.is_mapped_fvalue_updatable  = False
    elif feature_str[2]=='u':
      self.is_infinite_potential = True
      self.is_mapped_fvalue_updatable  = True
    else:
      assert 0
    
    if self.is_infinite_potential:
      self._mapped_fvalue = 0.5
    else:
      self._mapped_fvalue = 1.0
  
  @property
  def mapped_fvalue(self):
    return self._mapped_fvalue
  
  @mapped_fvalue.setter
  def mapped_fvalue(self, mapped_fvalue):
    if self.is_infinite_potential:
      self._mapped_fvalue = mapped_fvalue
    
  def update_mapped_fvalue(self, fvalue:float):
    if fvalue and self.is_infinite_potential:
      assert fvalue > 0, fvalue
      self._mapped_fvalue = fvalue
            
  def __repr__(self):
    result = (self.num_bits, self.is_signed_weight, self.is_signed_potential, self.is_infinite_potential)
    return str(result)
  
  @property
  def datatype_str(self):
    datatype = 'q' if self.is_quantized else 'c'
    return datatype + str(self.num_bits)
  
  @property
  def feature_str(self):
    result = ''
    result += 's' if self.is_signed_weight else 'u'
    result += 's' if self.is_signed_potential else 'u'
    if self.is_mapped_fvalue_updatable:
      result += 'u'
    elif self.is_infinite_potential:
      result += 'i'
    else:
      result += 'f'
    return result
  
  @property
  def name(self):
    return self.datatype_str + self.feature_str
          
  @property
  def umax(self):
    return int(2.**(self.num_bits)) - 1
  
  @property
  def umin(self):
    return 0
        
  @property
  def smax(self):
    return int(2.**(self.num_bits-1)) - 1
    
  @property
  def smin(self):
    return -self.smax
  
  @property
  def qscale(self):
    return self.smax if self.is_signed_weight else self.umax
  
  @property
  def qmax(self):
    return self.qscale

  @property
  def qmin(self):
    return self.smin if self.is_signed_weight else self.umin
  
  @property
  def qfactor(self):
    return float(self.qscale)/self._mapped_fvalue
  
  @property
  def qfactor(self):
    return float(self.qscale)/self._mapped_fvalue
  
  @property
  def dqfactor(self):
    return self._mapped_fvalue / self.qscale
  
  def can_learn_threshold(self):
    return True

  def can_learn_beta(self):
    return True
  
  def update_mapped_fvalue(self, x:Tensor):
    if self.is_mapped_fvalue_updatable:
      self._mapped_fvalue = float(x.abs().max())
  
  def synch_with_threshold(self, x:Tensor):
    if not self.is_infinite_potential:
      self._mapped_fvalue = float(x)

  def quantize_tensor(self, x:Tensor, bounded:bool):
    qx = x*self.qfactor
    if bounded:
      qx.clamp_(self.qmin, self.qmax)
    qx.round_()
    return QTensor(qx, self.dqfactor, 0)
  
  def clamp_weight_(self, x:Tensor, is_quantized:bool):
    if not self.is_signed_weight:
      x.clamp_(min=0)
  
  def clamp_mem_(self, x:Tensor, is_quantized:bool):
    if not self.is_signed_potential:
      x.clamp_(min=0)
      
  def quantize_beta(self, beta:float):
    denominator = 256
    beta_numerator = int(beta*denominator)
    result = float(beta_numerator) / denominator
    return result

  @staticmethod
  def dequantize_tensor(qx:QTensor):
    return qx.tensor.float()*qx.dqfactor
