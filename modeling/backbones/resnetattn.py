import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
# from torch.autograd import Function

# Resnets with CBAM: Convolutional Block Attention Module ability

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
  'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
  'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
  'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

'''
SUPPORT METHODS
'''

def conv3x3(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)

def remove_fc(state_dict):
  """Remove the fc layer parameters from state_dict."""
  no_fc_dict = state_dict.copy()
  for key, value in state_dict.items():
    if key.startswith('fc.'):
      del no_fc_dict[key]
  return no_fc_dict

class ChannelAttention(nn.Module):
  def __init__(self, in_planes, ratio=16):
    super(ChannelAttention, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.max_pool = nn.AdaptiveMaxPool2d(1)

    self.sharedMLP = nn.Sequential(
        nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
        nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    avgout = self.sharedMLP(self.avg_pool(x))
    maxout = self.sharedMLP(self.max_pool(x))
    return self.sigmoid(avgout + maxout)

class ChannelAttentionBn(nn.Module):
  def __init__(self, in_planes, ratio=16):
    super(ChannelAttentionBn, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.max_pool = nn.AdaptiveMaxPool2d(1)
    # self.bn_avg = nn.BatchNorm2d(in_planes)
    # self.bn_max = nn.BatchNorm2d(in_planes)
    self.bn = nn.BatchNorm2d(in_planes)

    self.sharedMLP = nn.Sequential(
        nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
        nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    avgout = self.sharedMLP(self.avg_pool(x))
    # avgout = self.bn_avg(avgout)
    maxout = self.sharedMLP(self.max_pool(x))
    # maxout = self.bn_max(maxout)
    sumout = self.bn(avgout + maxout)
    # return self.sigmoid(avgout + maxout)
    return self.sigmoid(sumout)

class SpatialAttention(nn.Module):
  def __init__(self, kernel_size=7):
    super(SpatialAttention, self).__init__()
    assert kernel_size in (3,7), "kernel size must be 3 or 7"
    padding = 3 if kernel_size == 7 else 1

    self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    avgout = torch.mean(x, dim=1, keepdim=True)
    maxout, _ = torch.max(x, dim=1, keepdim=True)
    x = torch.cat([avgout, maxout], dim=1)
    x = self.conv(x)
    return self.sigmoid(x)

class SpatialAttentionBn(nn.Module):
  def __init__(self, kernel_size=7):
    super(SpatialAttentionBn, self).__init__()
    assert kernel_size in (3,7), "kernel size must be 3 or 7"
    padding = 3 if kernel_size == 7 else 1

    self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
    self.bn = nn.BatchNorm2d(1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    avgout = torch.mean(x, dim=1, keepdim=True)
    maxout, _ = torch.max(x, dim=1, keepdim=True)
    x = torch.cat([avgout, maxout], dim=1)
    x = self.conv(x)
    x = self.bn(x)
    return self.sigmoid(x)

'''
END SUPPORT METHODS
'''

# class BasicBlock(nn.Module):
#   expansion = 1

#   def __init__(self, inplanes, planes, stride=1, downsample=None):
#     super(BasicBlock, self).__init__()
#     self.conv1 = conv3x3(inplanes, planes, stride)
#     self.bn1 = nn.BatchNorm2d(planes)
#     self.relu = nn.ReLU(inplace=True)
#     self.conv2 = conv3x3(planes, planes)
#     self.bn2 = nn.BatchNorm2d(planes)
#     self.downsample = downsample
#     self.stride = stride

#   def forward(self, x):
#     residual = x

#     out = self.conv1(x)
#     out = self.bn1(out)
#     out = self.relu(out)

#     out = self.conv2(out)
#     out = self.bn2(out)

#     if self.downsample is not None:
#       residual = self.downsample(x)

#     out += residual
#     out = self.relu(out)

#     return out


# class Bottleneck(nn.Module):
#   expansion = 4

#   def __init__(self, inplanes, planes, stride=1, downsample=None):
#     super(Bottleneck, self).__init__()
#     self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#     self.bn1 = nn.BatchNorm2d(planes)
#     self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                            padding=1, bias=False)
#     self.bn2 = nn.BatchNorm2d(planes)
#     self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#     self.bn3 = nn.BatchNorm2d(planes * 4)
#     self.relu = nn.ReLU(inplace=True)
#     self.downsample = downsample
#     self.stride = stride

#   def forward(self, x):
#     residual = x

#     out = self.conv1(x)
#     out = self.bn1(out)
#     out = self.relu(out)

#     out = self.conv2(out)
#     out = self.bn2(out)
#     out = self.relu(out)

#     out = self.conv3(out)
#     out = self.bn3(out)

#     if self.downsample is not None:
#       residual = self.downsample(x)

#     out += residual
#     out = self.relu(out)

#     return out

'''
VANILLA BLOCK DEFINITIONS
'''

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None, IN=False):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

    self.IN = None
    if IN:
      self.IN = nn.InstanceNorm2d(planes * self.expansion, affine=True)

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    if self.IN is not None:
      out = self.IN(out)
    out = self.relu(out)

    return out

class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None, IN=False):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                           padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion)

    self.IN = None
    if IN:
      self.IN = nn.InstanceNorm2d(planes * self.expansion, affine=True)

    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    if self.IN is not None:
      out = self.IN(out)
    out = self.relu(out)

    return out

'''
END VANILLA BLOCK DEFINITIONS
'''

'''
ATTENTION BLOCK DEFINITIONS
'''

class BasicBlockAttn(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None, IN=False):
    super(BasicBlockAttn, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.ca = ChannelAttention(planes)
    self.sa = SpatialAttention()
    self.downsample = downsample
    self.stride = stride

    self.IN = None
    if IN:
      self.IN = nn.InstanceNorm2d(planes * self.expansion, affine=True)

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.ca(out) * out
    out = self.sa(out) * out

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    if self.IN is not None:
      out = self.IN(out)
    out = self.relu(out)

    return out
  
class BottleneckAttn(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None, IN=False):
    super(BottleneckAttn, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                           padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion)

    self.ca = ChannelAttention(planes * self.expansion)
    self.sa = SpatialAttention()

    self.IN = None
    if IN:
      self.IN = nn.InstanceNorm2d(planes * self.expansion, affine=True)

    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    out = self.ca(out) * out
    out = self.sa(out) * out

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    if self.IN is not None:
      out = self.IN(out)
    out = self.relu(out)

    return out

# The only difference is in the additional BN in the attention layers
# I am hoping that will help training
class BottleneckAttnBn(BottleneckAttn):
  def __init__(self, inplanes, planes, stride=1, downsample=None, IN=False):
    super().__init__(inplanes, planes, stride, downsample, IN)

    self.ca = ChannelAttentionBn(planes * self.expansion)
    self.sa = SpatialAttentionBn()

'''
END ATTENTION BLOCK DEFINITIONS
'''

'''
TWIN ATTENTION BLOCK DEFINITIONS
'''

class BasicBlockAttnSingleInputTwinOutput(BasicBlockAttn):
  def forward(self, x):
    residual = x if self.downsample is None else self.downsample(x)

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    channel_attn = self.ca(out)
    spatial_attn = self.sa(out)

    # "filter" out the important channels
    channel_out = channel_attn * out

    # We will not use instance normalization
    id_out = spatial_attn * channel_out
    id_out += residual
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * channel_out
    cam_out += residual
    cam_out = self.relu(cam_out)

    return id_out, cam_out

class BottleneckAttnSingleInputTwinOutput(BottleneckAttn):
  def forward(self, x):
    residual = x if self.downsample is None else self.downsample(x)

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    channel_attn = self.ca(out)
    spatial_attn = self.sa(out)

    # "filter" out the important channels
    channel_out = channel_attn * out

    # We will not use instance normalization
    id_out = spatial_attn * channel_out
    id_out += residual
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * channel_out
    cam_out += residual
    cam_out = self.relu(cam_out)

    return id_out, cam_out

class BottleneckAttnSingleInputTwinOutput_v3(BottleneckAttn):
  def forward(self, x):
    residual = x if self.downsample is None else self.downsample(x)

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    channel_attn = self.ca(out)
    spatial_attn = self.sa(out)

    # "filter" out the important channels
    channel_out = channel_attn * out

    # We will not use instance normalization
    id_out = spatial_attn * channel_out
    id_out += residual
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * channel_out
    cam_out = self.relu(cam_out)

    return id_out, cam_out

# Same as above, but with BN in attention layers
class BottleneckAttnSingleInputTwinOutput_v4(BottleneckAttnBn):
  def forward(self, x):
    residual = x if self.downsample is None else self.downsample(x)

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    channel_attn = self.ca(out)
    spatial_attn = self.sa(out)

    # "filter" out the important channels
    channel_out = channel_attn * out

    # We will not use instance normalization
    id_out = spatial_attn * channel_out
    id_out += residual
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * channel_out
    # cam_out += residual
    cam_out = self.relu(cam_out)

    return id_out, cam_out

# Trying to use a separate downsample layer for the camera part
class BottleneckAttnSingleInputTwinOutput_v8(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, id_downsample=None, cam_downsample=None, IN=False):
    super().__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                           padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion)

    self.ca = ChannelAttention(planes * self.expansion)
    self.sa = SpatialAttention()

    self.IN = None
    if IN:
      self.IN = nn.InstanceNorm2d(planes * self.expansion, affine=True)

    self.relu = nn.ReLU(inplace=True)
    self.id_downsample = id_downsample
    self.cam_downsample = cam_downsample
    self.stride = stride

  def forward(self, x):
    id_res = x if self.id_downsample is None else self.id_downsample(x)
    cam_res = x if self.cam_downsample is None else self.cam_downsample(x)

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    channel_attn = self.ca(out)
    spatial_attn = self.sa(out)

    # "filter" out the important channels
    channel_out = channel_attn * out

    # We will not use instance normalization
    id_out = spatial_attn * channel_out
    id_out += id_res
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * channel_out
    cam_out += cam_res
    cam_out = self.relu(cam_out)

    return id_out, cam_out

class BasicBlockAttnTwinInputTwinOutput(BasicBlockAttn):
  def forward_path(self, x, is_camid):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    channel_attn = self.ca(out)
    spatial_attn = self.sa(out)
    if is_camid:
      spatial_attn = -spatial_attn + 1

    # "filter" out the important channels
    out = channel_attn * out
    out = spatial_attn * out
    
    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    if self.IN is not None:
      out = self.IN(out)
    out = self.relu(out)

    return out

  def forward(self, pid_in, camid_in):
    pid_out = self.forward_path(pid_in, False)
    camid_out = self.forward_path(camid_in, True)
    return pid_out, camid_out

class BottleneckAttnTwinInputTwinOutput(BottleneckAttn):
  def forward_path(self, x, is_camid):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    channel_attn = self.ca(out)
    spatial_attn = self.sa(out)
    if is_camid:
      spatial_attn = -spatial_attn + 1

    # "filter" out the important channels
    out = channel_attn * out
    out = spatial_attn * out

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    if self.IN is not None:
      out = self.IN(out)
    out = self.relu(out)

    return out

  def forward(self, x):
    pid_in, camid_in = x
    pid_out = self.forward_path(pid_in, False)
    camid_out = self.forward_path(camid_in, True)
    return pid_out, camid_out

'''
 In this version, we try to:
 - Let one tensor take the lead
 - The CBAM will separate as usual, but the camera based tensor will only be
   passed on and downsampled and added residually to the other camera tensors
   downstream. Each camera tensor is the result of a separation of the original
'''
class BottleneckAttnTwinInputTwinOutput_v2(BottleneckAttn):
  def forward_idpath(self, x):
    residual = x if self.downsample is None else self.downsample(x)

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    channel_attn = self.ca(out)
    spatial_attn = self.sa(out)

    # "filter" out the important channels
    channel_out = channel_attn * out

    # We will not use instance normalization
    id_out = spatial_attn * channel_out
    id_out += residual
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * channel_out
    cam_out += residual
    cam_out = self.relu(cam_out)

    return id_out, cam_out

  def forward(self, x):
    pid_in, camid_in = x
    # Only the ID branch gets "cleansed". camid_in is just added residually.
    pid_out, camid_out = self.forward_idpath(pid_in)
    cam_res = camid_in if self.downsample is None else self.downsample(camid_in)
    camid_out = cam_res + camid_out
    return pid_out, camid_out

class BottleneckAttnTwinInputTwinOutput_v3(BottleneckAttn):
  def forward_idpath(self, x):
    residual = x if self.downsample is None else self.downsample(x)

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    channel_attn = self.ca(out)
    spatial_attn = self.sa(out)

    # "filter" out the important channels
    channel_out = channel_attn * out

    id_out = spatial_attn * channel_out
    id_out += residual
    id_out = self.relu(id_out)

    # Notice here that we only multiply by the spatial attention, but do not add the
    # residue - that belongs to the ID-path. The residue for cam-path is outside this
    # function
    cam_out = (-spatial_attn + 1) * channel_out

    return id_out, cam_out

  def forward(self, x):
    pid_in, camid_in = x
    # Only the ID branch gets "cleansed". camid_in is just added residually.
    pid_out, camid_out = self.forward_idpath(pid_in)
    cam_res = camid_in if self.downsample is None else self.downsample(camid_in)
    camid_out = self.relu(cam_res + camid_out)
    return pid_out, camid_out

# Same as above bottleneck, but using BN in attention layers
class BottleneckAttnTwinInputTwinOutput_v4(BottleneckAttnBn):
  def forward_idpath(self, x):
    residual = x if self.downsample is None else self.downsample(x)

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    channel_attn = self.ca(out)
    spatial_attn = self.sa(out)

    # "filter" out the important channels
    channel_out = channel_attn * out

    id_out = spatial_attn * channel_out
    id_out += residual
    id_out = self.relu(id_out)

    # Notice here that we only multiply by the spatial attention, but do not add the
    # residue - that belongs to the ID-path. The residue for cam-path is outside this
    # function
    cam_out = (-spatial_attn + 1) * channel_out

    return id_out, cam_out

  def forward(self, x):
    pid_in, camid_in = x
    # Only the ID branch gets "cleansed". camid_in is just added residually.
    pid_out, camid_out = self.forward_idpath(pid_in)
    cam_res = camid_in if self.downsample is None else self.downsample(camid_in)
    camid_out = self.relu(cam_res + camid_out)
    return pid_out, camid_out

# Notice how the camid part is handled in this one
class BottleneckAttnTwinInputTwinOutput_v5(BottleneckAttn):
  def forward_idpath(self, x):
    residual = x if self.downsample is None else self.downsample(x)

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    channel_attn = self.ca(out)
    spatial_attn = self.sa(out)

    # "filter" out the important channels
    channel_out = channel_attn * out

    id_out = spatial_attn * channel_out
    id_out += residual
    id_out = self.relu(id_out)

    # Unlike v3/4, add the id-residue, but add cam-residue outside and take relu
    cam_out = (-spatial_attn + 1) * channel_out
    cam_out += residual

    return id_out, cam_out

  def forward(self, x):
    pid_in, camid_in = x
    # Only the ID branch gets "cleansed". camid_in is just added residually.
    pid_out, camid_out = self.forward_idpath(pid_in)
    cam_res = camid_in if self.downsample is None else self.downsample(camid_in)
    camid_out = self.relu(cam_res + camid_out)
    return pid_out, camid_out

class BottleneckAttnTwinInputTwinOutput_v8(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, id_downsample=None, cam_downsample=None, IN=False):
    super().__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                           padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion)

    self.ca = ChannelAttention(planes * self.expansion)
    self.sa = SpatialAttention()

    self.IN = None
    if IN:
      self.IN = nn.InstanceNorm2d(planes * self.expansion, affine=True)

    self.relu = nn.ReLU(inplace=True)
    self.id_downsample = id_downsample
    self.cam_downsample = cam_downsample
    self.stride = stride

  def forward_idpath(self, x):
    residual = x if self.id_downsample is None else self.id_downsample(x)

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    channel_attn = self.ca(out)
    spatial_attn = self.sa(out)

    # "filter" out the important channels
    channel_out = channel_attn * out

    id_out = spatial_attn * channel_out
    id_out += residual
    id_out = self.relu(id_out)

    # Notice here that we only multiply by the spatial attention, but do not add the
    # residue - that belongs to the ID-path. The residue for cam-path is outside this
    # function
    cam_out = (-spatial_attn + 1) * channel_out

    return id_out, cam_out

  def forward(self, x):
    pid_in, camid_in = x
    # Only the ID branch gets "cleansed". camid_in is just added residually.
    pid_out, camid_out = self.forward_idpath(pid_in)
    cam_res = camid_in if self.cam_downsample is None else self.cam_downsample(camid_in)
    camid_out = self.relu(cam_res + camid_out)
    return pid_out, camid_out

'''
END TWIN ATTENTION BLOCK DEFINITIONS
'''

'''
MODEL DEFINITIONS
'''

# Vanilla ResNet definition. Use attention blocks to make plain CBAM models.
class ResNet(nn.Module):

  def __init__(self, block, layers, last_stride=2, IN=False):

    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0], IN=IN)
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2, IN=IN)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2, IN=IN)
    self.layer4 = self._make_layer(
      block, 512, layers[3], stride=last_stride)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, IN=False):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, IN=IN))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x

# ResNet50 with multi-task, but no CBAM or rev-grad.
# For ablation study.
class ResNetMultiTask(ResNet):
  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    cam_out = self.layer1(x)
    x = self.layer2(cam_out)
    x = self.layer3(x)
    id_out = self.layer4(x)

    return id_out, cam_out

# Because it only applies the CBAM at the final stage, not within each Res-Block
class ResNetCbamLast(nn.Module):

  def __init__(self, block, layers, last_stride=2, IN=False):

    self.inplanes = 64
    super(ResNetCbamLast, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0], IN=IN)
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2, IN=IN)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2, IN=IN)
    self.layer4 = self._make_layer(
      block, 512, layers[3], stride=last_stride)

    # Add CBAM after the last layer
    self.ca = ChannelAttention(self.inplanes)
    # Use a smaller kernel_size, since we're already looking at a large FOV
    self.sa = SpatialAttention(kernel_size=3)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, IN=False):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, IN=IN))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    residual = x
    x = self.ca(x) * x
    x = self.sa(x) * x
    x += residual
    x = self.relu(x)

    return x

  def forward_spatial_attn(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    spatial_attn = self.sa(x)
    return [ spatial_attn ]

# Because it only applies the CBAM at the final stage, not within each Res-Block
# This version "splits up" the spatial attention - PID gets the original spatial
# attention, CAMID gets (1-c), the complement.
class ResNetCbamLastTwin(ResNetCbamLast):
  # Use the parent's __init__() function. Only forward() needs to be modified.

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    residual = x
    channel_attn = self.ca(x)
    spatial_attn = self.sa(x)

    # "filter" out the important channels
    channel_out = channel_attn * x

    id_out = spatial_attn * channel_out
    id_out += residual
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * channel_out
    cam_out += residual
    cam_out = self.relu(cam_out)

    return id_out, cam_out, residual

# This version is similar to the above, but the triplet loss gets it's own
# attention-modified tensor, rather than having to use a communal one.
class ResNetCbamLastTwinSeparateTripletAttn(ResNetCbamLast):
  # Use the parent's __init__() function. Only forward() needs to be modified.
  def __init__(self, block, layers, last_stride=2, IN=False):
    super().__init__(block, layers, last_stride, IN)
    # Triplet consumer has it's own channel and spatial attention
    self.triplet_ca = ChannelAttention(self.inplanes)
    self.triplet_sa = SpatialAttention(kernel_size=3)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    residual = x

    channel_attn = self.ca(x)
    spatial_attn = self.sa(x)
    triplet_channel_attn = self.triplet_ca(x)
    triplet_spatial_attn = self.triplet_sa(x)

    # "filter" out the important channels
    channel_out = channel_attn * x
    triplet_channel_out = triplet_channel_attn * x

    id_out = spatial_attn * channel_out
    id_out += residual
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * channel_out
    cam_out += residual
    cam_out = self.relu(cam_out)

    triplet_out = triplet_spatial_attn * triplet_channel_out
    triplet_out += residual
    triplet_out = self.relu(triplet_out)

    return id_out, cam_out, triplet_out

# Because it only applies the CBAM at the final stage, not within each Res-Block
# This version "splits up" the spatial attention - PID gets the original spatial
# attention, CAMID gets (1-c), the complement.
class ResNetCbamLastChannelSpatialTwin(ResNetCbamLast):
  # Use the parent's __init__() function. Only forward() needs to be modified.

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    residual = x
    channel_attn = self.ca(x)
    spatial_attn = self.sa(x)

    # "filter" out the important channels
    id_channel_out = channel_attn * x
    cam_channel_out = (-channel_attn + 1) * x

    id_out = spatial_attn * id_channel_out
    id_out += residual
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * cam_channel_out
    cam_out += residual
    cam_out = self.relu(cam_out)

    return id_out, cam_out, residual

# Applying only the spatial attention to the tensor
class ResNetCbamLastTwinSpatialOnly(ResNetCbamLast):
  # Use the parent's __init__() function. Only forward() needs to be modified.

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    residual = x
    spatial_attn = self.sa(x)

    # "filter" out the important channels
    id_out = spatial_attn * x
    id_out += residual
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * x
    cam_out += residual
    cam_out = self.relu(cam_out)

    return id_out, cam_out, residual

# Applying only the channel attention to the tensor
class ResNetCbamLastTwinChannelOnly(ResNetCbamLast):
  # Use the parent's __init__() function. Only forward() needs to be modified.

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    residual = x
    channel_attn = self.ca(x)

    # "filter" out the important channels
    id_out = channel_attn * x
    id_out += residual
    id_out = self.relu(id_out)

    cam_out = (-channel_attn + 1) * x
    cam_out += residual
    cam_out = self.relu(cam_out)

    return id_out, cam_out, residual

# class GradReverse(Function):
#     @staticmethod
#     def forward(ctx, x):
#         return x.view_as(x)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output.neg()

# # Applies the reverse-gradient to the output feature map of the resnet, 
# # before applying channel-wise and spatial attention
# class ResNetCbamEarlyAdvLastTwin(ResNetCbamLast):
#   # Use the parent's __init__() function. Only forward() needs to be modified.
#   revgrad = GradReverse.apply

#   def forward(self, x):
#     x = self.conv1(x)
#     x = self.bn1(x)
#     x = self.relu(x)
#     x = self.maxpool(x)

#     x = self.layer1(x)
#     x = self.layer2(x)
#     x = self.layer3(x)
#     x = self.layer4(x)

#     residual = x
    
#     channel_attn = self.ca(x)
#     spatial_attn = self.sa(x)

#     # "filter" out the important channels
#     channel_out = channel_attn * x

#     id_out = spatial_attn * channel_out
#     id_out += residual
#     id_out = self.relu(id_out)

#     cam_out = (-spatial_attn + 1) * channel_out
#     cam_out += residual
#     cam_out = self.relu(cam_out)

#     return id_out, cam_out

# I reduced the amount of CBAM-ing
class ResNetCbamTwin(nn.Module):

  def __init__(self, block, single_inp_block, twin_inp_block, layers, last_stride=2, IN=False):

    self.inplanes = 64
    super(ResNetCbamTwin, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    
    self.single_inp_block = single_inp_block
    self.twin_inp_block = twin_inp_block

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, block, 64, layers[0], IN=IN)
    self.layer2 = self._make_layer(block, block, 128, layers[1], stride=2, IN=IN)
    self.layer3 = self._make_layer(single_inp_block, twin_inp_block, 256, layers[2], stride=2, IN=IN)
    self.layer4 = self._make_layer(twin_inp_block, twin_inp_block, 512, layers[3], stride=last_stride)
    # Can also make layer4 the first _make_layer_single_inp

    # self.layer1 = self._make_layer(block, 64, layers[0])
    # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    # self.layer3 = self._make_layer_single_inp(256, layers[2], stride=2)
    # self.layer4 = self._make_layer_twin_inp(512, layers[3], stride=last_stride)
    # Can also make layer4 the first _make_layer_single_inp
    

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, headblock, afterblock, planes, blocks, stride=1, IN=False):
    downsample = None
    if stride != 1 or self.inplanes != planes * headblock.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * headblock.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * headblock.expansion),
      )

    layers = []
    layers.append(headblock(self.inplanes, planes, stride, downsample, IN=IN))
    self.inplanes = planes * headblock.expansion
    for _ in range(1, blocks):
      layers.append(afterblock(self.inplanes, planes))

    return nn.Sequential(*layers)

  # def _make_layer(self, block, planes, blocks, stride=1, IN=False):
  #   downsample = None
  #   if stride != 1 or self.inplanes != planes * block.expansion:
  #     downsample = nn.Sequential(
  #       nn.Conv2d(self.inplanes, planes * block.expansion,
  #                 kernel_size=1, stride=stride, bias=False),
  #       nn.BatchNorm2d(planes * block.expansion),
  #     )

  #   layers = []
  #   layers.append(block(self.inplanes, planes, stride, downsample, IN=IN))
  #   self.inplanes = planes * block.expansion
  #   for i in range(1, blocks):
  #     layers.append(block(self.inplanes, planes))

  #   return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x

class ResNetCbamTwin_v2(nn.Module):

  def __init__(self, layers, last_stride=2, IN=False):

    self.inplanes = 64
    super(ResNetCbamTwin_v2, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    
    self.block = Bottleneck
    self.single_inp_block = BottleneckAttnSingleInputTwinOutput
    self.twin_inp_block = BottleneckAttnTwinInputTwinOutput_v2

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(self.block, self.block, 64, layers[0], IN=IN)
    self.layer2 = self._make_layer(self.single_inp_block, self.twin_inp_block, 128, layers[1], stride=2, IN=IN)
    self.layer3 = self._make_layer(self.twin_inp_block, self.twin_inp_block, 256, layers[2], stride=2, IN=IN)
    self.layer4 = self._make_layer(self.twin_inp_block, self.twin_inp_block, 512, layers[3], stride=last_stride)
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, headblock, afterblock, planes, blocks, stride=1, IN=False):
    downsample = None
    if stride != 1 or self.inplanes != planes * headblock.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * headblock.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * headblock.expansion),
      )

    layers = []
    layers.append(headblock(self.inplanes, planes, stride, downsample, IN=IN))
    self.inplanes = planes * headblock.expansion
    for _ in range(1, blocks):
      layers.append(afterblock(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x

class ResNetCbamTwin_v3(nn.Module):

  def __init__(self, layers, last_stride=2, IN=False):

    self.inplanes = 64
    super(ResNetCbamTwin_v3, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    
    self.block = Bottleneck
    self.single_inp_block = BottleneckAttnSingleInputTwinOutput
    self.twin_inp_block = BottleneckAttnTwinInputTwinOutput_v3

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(self.block, self.block, 64, layers[0], IN=IN)
    self.layer2 = self._make_layer(self.single_inp_block, self.twin_inp_block, 128, layers[1], stride=2, IN=IN)
    self.layer3 = self._make_layer(self.twin_inp_block, self.twin_inp_block, 256, layers[2], stride=2, IN=IN)
    self.layer4 = self._make_layer(self.twin_inp_block, self.twin_inp_block, 512, layers[3], stride=last_stride)
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, headblock, afterblock, planes, blocks, stride=1, IN=False):
    downsample = None
    if stride != 1 or self.inplanes != planes * headblock.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * headblock.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * headblock.expansion),
      )

    layers = []
    layers.append(headblock(self.inplanes, planes, stride, downsample, IN=IN))
    self.inplanes = planes * headblock.expansion
    for _ in range(1, blocks):
      layers.append(afterblock(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x

# v8 is exactly the same as v3, except there are two specialized downsampling layers, one for id and one for cam.
# I think having just one downsampling layer maybe affecting the power of the model
class ResNetCbamTwin_v8(nn.Module):

  def __init__(self, layers, last_stride=2, IN=False):

    self.inplanes = 64
    super(ResNetCbamTwin_v8, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    
    self.block = Bottleneck
    self.single_inp_block = BottleneckAttnSingleInputTwinOutput_v8
    self.twin_inp_block = BottleneckAttnTwinInputTwinOutput_v8

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_basic_layer(self.block, 64, layers[0], IN=IN)
    self.layer2 = self._make_layer(self.single_inp_block, self.twin_inp_block, 128, layers[1], stride=2, IN=IN)
    self.layer3 = self._make_layer(self.twin_inp_block, self.twin_inp_block, 256, layers[2], stride=2, IN=IN)
    self.layer4 = self._make_layer(self.twin_inp_block, self.twin_inp_block, 512, layers[3], stride=last_stride)
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_basic_layer(self, block, planes, blocks, stride=1, IN=False):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, IN=IN))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def _make_layer(self, headblock, afterblock, planes, blocks, stride=1, IN=False):
    id_ds = None
    cam_ds = None
    if stride != 1 or self.inplanes != planes * headblock.expansion:
      id_ds = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * headblock.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * headblock.expansion),
      )
      cam_ds = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * headblock.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * headblock.expansion),
      )

    layers = []
    layers.append(headblock(self.inplanes, planes, stride, id_downsample=id_ds, cam_downsample=cam_ds, IN=IN))
    self.inplanes = planes * headblock.expansion
    for _ in range(1, blocks):
      layers.append(afterblock(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x

# When two ds layers are used in a v3-type model, the performance seems to suffer.

class ResNetCbamTwin_v4(nn.Module):

  def __init__(self, layers, last_stride=2, IN=False):

    self.inplanes = 64
    super(ResNetCbamTwin_v4, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    
    self.block = Bottleneck
    self.single_inp_block = BottleneckAttnSingleInputTwinOutput_v4
    self.twin_inp_block = BottleneckAttnTwinInputTwinOutput_v4

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(self.block, self.block, 64, layers[0], IN=IN)
    self.layer2 = self._make_layer(self.single_inp_block, self.twin_inp_block, 128, layers[1], stride=2, IN=IN)
    self.layer3 = self._make_layer(self.twin_inp_block, self.twin_inp_block, 256, layers[2], stride=2, IN=IN)
    self.layer4 = self._make_layer(self.twin_inp_block, self.twin_inp_block, 512, layers[3], stride=last_stride)
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, headblock, afterblock, planes, blocks, stride=1, IN=False):
    downsample = None
    if stride != 1 or self.inplanes != planes * headblock.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * headblock.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * headblock.expansion),
      )

    layers = []
    layers.append(headblock(self.inplanes, planes, stride, downsample, IN=IN))
    self.inplanes = planes * headblock.expansion
    for _ in range(1, blocks):
      layers.append(afterblock(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x

class ResNetCbamTwin_v5(nn.Module):

  def __init__(self, layers, last_stride=1, IN=False):

    self.inplanes = 64
    super(ResNetCbamTwin_v5, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    
    self.block = Bottleneck
    self.single_inp_block = BottleneckAttnSingleInputTwinOutput
    self.twin_inp_block = BottleneckAttnTwinInputTwinOutput_v5

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(self.block, self.block, 64, layers[0], IN=IN)
    self.layer2 = self._make_layer(self.single_inp_block, self.twin_inp_block, 128, layers[1], stride=2, IN=IN)
    self.layer3 = self._make_layer(self.twin_inp_block, self.twin_inp_block, 256, layers[2], stride=2, IN=IN)
    self.layer4 = self._make_layer(self.twin_inp_block, self.twin_inp_block, 512, layers[3], stride=last_stride)
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, headblock, afterblock, planes, blocks, stride=1, IN=False):
    downsample = None
    if stride != 1 or self.inplanes != planes * headblock.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * headblock.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * headblock.expansion),
      )

    layers = []
    layers.append(headblock(self.inplanes, planes, stride, downsample, IN=IN))
    self.inplanes = planes * headblock.expansion
    for _ in range(1, blocks):
      layers.append(afterblock(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x

# 1. Hardcore first twin layer
# 2. Adds the residue to the camid part at the end rather than at the start
class ResNetCbamTwin_v6(nn.Module):

  def __init__(self, layers, last_stride=2, IN=False):

    self.inplanes = 64
    super(ResNetCbamTwin_v6, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    
    self.block = Bottleneck
    self.single_inp_block = BottleneckAttnSingleInputTwinOutput_v3
    self.twin_inp_block = BottleneckAttnTwinInputTwinOutput_v3

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(self.block, self.block, 64, layers[0], IN=IN)
    self.layer2 = self._make_layer(self.single_inp_block, self.twin_inp_block, 128, layers[1], stride=2, IN=IN)
    self.layer3 = self._make_layer(self.twin_inp_block, self.twin_inp_block, 256, layers[2], stride=2, IN=IN)
    self.layer4 = self._make_layer(self.twin_inp_block, self.twin_inp_block, 512, layers[3], stride=last_stride)

    # Add CBAM after the last layer
    self.ca = ChannelAttention(self.inplanes)
    # Use a smaller kernel_size, since we're already looking at a large FOV
    self.sa = SpatialAttention(kernel_size=3)
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, headblock, afterblock, planes, blocks, stride=1, IN=False):
    downsample = None
    if stride != 1 or self.inplanes != planes * headblock.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * headblock.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * headblock.expansion),
      )

    layers = []
    layers.append(headblock(self.inplanes, planes, stride, downsample, IN=IN))
    self.inplanes = planes * headblock.expansion
    for _ in range(1, blocks):
      layers.append(afterblock(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    id_in, cam_in = x
    channel_attn = self.ca(id_in)
    spatial_attn = self.sa(id_in)

    # "filter" out the important channels
    channel_out = channel_attn * id_in

    id_out = spatial_attn * channel_out
    id_out += id_in
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * channel_out
    cam_out += cam_in
    cam_out = self.relu(cam_out)

    return id_out, cam_out

# Shift the CBAM to between conv_blocks, rather than within the conv_block
class ResNetCbamTwin_v7(nn.Module):

  def __init__(self, layers, last_stride=2, IN=False):

    self.inplanes = 64
    super(ResNetCbamTwin_v7, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    
    self.block = Bottleneck

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1, _ = self._make_layer(self.block, self.block, 64, layers[0], IN=IN)
    self.ca12 = ChannelAttention(self.inplanes)
    self.sa12 = SpatialAttention(kernel_size=3)
    self.layer2, self.ds2 = self._make_layer(self.block, self.block, 128, layers[1], stride=2, IN=IN)
    self.ca23 = ChannelAttention(self.inplanes)
    self.sa23 = SpatialAttention(kernel_size=3)
    self.layer3, self.ds3 = self._make_layer(self.block, self.block, 256, layers[2], stride=2, IN=IN)
    self.ca34 = ChannelAttention(self.inplanes)
    self.sa34 = SpatialAttention(kernel_size=3)
    self.layer4, self.ds4 = self._make_layer(self.block, self.block, 512, layers[3], stride=last_stride)
    self.ca4e = ChannelAttention(self.inplanes)
    self.sa4e = SpatialAttention(kernel_size=3)
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, headblock, afterblock, planes, blocks, stride=1, IN=False):
    downsample = None
    ds = None
    if stride != 1 or self.inplanes != planes * headblock.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * headblock.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * headblock.expansion),
      )
      ds = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * headblock.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * headblock.expansion),
      )

    layers = []
    layers.append(headblock(self.inplanes, planes, stride, downsample, IN=IN))
    self.inplanes = planes * headblock.expansion
    for _ in range(1, blocks):
      layers.append(afterblock(self.inplanes, planes))

    # return nn.Sequential(*layers), downsample
    return nn.Sequential(*layers), ds

  def forward_attn_1IN_2OUT(self, ca, sa, x):
    channel_attn = ca(x)
    spatial_attn = sa(x)

    # "filter" out the important channels
    channel_out = channel_attn * x

    id_out = spatial_attn * channel_out
    id_out += x
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * channel_out
    cam_out += x
    cam_out = self.relu(cam_out)
    return id_out, cam_out

  def forward_attn_2IN_2OUT(self, ca, sa, id_in, cam_in):
    channel_attn = ca(id_in)
    spatial_attn = sa(id_in)

    # "filter" out the important channels
    channel_out = channel_attn * id_in

    id_out = spatial_attn * channel_out
    id_out += id_in
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * channel_out
    cam_out += cam_in
    cam_out = self.relu(cam_out)
    return id_out, cam_out

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    id12, cam12 = self.forward_attn_1IN_2OUT(self.ca12, self.sa12, x)
    
    id12 = self.layer2(id12)
    
    if self.ds2:
      cam12 = self.ds2(cam12)
    
    id23, cam23 = self.forward_attn_2IN_2OUT(self.ca23, self.sa23, id12, cam12)
    
    id23 = self.layer3(id23)

    if self.ds3:
      cam23 = self.ds3(cam23)

    id34, cam34 = self.forward_attn_2IN_2OUT(self.ca34, self.sa34, id23, cam23)

    id34 = self.layer4(id34)

    if self.ds4:
      cam34 = self.ds4(cam34)

    id_out, cam_out = self.forward_attn_2IN_2OUT(self.ca4e, self.sa4e, id34, cam34)

    return id_out, cam_out
  
# CBAM between last three conv_blocks
class ResNetCbamTwin_v9(nn.Module):

  def __init__(self, layers, last_stride=1, IN=False):

    self.inplanes = 64
    super().__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    
    self.block = Bottleneck

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1, _ = self._make_layer(self.block, self.block, 64, layers[0], IN=IN)
    self.layer2, _ = self._make_layer(self.block, self.block, 128, layers[1], stride=2, IN=IN)
    self.ca23 = ChannelAttention(self.inplanes)
    self.sa23 = SpatialAttention(kernel_size=3)
    self.layer3, self.ds3 = self._make_layer(self.block, self.block, 256, layers[2], stride=2, IN=IN)
    self.ca34 = ChannelAttention(self.inplanes)
    self.sa34 = SpatialAttention(kernel_size=3)
    self.layer4, self.ds4 = self._make_layer(self.block, self.block, 512, layers[3], stride=last_stride)
    self.ca4e = ChannelAttention(self.inplanes)
    self.sa4e = SpatialAttention(kernel_size=3)
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, headblock, afterblock, planes, blocks, stride=1, IN=False):
    downsample = None
    ds = None
    if stride != 1 or self.inplanes != planes * headblock.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * headblock.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * headblock.expansion),
      )
      ds = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * headblock.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * headblock.expansion),
      )

    layers = []
    layers.append(headblock(self.inplanes, planes, stride, downsample, IN=IN))
    self.inplanes = planes * headblock.expansion
    for _ in range(1, blocks):
      layers.append(afterblock(self.inplanes, planes))

    return nn.Sequential(*layers), ds

  def forward_attn_1IN_2OUT(self, ca, sa, x):
    channel_attn = ca(x)
    spatial_attn = sa(x)

    # "filter" out the important channels
    channel_out = channel_attn * x

    id_out = spatial_attn * channel_out
    id_out += x
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * channel_out
    cam_out += x
    cam_out = self.relu(cam_out)
    return id_out, cam_out

  def forward_attn_2IN_2OUT(self, ca, sa, id_in, cam_in):
    channel_attn = ca(id_in)
    spatial_attn = sa(id_in)

    # "filter" out the important channels
    channel_out = channel_attn * id_in

    id_out = spatial_attn * channel_out
    id_out += id_in
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * channel_out
    cam_out += cam_in
    cam_out = self.relu(cam_out)
    return id_out, cam_out

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    id23, cam23 = self.forward_attn_1IN_2OUT(self.ca23, self.sa23, x)    
    
    id23 = self.layer3(id23)

    if self.ds3:
      cam23 = self.ds3(cam23)

    id34, cam34 = self.forward_attn_2IN_2OUT(self.ca34, self.sa34, id23, cam23)

    id34 = self.layer4(id34)

    if self.ds4:
      cam34 = self.ds4(cam34)

    id_out, cam_out = self.forward_attn_2IN_2OUT(self.ca4e, self.sa4e, id34, cam34)

    return id_out, cam_out

# Just CBAM between the last two conv_blocks
class ResNetCbamTwin_v10(nn.Module):

  def __init__(self, layers, last_stride=1, IN=False):

    self.inplanes = 64
    super().__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    
    self.block = Bottleneck

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1, _ = self._make_layer(self.block, self.block, 64, layers[0], IN=IN)
    self.layer2, _ = self._make_layer(self.block, self.block, 128, layers[1], stride=2, IN=IN)
    self.layer3, _ = self._make_layer(self.block, self.block, 256, layers[2], stride=2, IN=IN)
    self.ca34 = ChannelAttention(self.inplanes)
    self.sa34 = SpatialAttention(kernel_size=3)
    self.layer4, self.ds4 = self._make_layer(self.block, self.block, 512, layers[3], stride=last_stride)
    self.ca4e = ChannelAttention(self.inplanes)
    self.sa4e = SpatialAttention(kernel_size=3)
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, headblock, afterblock, planes, blocks, stride=1, IN=False):
    downsample = None
    ds = None
    if stride != 1 or self.inplanes != planes * headblock.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * headblock.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * headblock.expansion),
      )
      ds = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * headblock.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * headblock.expansion),
      )

    layers = []
    layers.append(headblock(self.inplanes, planes, stride, downsample, IN=IN))
    self.inplanes = planes * headblock.expansion
    for _ in range(1, blocks):
      layers.append(afterblock(self.inplanes, planes))

    return nn.Sequential(*layers), ds

  def forward_attn_1IN_2OUT(self, ca, sa, x):
    channel_attn = ca(x)
    spatial_attn = sa(x)

    # "filter" out the important channels
    channel_out = channel_attn * x

    id_out = spatial_attn * channel_out
    id_out += x
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * channel_out
    cam_out += x
    cam_out = self.relu(cam_out)
    return id_out, cam_out

  def forward_attn_2IN_2OUT(self, ca, sa, id_in, cam_in):
    channel_attn = ca(id_in)
    spatial_attn = sa(id_in)

    # "filter" out the important channels
    channel_out = channel_attn * id_in

    id_out = spatial_attn * channel_out
    id_out += id_in
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * channel_out
    cam_out += cam_in
    cam_out = self.relu(cam_out)
    return id_out, cam_out

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    
    id34, cam34 = self.forward_attn_1IN_2OUT(self.ca34, self.sa34, x)    

    id34 = self.layer4(id34)

    if self.ds4:
      cam34 = self.ds4(cam34)

    id_out, cam_out = self.forward_attn_2IN_2OUT(self.ca4e, self.sa4e, id34, cam34)

    return id_out, cam_out

# Same as v7, but with only one shared downsampling layer per conv_block
# Trying because observed degraded performance in v8 when dedicated downsampling layers for id/cam.
class ResNetCbamTwin_v11(nn.Module):

  def __init__(self, layers, last_stride=2, IN=False):

    self.inplanes = 64
    super(ResNetCbamTwin_v11, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    
    self.block = Bottleneck

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1, _ = self._make_layer(self.block, self.block, 64, layers[0], IN=IN)
    self.ca12 = ChannelAttention(self.inplanes)
    self.sa12 = SpatialAttention(kernel_size=3)
    self.layer2, self.ds2 = self._make_layer(self.block, self.block, 128, layers[1], stride=2, IN=IN)
    self.ca23 = ChannelAttention(self.inplanes)
    self.sa23 = SpatialAttention(kernel_size=3)
    self.layer3, self.ds3 = self._make_layer(self.block, self.block, 256, layers[2], stride=2, IN=IN)
    self.ca34 = ChannelAttention(self.inplanes)
    self.sa34 = SpatialAttention(kernel_size=3)
    self.layer4, self.ds4 = self._make_layer(self.block, self.block, 512, layers[3], stride=last_stride)
    self.ca4e = ChannelAttention(self.inplanes)
    self.sa4e = SpatialAttention(kernel_size=3)
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, headblock, afterblock, planes, blocks, stride=1, IN=False):
    downsample = None
    if stride != 1 or self.inplanes != planes * headblock.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * headblock.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * headblock.expansion),
      )

    layers = []
    layers.append(headblock(self.inplanes, planes, stride, downsample, IN=IN))
    self.inplanes = planes * headblock.expansion
    for _ in range(1, blocks):
      layers.append(afterblock(self.inplanes, planes))

    return nn.Sequential(*layers), downsample

  def forward_attn_1IN_2OUT(self, ca, sa, x):
    channel_attn = ca(x)
    spatial_attn = sa(x)

    # "filter" out the important channels
    channel_out = channel_attn * x

    id_out = spatial_attn * channel_out
    id_out += x
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * channel_out
    cam_out += x
    cam_out = self.relu(cam_out)
    return id_out, cam_out

  def forward_attn_2IN_2OUT(self, ca, sa, id_in, cam_in):
    channel_attn = ca(id_in)
    spatial_attn = sa(id_in)

    # "filter" out the important channels
    channel_out = channel_attn * id_in

    id_out = spatial_attn * channel_out
    id_out += id_in
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * channel_out
    cam_out += cam_in
    cam_out = self.relu(cam_out)
    return id_out, cam_out

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    id12, cam12 = self.forward_attn_1IN_2OUT(self.ca12, self.sa12, x)
    
    id12 = self.layer2(id12)
    
    if self.ds2:
      cam12 = self.ds2(cam12)
    
    id23, cam23 = self.forward_attn_2IN_2OUT(self.ca23, self.sa23, id12, cam12)
    
    id23 = self.layer3(id23)

    if self.ds3:
      cam23 = self.ds3(cam23)

    id34, cam34 = self.forward_attn_2IN_2OUT(self.ca34, self.sa34, id23, cam23)

    id34 = self.layer4(id34)

    if self.ds4:
      cam34 = self.ds4(cam34)

    id_out, cam_out = self.forward_attn_2IN_2OUT(self.ca4e, self.sa4e, id34, cam34)

    return id_out, cam_out

# Same as v11, but with both channel and spatial splits for id/cam
# Also, two dedicated downsampling layers instead of one shared, since that seems to work better
class ResNetCbamTwin_v12(nn.Module):

  def __init__(self, layers, last_stride=2, IN=False):

    self.inplanes = 64
    super().__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    
    self.block = Bottleneck

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1, _ = self._make_layer(self.block, self.block, 64, layers[0], IN=IN)
    self.ca12 = ChannelAttention(self.inplanes)
    self.sa12 = SpatialAttention(kernel_size=3)
    self.layer2, self.ds2 = self._make_layer(self.block, self.block, 128, layers[1], stride=2, IN=IN)
    self.ca23 = ChannelAttention(self.inplanes)
    self.sa23 = SpatialAttention(kernel_size=3)
    self.layer3, self.ds3 = self._make_layer(self.block, self.block, 256, layers[2], stride=2, IN=IN)
    self.ca34 = ChannelAttention(self.inplanes)
    self.sa34 = SpatialAttention(kernel_size=3)
    self.layer4, self.ds4 = self._make_layer(self.block, self.block, 512, layers[3], stride=last_stride)
    self.ca4e = ChannelAttention(self.inplanes)
    self.sa4e = SpatialAttention(kernel_size=3)
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, headblock, afterblock, planes, blocks, stride=1, IN=False):
    downsample = None
    ds = None
    if stride != 1 or self.inplanes != planes * headblock.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * headblock.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * headblock.expansion),
      )
      ds = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * headblock.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * headblock.expansion),
      )

    layers = []
    layers.append(headblock(self.inplanes, planes, stride, downsample, IN=IN))
    self.inplanes = planes * headblock.expansion
    for _ in range(1, blocks):
      layers.append(afterblock(self.inplanes, planes))

    # return nn.Sequential(*layers), downsample
    return nn.Sequential(*layers), ds

  def forward_attn_1IN_2OUT(self, ca, sa, x):
    channel_attn = ca(x)
    spatial_attn = sa(x)

    # ID gets all the attention
    id_out = spatial_attn * channel_attn * x
    id_out += x
    id_out = self.relu(id_out)

    # Cam gets the leftovers =(
    cam_out = (-spatial_attn + 1) * (-channel_attn + 1) * x
    cam_out += x
    cam_out = self.relu(cam_out)
    return id_out, cam_out

  def forward_attn_2IN_2OUT(self, ca, sa, id_in, cam_in):
    channel_attn = ca(id_in)
    spatial_attn = sa(id_in)

    id_out = spatial_attn * channel_attn * id_in
    id_out += id_in
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * (-channel_attn + 1) * id_in
    cam_out += cam_in
    cam_out = self.relu(cam_out)
    return id_out, cam_out

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    id12, cam12 = self.forward_attn_1IN_2OUT(self.ca12, self.sa12, x)
    
    id12 = self.layer2(id12)
    
    if self.ds2:
      cam12 = self.ds2(cam12)
    
    id23, cam23 = self.forward_attn_2IN_2OUT(self.ca23, self.sa23, id12, cam12)
    
    id23 = self.layer3(id23)

    if self.ds3:
      cam23 = self.ds3(cam23)

    id34, cam34 = self.forward_attn_2IN_2OUT(self.ca34, self.sa34, id23, cam23)

    id34 = self.layer4(id34)

    if self.ds4:
      cam34 = self.ds4(cam34)

    id_out, cam_out = self.forward_attn_2IN_2OUT(self.ca4e, self.sa4e, id34, cam34)

    return id_out, cam_out

# Same as v12, with CBAM between last 3 conv_blocks
class ResNetCbamTwin_v13(nn.Module):

  def __init__(self, layers, last_stride=2, IN=False):

    self.inplanes = 64
    super().__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    
    self.block = Bottleneck

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1, _ = self._make_layer(self.block, self.block, 64, layers[0], IN=IN)
    self.layer2, _ = self._make_layer(self.block, self.block, 128, layers[1], stride=2, IN=IN)
    self.ca23 = ChannelAttention(self.inplanes)
    self.sa23 = SpatialAttention(kernel_size=3)
    self.layer3, self.ds3 = self._make_layer(self.block, self.block, 256, layers[2], stride=2, IN=IN)
    self.ca34 = ChannelAttention(self.inplanes)
    self.sa34 = SpatialAttention(kernel_size=3)
    self.layer4, self.ds4 = self._make_layer(self.block, self.block, 512, layers[3], stride=last_stride)
    self.ca4e = ChannelAttention(self.inplanes)
    self.sa4e = SpatialAttention(kernel_size=3)
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, headblock, afterblock, planes, blocks, stride=1, IN=False):
    downsample = None
    ds = None
    if stride != 1 or self.inplanes != planes * headblock.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * headblock.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * headblock.expansion),
      )
      ds = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * headblock.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * headblock.expansion),
      )

    layers = []
    layers.append(headblock(self.inplanes, planes, stride, downsample, IN=IN))
    self.inplanes = planes * headblock.expansion
    for _ in range(1, blocks):
      layers.append(afterblock(self.inplanes, planes))

    # return nn.Sequential(*layers), downsample
    return nn.Sequential(*layers), ds

  def forward_attn_1IN_2OUT(self, ca, sa, x):
    channel_attn = ca(x)
    spatial_attn = sa(x)

    # ID gets all the attention
    id_out = spatial_attn * channel_attn * x
    id_out += x
    id_out = self.relu(id_out)

    # Cam gets the leftovers =(
    cam_out = (-spatial_attn + 1) * (-channel_attn + 1) * x
    cam_out += x
    cam_out = self.relu(cam_out)
    return id_out, cam_out

  def forward_attn_2IN_2OUT(self, ca, sa, id_in, cam_in):
    channel_attn = ca(id_in)
    spatial_attn = sa(id_in)

    id_out = spatial_attn * channel_attn * id_in
    id_out += id_in
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * (-channel_attn + 1) * id_in
    cam_out += cam_in
    cam_out = self.relu(cam_out)
    return id_out, cam_out

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    id23, cam23 = self.forward_attn_1IN_2OUT(self.ca23, self.sa23, x)    
    
    id23 = self.layer3(id23)

    if self.ds3:
      cam23 = self.ds3(cam23)

    id34, cam34 = self.forward_attn_2IN_2OUT(self.ca34, self.sa34, id23, cam23)

    id34 = self.layer4(id34)

    if self.ds4:
      cam34 = self.ds4(cam34)

    id_out, cam_out = self.forward_attn_2IN_2OUT(self.ca4e, self.sa4e, id34, cam34)

    return id_out, cam_out

# Same as v12, with CBAM between last 2 conv_blocks
class ResNetCbamTwin_v14(nn.Module):

  def __init__(self, layers, last_stride=2, IN=False):

    self.inplanes = 64
    super().__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    
    self.block = Bottleneck

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1, _ = self._make_layer(self.block, self.block, 64, layers[0], IN=IN)
    self.layer2, _ = self._make_layer(self.block, self.block, 128, layers[1], stride=2, IN=IN)
    self.layer3, _ = self._make_layer(self.block, self.block, 256, layers[2], stride=2, IN=IN)
    self.ca34 = ChannelAttention(self.inplanes)
    self.sa34 = SpatialAttention(kernel_size=3)
    self.layer4, self.ds4 = self._make_layer(self.block, self.block, 512, layers[3], stride=last_stride)
    self.ca4e = ChannelAttention(self.inplanes)
    self.sa4e = SpatialAttention(kernel_size=3)
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, headblock, afterblock, planes, blocks, stride=1, IN=False):
    downsample = None
    ds = None
    if stride != 1 or self.inplanes != planes * headblock.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * headblock.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * headblock.expansion),
      )
      ds = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * headblock.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * headblock.expansion),
      )

    layers = []
    layers.append(headblock(self.inplanes, planes, stride, downsample, IN=IN))
    self.inplanes = planes * headblock.expansion
    for _ in range(1, blocks):
      layers.append(afterblock(self.inplanes, planes))

    # return nn.Sequential(*layers), downsample
    return nn.Sequential(*layers), ds

  def forward_attn_1IN_2OUT(self, ca, sa, x):
    channel_attn = ca(x)
    spatial_attn = sa(x)

    # ID gets all the attention
    id_out = spatial_attn * channel_attn * x
    id_out += x
    id_out = self.relu(id_out)

    # Cam gets the leftovers =(
    cam_out = (-spatial_attn + 1) * (-channel_attn + 1) * x
    cam_out += x
    cam_out = self.relu(cam_out)
    return id_out, cam_out

  def forward_attn_2IN_2OUT(self, ca, sa, id_in, cam_in):
    channel_attn = ca(id_in)
    spatial_attn = sa(id_in)

    id_out = spatial_attn * channel_attn * id_in
    id_out += id_in
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * (-channel_attn + 1) * id_in
    cam_out += cam_in
    cam_out = self.relu(cam_out)
    return id_out, cam_out

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    
    id34, cam34 = self.forward_attn_1IN_2OUT(self.ca34, self.sa34, x)    

    id34 = self.layer4(id34)

    if self.ds4:
      cam34 = self.ds4(cam34)

    id_out, cam_out = self.forward_attn_2IN_2OUT(self.ca4e, self.sa4e, id34, cam34)

    return id_out, cam_out

# Starting a new train of thought:
# What if instead of CBAM-ing the head of the model, we just CBAM the tail?
class ResNetCbamTwinTail(nn.Module):

  def __init__(self, layers, last_stride=2, IN=False):

    self.inplanes = 64
    super(ResNetCbamTwinTail, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    
    self.block = Bottleneck

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(self.block, 64, layers[0], IN=IN)
    self.ca12 = ChannelAttention(self.inplanes)
    self.sa12 = SpatialAttention(kernel_size=3)
    self.layer2 = self._make_layer(self.block, 128, layers[1], stride=2, IN=IN)
    # self.ca23 = ChannelAttention(self.inplanes)
    # self.sa23 = SpatialAttention(kernel_size=3)
    self.layer3 = self._make_layer(self.block, 256, layers[2], stride=2, IN=IN)
    # self.ca34 = ChannelAttention(self.inplanes)
    # self.sa34 = SpatialAttention(kernel_size=3)
    self.layer4 = self._make_layer(self.block, 512, layers[3], stride=last_stride)
    # self.ca4e = ChannelAttention(self.inplanes)
    # self.sa4e = SpatialAttention(kernel_size=3)
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, IN=False):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, IN=IN))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward_attn_1IN_2OUT(self, ca, sa, x):
    channel_attn = ca(x)
    spatial_attn = sa(x)

    # "filter" out the important channels
    channel_out = channel_attn * x

    id_out = spatial_attn * channel_out
    id_out += x
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * channel_out
    cam_out += x
    cam_out = self.relu(cam_out)
    return id_out, cam_out

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x, cam_out = self.forward_attn_1IN_2OUT(self.ca12, self.sa12, x)
    
    x = self.layer2(x)
    x = self.layer3(x)
    id_out = self.layer4(x)

    return id_out, cam_out

  def forward_spatial_attn(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    spatial_attn = self.sa12(x)

    return [ spatial_attn ]

# In this version, we apply to two tail nodes instead of the tail one
class ResNetCbamTwinTail_v2(nn.Module):

  def __init__(self, layers, last_stride=2, IN=False):

    self.inplanes = 64
    super(ResNetCbamTwinTail_v2, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    
    self.block = Bottleneck

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(self.block, 64, layers[0], IN=IN)
    self.ca12 = ChannelAttention(self.inplanes)
    self.sa12 = SpatialAttention(kernel_size=3)
    self.layer2 = self._make_layer(self.block, 128, layers[1], stride=2, IN=IN)
    self.ca23 = ChannelAttention(self.inplanes)
    self.sa23 = SpatialAttention(kernel_size=3)
    self.layer3 = self._make_layer(self.block, 256, layers[2], stride=2, IN=IN)
    # self.ca34 = ChannelAttention(self.inplanes)
    # self.sa34 = SpatialAttention(kernel_size=3)
    self.layer4 = self._make_layer(self.block, 512, layers[3], stride=last_stride)
    # self.ca4e = ChannelAttention(self.inplanes)
    # self.sa4e = SpatialAttention(kernel_size=3)
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, IN=False):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, IN=IN))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward_attn_1IN_2OUT(self, ca, sa, x):
    channel_attn = ca(x)
    spatial_attn = sa(x)

    # "filter" out the important channels
    channel_out = channel_attn * x

    id_out = spatial_attn * channel_out
    id_out += x
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * channel_out
    cam_out += x
    cam_out = self.relu(cam_out)
    return id_out, cam_out

  # def forward_attn_2IN_2OUT(self, ca, sa, id_in, cam_in):
  #   channel_attn = ca(id_in)
  #   spatial_attn = sa(id_in)

  #   # "filter" out the important channels
  #   channel_out = channel_attn * id_in

  #   id_out = spatial_attn * channel_out
  #   id_out += id_in
  #   id_out = self.relu(id_out)

  #   cam_out = (-spatial_attn + 1) * channel_out
  #   cam_out += cam_in
  #   cam_out = self.relu(cam_out)
  #   return id_out, cam_out

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    id1 = self.layer1(x)
    id12, cam12 = self.forward_attn_1IN_2OUT(self.ca12, self.sa12, id1)
    
    id2 = self.layer2(id12)

    id23, cam23 = self.forward_attn_1IN_2OUT(self.ca23, self.sa23, id2)
    id3 = self.layer3(id23)
    id_out = self.layer4(id3)

    cam_out = [cam12, cam23]

    return id_out, cam_out

# In this version, we change the way the 1INPUT2OUTPUT works
class ResNetCbamTwinTail_v3(ResNetCbamTwinTail):
  def forward_attn_1IN_2OUT(self, ca, sa, x):
    channel_out = ca(x) * x
    spatial_attn = sa(channel_out)

    id_out = spatial_attn * channel_out
    id_out += x
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * channel_out
    cam_out += x
    cam_out = self.relu(cam_out)
    return id_out, cam_out

# Added inverse channel on top of inverse spatial attention
class ResNetCbamTwinTail_v4(ResNetCbamTwinTail):

  # In this version, channel and spatial are not intertwined
  def forward_attn_1IN_2OUT(self, ca, sa, x):
    channel_attn = ca(x)
    spatial_attn = sa(x)

    id_out = spatial_attn * channel_attn * x
    id_out += x
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * (-channel_attn + 1) * x
    cam_out += x
    cam_out = self.relu(cam_out)
    return id_out, cam_out

# Added inverse channel on top of inverse spatial attention
class ResNetCbamTwinTail_v5(ResNetCbamTwinTail):

  # In this version, channel and spatial ARE intertwined
  def forward_attn_1IN_2OUT(self, ca, sa, x):
    channel_attn = ca(x)
    channel_out = channel_attn * x

    spatial_attn = sa(channel_out)

    id_out = spatial_attn * channel_out
    id_out += x
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * (-channel_attn + 1) * x
    cam_out += x
    cam_out = self.relu(cam_out)
    return id_out, cam_out

  def forward_spatial_attn(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)

    channel_attn = self.ca12(x)
    channel_out = channel_attn * x
    spatial_attn = self.sa12(channel_out)

    return [ spatial_attn ]

# For ablation study
# CAPRA after layer2
class ResNetCapraLayer2_v5(nn.Module):
  def __init__(self, layers, last_stride=2, IN=False):

    self.inplanes = 64
    super(ResNetCapraLayer2_v5, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    
    self.block = Bottleneck

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(self.block, 64, layers[0], IN=IN)
    self.layer2 = self._make_layer(self.block, 128, layers[1], stride=2, IN=IN)
    self.ca = ChannelAttention(self.inplanes)
    self.sa = SpatialAttention(kernel_size=3)
    self.layer3 = self._make_layer(self.block, 256, layers[2], stride=2, IN=IN)
    self.layer4 = self._make_layer(self.block, 512, layers[3], stride=last_stride)
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, IN=False):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, IN=IN))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward_attn_1IN_2OUT(self, ca, sa, x):
    channel_attn = ca(x)
    channel_out = channel_attn * x

    spatial_attn = sa(channel_out)

    id_out = spatial_attn * channel_out
    id_out += x
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * (-channel_attn + 1) * x
    cam_out += x
    cam_out = self.relu(cam_out)
    return id_out, cam_out

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x, cam_out = self.forward_attn_1IN_2OUT(self.ca, self.sa, x)
    x = self.layer3(x)
    id_out = self.layer4(x)

    return id_out, cam_out

  def forward_spatial_attn(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)

    channel_attn = self.ca(x)
    channel_out = channel_attn * x
    spatial_attn = self.sa(channel_out)

    return [ spatial_attn ]

# For ablation study
# CAPRA after layer3
class ResNetCapraLayer3_v5(nn.Module):
  def __init__(self, layers, last_stride=2, IN=False):

    self.inplanes = 64
    super(ResNetCapraLayer3_v5, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    
    self.block = Bottleneck

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(self.block, 64, layers[0], IN=IN)
    self.layer2 = self._make_layer(self.block, 128, layers[1], stride=2, IN=IN)
    self.layer3 = self._make_layer(self.block, 256, layers[2], stride=2, IN=IN)
    self.ca = ChannelAttention(self.inplanes)
    self.sa = SpatialAttention(kernel_size=3)
    self.layer4 = self._make_layer(self.block, 512, layers[3], stride=last_stride)
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, IN=False):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, IN=IN))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward_attn_1IN_2OUT(self, ca, sa, x):
    channel_attn = ca(x)
    channel_out = channel_attn * x

    spatial_attn = sa(channel_out)

    id_out = spatial_attn * channel_out
    id_out += x
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * (-channel_attn + 1) * x
    cam_out += x
    cam_out = self.relu(cam_out)
    return id_out, cam_out

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x, cam_out = self.forward_attn_1IN_2OUT(self.ca, self.sa, x)
    id_out = self.layer4(x)

    return id_out, cam_out

  def forward_spatial_attn(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)

    channel_attn = self.ca(x)
    channel_out = channel_attn * x
    spatial_attn = self.sa(channel_out)

    return [ spatial_attn ]

# For ablation study
# CAPRA only at the head of the network, exploiting high-level features
class ResNetCapraLayer4_v5(nn.Module):
  def __init__(self, layers, last_stride=2, IN=False):

    self.inplanes = 64
    super(ResNetCapraLayer4_v5, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    
    self.block = Bottleneck

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(self.block, 64, layers[0], IN=IN)
    self.layer2 = self._make_layer(self.block, 128, layers[1], stride=2, IN=IN)
    self.layer3 = self._make_layer(self.block, 256, layers[2], stride=2, IN=IN)
    self.layer4 = self._make_layer(self.block, 512, layers[3], stride=last_stride)
    self.ca = ChannelAttention(self.inplanes)
    self.sa = SpatialAttention(kernel_size=3)
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, IN=False):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, IN=IN))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward_attn_1IN_2OUT(self, ca, sa, x):
    channel_attn = ca(x)
    channel_out = channel_attn * x

    spatial_attn = sa(channel_out)

    id_out = spatial_attn * channel_out
    id_out += x
    id_out = self.relu(id_out)

    cam_out = (-spatial_attn + 1) * (-channel_attn + 1) * x
    cam_out += x
    cam_out = self.relu(cam_out)
    return id_out, cam_out

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    id_out, cam_out = self.forward_attn_1IN_2OUT(self.ca, self.sa, x)

    return id_out, cam_out

# For ablation study
# CBAM after conv block 1, no multitask learning
class ResNetCbamTailOnly(nn.Module):

  def __init__(self, layers, last_stride=2, IN=False):

    self.inplanes = 64
    super(ResNetCbamTailOnly, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    
    self.block = Bottleneck

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(self.block, 64, layers[0], IN=IN)
    self.ca = ChannelAttention(self.inplanes)
    self.sa = SpatialAttention(kernel_size=3)
    self.layer2 = self._make_layer(self.block, 128, layers[1], stride=2, IN=IN)
    self.layer3 = self._make_layer(self.block, 256, layers[2], stride=2, IN=IN)
    self.layer4 = self._make_layer(self.block, 512, layers[3], stride=last_stride)
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, IN=False):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, IN=IN))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward_attn(self, ca, sa, x):
    channel_attn = ca(x)
    channel_out = channel_attn * x

    spatial_attn = sa(channel_out)

    id_out = spatial_attn * channel_out
    id_out += x
    id_out = self.relu(id_out)

    return id_out

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.forward_attn(self.ca, self.sa, x)
    
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x
'''
END MODEL DEFINITIONS
'''

'''
Schemes

Singles
- Last (done)
- Last-Twin 
- Bottleneck (done)
- Bottleneck-Twin

Combinations
- Bottleneck+Last
- Bottleneck+Last-Twin
- Bottleneck-Twin+Last
- Bottlenect-Twin+Last-Twin

Mixing other techniques
- Adversarial
- Extreme Label Smoothing

'''
def resnet50_multitask(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model with multi-task id/camera learning
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetMultiTask(Bottleneck, [3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_last(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model with CBAM only at the end. CBAM is based on the paper:
  CBAM: Convolutional Block Attention Module (ECCV 2018)
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamLast(Bottleneck, [3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_last_twin(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model with CBAM only at the end, 
     and applied to PID features and CAMID features separately. 
     CBAM is based on the paper:
     CBAM: Convolutional Block Attention Module (ECCV 2018)
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamLastTwin(Bottleneck, [3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_last_channel_spatial_twin(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model with CBAM only at the end, 
     and applied to PID features and CAMID features separately. 
     CBAM is based on the paper:
     CBAM: Convolutional Block Attention Module (ECCV 2018)
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamLastChannelSpatialTwin(Bottleneck, [3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_last_twin_septrip(pretrained=False, **kwargs):
  """ TODO: A proper description please. 
     CBAM is based on the paper:
     CBAM: Convolutional Block Attention Module (ECCV 2018)
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamLastTwinSeparateTripletAttn(Bottleneck, [3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_last_twin_spatial_only(pretrained=False, **kwargs):
  """ TODO: A proper description please. 
     CBAM is based on the paper:
     CBAM: Convolutional Block Attention Module (ECCV 2018)
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamLastTwinSpatialOnly(Bottleneck, [3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_last_twin_channel_only(pretrained=False, **kwargs):
  """ TODO: A proper description please. 
     CBAM is based on the paper:
     CBAM: Convolutional Block Attention Module (ECCV 2018)
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamLastTwinChannelOnly(Bottleneck, [3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_twin(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model with twin CBAM in all res-blocks, but not at the end. 
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamTwin(Bottleneck, BottleneckAttnSingleInputTwinOutput, BottleneckAttnTwinInputTwinOutput, [3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_twin_v2(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model with ver. 2 twin CBAM in most of the later res-blocks.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  # In v2 we pre-determine the layers to build, so no need to pass in as args
  model = ResNetCbamTwin_v2([3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_twin_v3(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model with ver. 3 twin CBAM in most of the later res-blocks.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamTwin_v3([3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_twin_v4(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model with ver. 4 twin CBAM in most of the later res-blocks.
     Version 4 is mostly just putting in BN into the channel/spatial attention and seeing what
     happens
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamTwin_v4([3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_twin_v5(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model with ver. 5 twin CBAM in most of the later res-blocks.
     Version 5 retains all the residual addition with id channel, but shifts the relu to the outside
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamTwin_v5([3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_twin_v6(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model with ver. 6 twin CBAM in most of the later res-blocks.
     Version 6 does not add residue from id channel to the cam channel, except at the very end
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamTwin_v6([3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_twin_v7(pretrained=False, **kwargs):
  """ Totally new remake
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamTwin_v7([3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_twin_v8(pretrained=False, **kwargs):
  """ Totally new remake
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamTwin_v8([3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_twin_v9(pretrained=False, **kwargs):
  """ Totally new remake
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamTwin_v9([3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_twin_v10(pretrained=False, **kwargs):
  """ Totally new remake
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamTwin_v10([3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_twin_v11(pretrained=False, **kwargs):
  """ Totally new remake
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamTwin_v11([3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_twin_v12(pretrained=False, **kwargs):
  """ Totally new remake
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamTwin_v12([3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_twin_v13(pretrained=False, **kwargs):
  """ Totally new remake
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamTwin_v13([3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_twin_v14(pretrained=False, **kwargs):
  """ Totally new remake
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamTwin_v14([3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_twin_tail(pretrained=False, **kwargs):
  """ Totally new remake
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamTwinTail([3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_twin_tail_v2(pretrained=False, **kwargs):
  """ Totally new remake
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamTwinTail_v2([3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_twin_tail_v3(pretrained=False, **kwargs):
  """ Totally new remake
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamTwinTail_v3([3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_twin_tail_v4(pretrained=False, **kwargs):
  """ Totally new remake
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamTwinTail_v4([3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CBAM_twin_tail_v5(pretrained=False, **kwargs):
  """ Totally new remake
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCbamTwinTail_v5([3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CAPRA_layer2_v5(pretrained=False, **kwargs):
  """ Totally new remake
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCapraLayer2_v5([3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CAPRA_layer3_v5(pretrained=False, **kwargs):
  """ Totally new remake
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCapraLayer3_v5([3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

def resnet50_CAPRA_layer4_v5(pretrained=False, **kwargs):
  """ Totally new remake
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNetCapraLayer4_v5([3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

# For ablation study. CBAM in convblock1 only
def resnet50_CBAM_tail_only(pretrained=False, **kwargs):
  model = ResNetCbamTailOnly([3, 4, 6, 3], **kwargs)
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model
  

def resnet50_CBAM(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model with CBAM in every res-block. CBAM is based on the paper:
  CBAM: Convolutional Block Attention Module (ECCV 2018)
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BottleneckAttn, [3, 4, 6, 3], **kwargs)
  
  if pretrained:
    pretrain_dict = remove_fc(model_zoo.load_url(model_urls['resnet50']))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
  return model

'''
Older models copied over from resnet.py (might reinstate)
'''

# def resnet18(pretrained=False, **kwargs):
#   """Constructs a ResNet-18 model.
#   Args:
#       pretrained (bool): If True, returns a model pre-trained on ImageNet
#   """
#   model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#   if pretrained:
#     model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet18'])))
#   return model


# def resnet34(pretrained=False, **kwargs):
#   """Constructs a ResNet-34 model.
#   Args:
#       pretrained (bool): If True, returns a model pre-trained on ImageNet
#   """
#   model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#   if pretrained:
#     model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet34'])))
#   return model


# def resnet50(pretrained=False, **kwargs):
#   """Constructs a ResNet-50 model.
#   Args:
#       pretrained (bool): If True, returns a model pre-trained on ImageNet
#   """
#   model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#   if pretrained:
#     model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet50'])))
#   return model

# def resnet101(pretrained=False, **kwargs):
#   """Constructs a ResNet-101 model.
#   Args:
#       pretrained (bool): If True, returns a model pre-trained on ImageNet
#   """
#   model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
#   if pretrained:
#     model.load_state_dict(
#       remove_fc(model_zoo.load_url(model_urls['resnet101'])))
#   return model


# def resnet152(pretrained=False, **kwargs):
#   """Constructs a ResNet-152 model.
#   Args:
#       pretrained (bool): If True, returns a model pre-trained on ImageNet
#   """
#   model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
#   if pretrained:
#     model.load_state_dict(
#       remove_fc(model_zoo.load_url(model_urls['resnet152'])))
#   return model