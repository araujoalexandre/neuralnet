
from collections import defaultdict
from itertools import product
from functools import reduce
import logging

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


class LipschitzBound:

  def __init__(self, kernel_shape, padding, sample=50, backend='torch',
               cuda=True):

    self.kernel_shape = kernel_shape
    self.padding = padding
    self.sample = sample
    self.backend = backend
    self.cuda = cuda

    cout, cin, ksize, _ = kernel_shape

    # verify the kernel is square
    if not kernel_shape[-1] == kernel_shape[-2]:
      raise ValueError("The last 2 dim of the kernel must be equal.")
    # verify if all kernel have odd shape
    if not kernel_shape[-1] % 2 == 1:
      raise ValueError("The dimension of the kernel must be odd.")

    # define search space
    x = np.linspace(0, 2*np.pi, num=self.sample)
    w = np.array(list(product(x, x)))
    self.w0 = w[:, 0].reshape(-1, 1)
    self.w1 = w[:, 1].reshape(-1, 1)

    # convert search space to torch tensor
    if self.backend == 'torch':
      self.w0 = torch.FloatTensor(np.float32(self.w0))
      self.w1 = torch.FloatTensor(np.float32(self.w1))
      if self.cuda:
        self.w0 = self.w0.cuda()
        self.w1 = self.w1.cuda()

   # create samples
    if self.backend == 'numpy':
      p_index = np.arange(-ksize + 1., 1.) + padding
      H0 = 1j * np.tile(p_index, ksize).reshape(ksize, ksize).T.reshape(-1)
      H1 = 1j * np.tile(p_index, ksize)
      self.samples = np.exp(self.w0 * H0 + self.w1 * H1).T

    elif self.backend == 'torch':
      p_index = torch.arange(-ksize + 1.0, 1.0) + padding
      H0 = p_index.repeat(ksize).reshape(ksize, ksize).T.reshape(-1)
      H1 = p_index.repeat(ksize)
      if self.cuda:
        H0 = H0.cuda()
        H1 = H1.cuda()
      real = torch.cos(self.w0 * H0 + self.w1 * H1).T
      imag = torch.sin(self.w0 * H0 + self.w1 * H1).T
      self.samples = (real, imag)

  def compute(self, *args, **kwargs):
    if self.backend == 'torch':
      return self._compute_from_torch(*args, **kwargs)
    return self._compute_from_numpy(*args, **kwargs)

  def _compute_from_numpy(self, kernel):
    """Compute the LipGrid Algorithm."""
    pad = self.padding
    cout, cin, ksize, _ = kernel.shape
    if cout > cin:
      kernel = np.transpose(kernel, axes=[1, 0, 2, 3])
      cout, cin = cin, cout
    ker = kernel.reshape(cout, cin, -1)[..., np.newaxis]
    poly = (ker * self.samples).sum(axis=2)
    poly = np.square(np.abs(poly)).sum(axis=1)
    sv_max = np.sqrt(poly.max(axis=-1).sum())
    return sv_max

  def _compute_from_torch_naive(self, kernel):
    """Compute the LipGrid Algo with Torch"""
    pad = self.padding
    cout, cin, ksize, _ = kernel.shape
    if cout > cin:
      kernel = torch.transpose(kernel, 0, 1)
      cout, cin = cin, cout

    ker = kernel.view(cout, cin, -1, 1)
    real, imag = self.samples

    poly_real = torch.mul(ker, real).sum(axis=2)
    poly_imag = torch.mul(ker, imag).sum(axis=2)

    poly = torch.mul(poly_real, poly_real) + \
        torch.mul(poly_imag, poly_imag)
    poly = poly.sum(axis=1)
    sv_max = torch.sqrt(poly.max(axis=-1)[0].sum())
    return sv_max

  def _compute_from_torch(self, kernel):
    """Compute the LipGrid Algo with Torch"""
    pad = self.padding
    cout, cin, ksize, _ = kernel.shape
    # special case kernel 1x1
    if ksize == 1:
      ker = kernel.reshape(-1)
      return torch.sqrt(torch.einsum('i,i->', ker, ker)) 

    if cout > cin:
      kernel = torch.transpose(kernel, 0, 1)
      cout, cin = cin, cout

    real, imag = self.samples
    ker = kernel.reshape(cout*cin, -1)
    poly_real = torch.matmul(ker, real).view(cout, cin, -1)
    poly_imag = torch.matmul(ker, imag).view(cout, cin, -1)

    poly1 = torch.einsum('ijk,ijk->ik', poly_real, poly_real)
    poly2 = torch.einsum('ijk,ijk->ik', poly_imag, poly_imag)
    poly = poly1 + poly2

    sv_max = torch.sqrt(poly.max(axis=-1)[0].sum())
    return sv_max



class LipschitzRegularization:

  def __init__(self, model, params, reader, local_rank):

    self.params = params
    self.decay = self.params.lipschitz_decay

    self.conv_id = set()
    self.batch_bn_id = set()
    self.linear_id = set()

    for i, module in enumerate(model.modules()):
      name = module.__class__.__name__.lower()
      if 'conv2d' in name:
        self.conv_id.add(i)
      elif 'batchnorm' in name:
        self.batch_bn_id.add(i)

    self.lip_bound_cls = {}
    self.sample = params.lipschitz_bound_sample

    # we pre-create LipschitzBound for each kernel
    for i, module in enumerate(model.modules()):
      name = module.__class__.__name__
      if len(getattr(module, 'weight', [])):
        if len(module.weight.shape) == 4:
          padding = module.padding[0]
          kernel = module.weight
          self.lip_bound_cls[i] = \
            LipschitzBound(kernel.shape, padding, sample=self.sample)

  def _compute_batch_norm(self, module):
    """Compute the Lipschitz of Batch Norm layer."""
    weight = module.weight
    running_var = module.running_var
    eps = module.eps
    values = torch.abs(weight / torch.sqrt(running_var + eps))
    lipbound = torch.max(values)
    return lipbound

  def _compute_conv(self, i, module):
    """Compute a bound on the Lipchitz of Convolution layer."""
    padding = module.padding[0]
    kernel = module.weight
    lipbound = self.lip_bound_cls[i].compute(kernel)
    return lipbound

  def compute_full_network(self, model):
    """Compute Lipschitz of full Network."""
    lip_loss = []
    for i, module in enumerate(model.modules()):
      name = module.__class__.__name__
      if i in self.conv_id:
        lip_conv = self._compute_conv(i, module)
        lip_loss.append(lip_conv)
      elif i in self.batch_bn_id:
        lip_bn = self._compute_batch_norm(module)
        lip_loss.append(lip_bn)
      elif i in self.linear_id:
        lip_linear = self._compute_linear_sv(module)
        lip_loss.append(lip_linear)
    return lip_loss

  def get_loss(self, epoch, model):
    lip_cst = self.compute_full_network(model)
    lipreg = self.decay * sum([torch.log(x) for x in lip_cst])
    return lipreg



