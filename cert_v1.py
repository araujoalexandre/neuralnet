
import numpy as np
import torch
from scipy.stats import norm as sp_norm


class Certificate:

  def __init__(self):

    self.sigma0 = 0.25
    self.pABar_sigma0 = 0.90
    self.pABar_sigma1 = 0.90
    self.dim = 20
    self.sample = 10000000
    self.device = 'cuda'
    self.close_below_tol = 0.001
    self.binary_search_tol = 0.000001

    self.delta = torch.zeros(self.dim, device=self.device)

    self.cohen_cert = self.sigma0 * sp_norm.ppf(self.pABar_sigma0)
    print('cert cohen {:.4f}'.format(self.cohen_cert))

  def make_noise(self, sigma):
    return torch.randn(self.sample, self.dim, device=self.device) * sigma

  def left_term(self, noise):
    return torch.exp(-(self.eps**2 - torch.matmul(noise, self.delta)) / (2*self.sigma0**2))

  def right_term(self, noise):
    return torch.exp((-torch.norm(noise, p=2, dim=1)**2 / 2) * (1/self.sigma1**2 - 1/self.sigma0**2))

  def compute_set(self, noise, k1, k2):
    return torch.mean(
      (self.left_term(noise) <= k1 + k2 * self.right_term(noise)) * 1.
    )

  def find_k1(self, k2):
    noise = self.make_noise(self.sigma0)
    bound = self.left_term(noise) - k2 * self.right_term(noise)
    k1 = torch.quantile(bound, self.pABar_sigma0)
    return k1

  def find_k2(self, k1):
    noise = self.make_noise(self.sigma1)
    bound = (self.left_term(noise) - k1) / self.right_term(noise)
    # print('left term', self.left_term(noise))
    # print('right term', self.right_term(noise))
    k2 = torch.quantile(bound, self.pABar_sigma1)
    return k2

  def compute(self, sigma1):
    """ Binary search to find max certificate radius """
    self.sigma1 = sigma1
    start = round((self.cohen_cert - 0.02) * 100) / 100
    end = start + 0.4
    while (end - start) > self.binary_search_tol:
      print('start {:.4f}, end {:.4f}'.format(start, end))
      eps = (start + end) / 2
      if self.compute_cert(eps) >= 1/2:
        start = eps
      else:
        end = eps
      # break
    return start

  def compute_cert(self, eps):
    self.eps = eps
    self.delta[0] = eps
    k2 = 0.
    p1, p2 = 2., 2.

    pABar_sigma0 = self.pABar_sigma0
    pABar_sigma1 = self.pABar_sigma1
    is_close_below = lambda x, y: y - self.close_below_tol <= x <= y

    while not is_close_below(p1, pABar_sigma0) or not is_close_below(p2, pABar_sigma1):
      k1 = self.find_k1(k2)
      k2 = self.find_k2(k1)
      p1 = self.compute_set(self.make_noise(self.sigma0), k1, k2)
      p2 = self.compute_set(self.make_noise(self.sigma1), k1, k2)
      # print(f'k1 {k1}')
      # print(f'k2 {k2}')
      # print()

    cert = self.compute_set(self.make_noise(self.sigma0) + self.delta, k1, k2)
    return cert




if __name__ == '__main__':

  cert = Certificate()

  # for sigma1 in np.arange(0.25, 1.5, 0.05):
  # for eps in [0.01, 0.02, 0.03, 0.04]:
  #   radius = cert.compute(0.25 + eps)
  #   print('sigma1 {:.3f} cert {:.4f}'.format(0.25 + eps, radius))

  radius = cert.compute(0.25 + 0.000001)
  print('cert {:.4f}'.format(radius))



