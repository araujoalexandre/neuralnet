
import numpy as np
import torch
from scipy.stats import norm as sp_norm


class Certificate:

  def __init__(self):

    self.sigma0 = 0.25
    self.pABar_sigma0 = 0.90
    self.pABar_sigma1 = 0.90
    # self.dim = 2
    self.dim_data = 10
    self.sample = 15000000
    self.device = 'cuda'
    self.close_below_tol = 0.001
    self.binary_search_tol = 0.001

    # self.delta = torch.zeros(self.dim, device=self.device)

    self.cohen_cert = self.sigma0 * sp_norm.ppf(self.pABar_sigma0)
    print('cert cohen {:.4f}'.format(self.cohen_cert))

  def define_delta(self, dim, eps):
    delta = torch.zeros(dim, device=self.device)
    delta[-1] = eps
    return delta

  def make_noise(self, sigma, dim):
    return torch.randn(self.sample, dim, device=self.device) * sigma

  def left_term(self, noise, dim):
    delta = self.define_delta(dim, self.eps)
    term = torch.exp(-torch.norm(noise - delta, p=2, dim=1)**2 / (2 * self.sigma0**2))
    return term

  def k1_term(self, noise):
    term = torch.exp(-torch.norm(noise, p=2, dim=1)**2 / (2 * self.sigma0**2))
    return term

  def k2_term(self, noise):
    term = torch.exp(-torch.norm(noise, p=2, dim=1)**2 / (2 * self.sigma1**2))
    return term

  def compute_set(self, noise, k1, k2, dim):
    return torch.mean(
      (self.left_term(noise, dim) <= k1 * self.k1_term(noise) + k2 * self.k2_term(noise)) * 1.
    )

  def compute_ev(self, noise, k1, k2, dim):
    in_set = self.left_term(noise, dim) <= k1 * self.k1_term(noise) + k2 * self.k2_term(noise)
    return torch.mean(torch.abs(noise[in_set][:, 0]**(self.dim_data-2))) / 2

  def find_k1(self, k2, dim):
    noise = self.make_noise(self.sigma0, dim)
    bound = (self.left_term(noise, dim) - k2 * self.k2_term(noise)) / self.k1_term(noise)
    k1 = torch.quantile(bound, self.pABar_sigma0)
    return k1

  def find_k2(self, k1, dim):
    noise = self.make_noise(self.sigma1, dim)
    bound = (self.left_term(noise, dim) - k1 * self.k1_term(noise)) / self.k2_term(noise)
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
    k2 = 0.
    p1, p2 = 2., 2.

    make_noise = self.make_noise
    pABar_sigma0 = self.pABar_sigma0
    pABar_sigma1 = self.pABar_sigma1
    is_close_below = lambda x, y: y - self.close_below_tol <= x <= y

    while not is_close_below(p1, pABar_sigma0) or not is_close_below(p2, pABar_sigma1):
      k1 = self.find_k1(k2, self.dim_data)
      k2 = self.find_k2(k1, self.dim_data)
      p1 = self.compute_set(make_noise(self.sigma0, self.dim_data), k1, k2, self.dim_data)
      p2 = self.compute_set(make_noise(self.sigma1, self.dim_data), k1, k2, self.dim_data)

    # need one last adjustment of k1
    # k1 = self.find_k1(k2, self.dim_data)

    # print(f'k1={k1:.4f} k2={k2:.4f}')

    # p1 = self.compute_set(make_noise(self.sigma0, self.dim_data), k1, k2, self.dim_data)
    # p2 = self.compute_set(make_noise(self.sigma1, self.dim_data), k1, k2, self.dim_data)

    # print(f'{p1:.7f} / {pABar_sigma0:.2f}')
    # print(f'{p2:.7f} / {pABar_sigma1:.2f}')
    # print()

    # assert is_close_below(p1, pABar_sigma0)
    # assert is_close_below(p2, pABar_sigma1)

    # ev1 = self.compute_ev(make_noise(self.sigma0, self.dim), k1, k2, self.dim)
    # ev2 = self.compute_ev(make_noise(self.sigma1, self.dim), k1, k2, self.dim)
    # print(f'ev: {ev1:.7f} / {ev2:.7f}')

    # cst1 = pABar_sigma0 / ev1 
    # cst2 = pABar_sigma1 / ev2
    # print(f'cst: {cst1:.4f} / {cst2:.4f}')

    delta = self.define_delta(self.dim_data, eps)
    # cert = self.compute_ev(make_noise(self.sigma0, self.dim) + self.delta, k1, k2, self.dim)
    cert = self.compute_set(make_noise(self.sigma0, self.dim_data) + delta, k1, k2, self.dim_data)
    return cert




if __name__ == '__main__':

  cert = Certificate()
  radius = cert.compute(0.25 + 0.2)
  print('cert {:.4f}'.format(radius))




