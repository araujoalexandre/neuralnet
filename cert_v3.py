
import numpy as np
import torch
from scipy.stats import norm as sp_norm


class Certificate:

  def __init__(self):

    self.sigma0 = 0.5
    self.pABar_sigma0 = 0.90
    self.pABar_sigma1 = 0.98
    self.dim = 100
    self.sample = 1000000
    self.device = 'cuda'
    self.close_below_tol = 0.03
    self.binary_search_tol = 0.0001

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

  def log_left_term(self, noise, dim):
    delta = self.define_delta(dim, self.eps)
    term = -torch.norm(noise - delta, p=2, dim=1)**2 / (2 * self.sigma0**2)
    return term

  def k1_term(self, noise):
    term = torch.exp(-torch.norm(noise, p=2, dim=1)**2 / (2 * self.sigma0**2))
    return term

  def k2_term(self, noise):
    term = torch.exp(-torch.norm(noise, p=2, dim=1)**2 / (2 * self.sigma1**2))
    return term

  def Y1_term(self, noise, k1, cst):
    Y1 = -torch.norm(noise, p=2, dim=1)**2 / (2 * self.sigma0**2) - cst
    return Y1

  def Y2_term(self, noise, log_k2, cst):
    log_k2 = torch.FloatTensor([log_k2]).to(self.device)
    Y2 = -torch.norm(noise, p=2, dim=1)**2 / (2 * self.sigma1**2) + log_k2 - cst
    return Y2

  def compute_set(self, noise, k1, k2, dim):
    k1 = torch.FloatTensor([k1]).to(self.device)
    k2 = torch.FloatTensor([k2]).to(self.device)
    return torch.mean(
      (self.left_term(noise, dim) <= k1 * self.k1_term(noise) + k2 * self.k2_term(noise)) * 1.
    )

  def compute_set_with_log(self, noise, k1, log_k2, dim):
 
    cst = torch.zeros(self.sample, device=self.device)
    Y1 = (self.Y1_term(noise, k1, cst)).reshape(-1, 1)
    Y2 = (self.Y2_term(noise, log_k2, cst)).reshape(-1, 1)

    cst = torch.max(torch.cat([Y1, Y2], dim=1), dim=1)[0]
    Y1 = self.Y1_term(noise, k1, cst)
    Y2 = self.Y2_term(noise, log_k2, cst)

    log_left_term = self.log_left_term(noise, dim)
    exp_Y1 = k1 * torch.exp(Y1)
    exp_Y2 = torch.exp(Y2)

    return torch.mean(
      ( log_left_term <= cst + torch.log( exp_Y1 + exp_Y2) ) * 1.
    )

  # def find_log_k1(self, dim):
  #   norm = lambda x: torch.norm(x, p=2, dim=1)
  #   noise = self.make_noise(self.sigma0, dim)
  #   delta = self.define_delta(dim, self.eps)
  #   bound = -1 / (2 * self.sigma0**2) * (norm(noise - delta)**2 + norm(noise)**2)
  #   return  torch.quantile(bound, self.pABar_sigma0)

  # def find_k1(self, k2, dim):
  #   noise = self.make_noise(self.sigma0, dim)
  #   bound = (self.left_term(noise, dim) - k2 * self.k2_term(noise)) / self.k1_term(noise)
  #   k1 = torch.quantile(bound, self.pABar_sigma0)
  #   return k1
  #
  # def find_k2(self, k1, dim):
  #   noise = self.make_noise(self.sigma1, dim)
  #   bound = (self.left_term(noise, dim) - k1 * self.k1_term(noise)) / self.k2_term(noise)
  #   k2 = torch.quantile(bound, self.pABar_sigma1)
  #   return k2

  def compute(self, sigma1):
    """ Binary search to find max certificate radius """
    self.sigma1 = sigma1
    start = round((self.cohen_cert - 0.02) * 100) / 100
    end = start + 0.4
    while (end - start) > self.binary_search_tol:
      print('start {:.4f}, end {:.4f}'.format(start, end))
      eps = (start + end) / 2
      # print(f'eps = {eps}')
      if self.compute_cert(eps) >= 1/2:
        start = eps
      else:
        end = eps
      # break
    return start


  def find_k1_binary_search(self, log_k2, dim):

    k2 = torch.exp(log_k2).to(self.device)

    def binary_search(start, end):
      while (end - start) > 0.001:
        k1 = (start + end) / 2
        p1 = self.compute_set(
          self.make_noise(self.sigma0, dim), k1, k2, dim)
        if p1 <= self.pABar_sigma0:
          start = k1
        else:
          end = k1
      return start

    search_space = np.array([-10, 10])
    k1 = search_space[0]
    while True:
      k1 = binary_search(*search_space)
      if k1 not in search_space:
        break
      search_space *= 10
    return k1


  def find_k2_binary_search(self, k1, dim):

    def binary_search(start, end):
      # print('Search log_k2 with [{}, {}]'.format(start, end))
      while (end - start) > 0.01:
        log_k2 = (start + end) / 2
        p2 = self.compute_set_with_log(
          self.make_noise(self.sigma1, dim), k1, log_k2, dim)
        # print('log_k2 / p2 / k1', log_k2, p2, k1)
        if p2 <= self.pABar_sigma1:
          start = log_k2
        else:
          end = log_k2
      return start

    search_space = np.array([-100000, 100000])
    log_k2 = search_space[0]
    itertation = 0
    while True:
      itertation += 1
      if itertation == 100:
        print('10k iterations find_k2_binary_search')
      log_k2 = binary_search(*search_space)
      if log_k2 not in search_space:
        break
      k1 -= 0.01
      # print('k1', k1)
    return log_k2, k1


  def compute_cert(self, eps):
    self.eps = eps
    log_k2 = torch.FloatTensor([-1000])
    p1, p2 = 2., 2.
    dim = self.dim

    make_noise = self.make_noise
    pABar_sigma0 = self.pABar_sigma0
    pABar_sigma1 = self.pABar_sigma1
    is_close_below = lambda x, y: y - self.close_below_tol <= x <= y


    # print('#############')
    # print('p1 / p2 without k2')
    k1 = self.find_k1_binary_search(log_k2, dim)
    # p1 = self.compute_set_with_log(make_noise(self.sigma0, dim), k1, log_k2, dim)
    # p2 = self.compute_set_with_log(make_noise(self.sigma1, dim), k1, log_k2, dim)
    # print('k1', k1)
    # print('k2', log_k2)
    # print(f'p1 = {p1}')
    # print(f'p2 = {p2}')
    # print('#############')
    # print()


    iteration = 0
    while not is_close_below(p1, pABar_sigma0) or not is_close_below(p2, pABar_sigma1):
      iteration += 1
      if iteration == 100:
        print('100 iterations, adjust k1/k2')
      # k1 = self.find_k1_binary_search(log_k2, dim)
      # p1 = self.compute_set_with_log(make_noise(self.sigma0, dim), k1, log_k2, dim)
      # p2 = self.compute_set_with_log(make_noise(self.sigma1, dim), k1, log_k2, dim)
      # print(f'p1 = {p1}')
      # print(f'p2 = {p2}')
      log_k2, k1 = self.find_k2_binary_search(k1, dim)
      log_k2 = torch.FloatTensor([log_k2]).to(self.device)
      p1 = self.compute_set_with_log(make_noise(self.sigma0, dim), k1, log_k2, dim)
      p2 = self.compute_set_with_log(make_noise(self.sigma1, dim), k1, log_k2, dim)
      # print('while loop')
      # print(f'k1 {k1}')
      # print(f'log_k2 {log_k2}')
      # print(f'p1 = {p1}')
      # print(f'p2 = {p2}')
      # print()
    # print('exit')

    # print('k1', k1)
    # print('k2', log_k2)
    # for _ in range(30):
    #   p1 = self.compute_set_with_log(make_noise(self.sigma0, dim), k1, log_k2, dim)
    #   p2 = self.compute_set_with_log(make_noise(self.sigma1, dim), k1, log_k2, dim)
    #   # print(f'p1 = {p1}')
    #   print(f'p2 = {p2}')

    
    delta = self.define_delta(self.dim, eps)
    # cert = self.compute_set(
    #   make_noise(self.sigma0, self.dim) + delta, k1, torch.exp(log_k2), self.dim)
    cert = self.compute_set_with_log(
      make_noise(self.sigma0, self.dim) + delta, k1, log_k2, self.dim)
    return cert




if __name__ == '__main__':

  cert = Certificate()
  radius = cert.compute(0.5 + 0.5)
  print('cert {:.4f}'.format(radius))




