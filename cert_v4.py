

import numpy as np
import scipy as sp
import torch
from scipy.stats import norm as sp_norm
from scipy.special import gamma, loggamma

import vegas


class Analyze:
  
  def begin(self, itn, integrator):
    print(integrator.map())
  
  def end(self, itn_result, result):
    pass




class Certificate:

  def __init__(self, sigma0, p1, p2):

    self.sigma0 = sigma0
    self.pABar_sigma0 = p1
    self.pABar_sigma1 = p2
    self.dim = 5
    self.sample = 10000000
    self.niter = 25
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
    # return torch.randn(self.sample, dim, device=self.device) * sigma
    noise = np.random.normal(0, sigma, size=(self.sample, dim))
    noise = torch.FloatTensor(noise)
    if self.device == 'cuda':
      noise = noise.cuda()
    return noise

  def compute(self, sigma1):
    """ Binary search to find max certificate radius """
    self.sigma1 = sigma1
    start = round((self.cohen_cert - 0.02) * 100) / 100
    end = start + 0.4
    while (end - start) > self.binary_search_tol:
      # print('start {:.4f}, end {:.4f}'.format(start, end))
      eps = (start + end) / 2
      if self.compute_cert(eps) >= 1/2:
        start = eps
      else:
        end = eps
      break
    return start

  def X1_term(self, noise, eps):
    dim = noise.shape[-1]
    delta = self.define_delta(dim, eps)
    return -torch.norm(noise + delta, p=2, dim=1)**2 / (2 * self.sigma0**2)

  def Y1_term(self, noise, cst=None):
    sample = len(noise)
    if cst == None:
      cst = torch.zeros(sample, device=self.device)
    return -torch.norm(noise, p=2, dim=1)**2 / (2 * self.sigma0**2) - cst

  def Y2_term(self, noise, log_k2, cst=None):
    sample = len(noise)
    if cst == None:
      cst = torch.zeros(sample, device=self.device)
    return -torch.norm(noise, p=2, dim=1)**2 / (2 * self.sigma1**2) + log_k2 - cst

  def compute_set(self, noise, eps, k1, log_k2, return_mean=True):

    X1 = self.X1_term(noise, eps)
    Y1 = self.Y1_term(noise).reshape(-1, 1)
    Y2 = self.Y2_term(noise, log_k2).reshape(-1, 1)

    cst = torch.max(torch.cat([Y1, Y2], dim=1), dim=1)[0]
    Y1 = self.Y1_term(noise, cst)
    Y2 = self.Y2_term(noise, log_k2, cst)

    exp_Y1 = torch.exp(Y1)
    exp_Y2 = torch.exp(Y2)

    if return_mean:
      return torch.mean( ( X1 <= cst + torch.log( k1 * exp_Y1 + exp_Y2 ) ) * 1.)
    else:
      return X1 <= cst + torch.log( k1 * exp_Y1 + exp_Y2 )

  def find_k1_cohen_v1(self, eps):
    k1 = np.exp((eps / self.sigma0 * sp_norm.ppf(self.pABar_sigma0) ) - eps**2 / (2 * self.sigma0 **2))
    return k1

  def find_k1_cohen_v2(self, eps):
    noise = self.make_noise(self.sigma0, self.dim)
    X1 = self.X1_term(noise, self.dim, eps)
    Y1 = self.Y1_term(noise, self.dim)
    bound = X1 - Y1
    log_k1 = torch.quantile(bound, self.pABar_sigma0)
    return torch.exp(log_k1)

  def compute_log_k(self, sigma):
    pi = np.pi
    dim = self.dim
    # k = 1/sigma**(dim-2) * (1/2**((dim-2)/2)) * (1/gamma((dim-1)/2)) * np.sqrt(pi)
    # return np.log(k)

    # # k empirique
    # n = np.random.randn(10000000) * self.sigma0
    # n = np.abs(n)
    # K = np.mean(n**(self.dim-2))
    # print('log K', -np.log(K))

    log_k = -(dim-2) * np.log(sigma) - ((dim-2)/2) * np.log(2) - loggamma((dim-1)/2) + 1/2 * np.log(pi)
    print('log K', log_k)
    return log_k






  # def compute_set_ev(self, noise, eps, k1, log_k2, log_k):
  #
  #   sigma0, sigma1 = self.sigma0, self.sigma0
  #   # in_set = self.compute_set(noise, dim, eps, k1, log_k2, return_mean=False)
  #
  #   X1 = self.X1_term(noise, eps)
  #   Y1 = self.Y1_term(noise)
  #   in_set = torch.exp(X1) <= k1 * torch.exp(Y1)
  #
  #   r = noise[in_set][:, 0]
  #   norm1 = K_sigma0 * (sigma0**2/ sigma1**2)
  #   norm2 = (1/R) * (sigma0 / sigma1)
  #   p = self.dim - 2
  #   print('norm1', norm1)
  #   print('norm2', 1/norm2)
  #   print('int', torch.mean(torch.abs(r*norm2)**p))
  #   integral = norm1 * torch.mean(torch.abs(r*norm2)**p) / 2
  #   return integral

  
  # def compute_from_integral(self, noise, eps, k1, log_k2, log_k):
  #   sigma0, sigma1 = self.sigma0, self.sigma1
  #   mask = self.compute_set(noise, eps, k1, log_k2, return_mean=False)
  #   p = self.dim-2
  #   r = noise[mask][:, 0]
  #   interieur = torch.abs(r) * (sigma0 / sigma1) * np.exp(1/p * log_k)
  #   return torch.mean(interieur**p) / 2

  def compute_from_integral(self, noise, eps, k1, log_k2, log_k):
    # p = self.dim - 2
    dim = self.dim

    g = self.compute_set(noise, eps, k1, log_k2, return_mean=False)
    r = torch.abs(noise[:, 0]) * (g * 1.)
    int1  = (r * np.exp(1/(dim-2) * log_k) )**(dim-2)
    p1 = int1.mean().cpu().numpy()
    print('old p1', p1)
    
    # print('K', np.exp(1/(dim-2) * log_k)**(dim-2))
    # noise = self.make_noise(self.sigma0, 2)
    # g = self.compute_set(noise, eps, k1, log_k2, return_mean=False)
    # r = torch.abs(noise[:, 0]) * (g * 1.)
    # r_p = r**(dim-2)



    norm1 = 1.1
    norm2 = 0.75
    # norm1 = 1.
    # norm2 = 1.
    analyzer = Analyze()

    @vegas.batchintegrand
    def f1(x):
      x = torch.DoubleTensor(x).cuda()
      x[:, 0] = torch.abs(x[:, 0])
      i1 = 1/(np.pi*self.sigma0**2) * torch.exp(-torch.norm(x, p=2, dim=1)**2 / (2*self.sigma0**2))
      g = self.compute_set(x, eps, k1, log_k2, return_mean=False)
      r = x[:, 0] * norm1
      integral = i1 * (r * np.exp(1/(dim-2) * log_k) * (g * 1.))**(dim-2)
      integral = torch.nan_to_num(integral)
      return np.double(integral.detach().cpu().numpy())

    @vegas.batchintegrand
    def f2(x):
      x = torch.DoubleTensor(x).cuda()
      x[:, 0] = torch.abs(x[:, 0])
      i1 = 1/(np.pi*self.sigma0**2) * torch.exp(-torch.norm(x, p=2, dim=1)**2 / (2*self.sigma0**2))
      g = self.compute_set(x, eps, k1, log_k2, return_mean=False)
      r = x[:, 0] * norm2
      integral = i1 * (r * np.exp(1/(dim-2) * log_k))**(dim-2) * (g * 1.)
      return np.double(integral.detach().cpu().numpy())

    def compute_integral_from_support(f, x1, x2):
      integ = vegas.Integrator([x1, x2])
      integ(f, nitn=self.niter, neval=self.sample)
      # result = integ(f, nitn=self.niter, neval=self.sample, analyzer=analyzer)
      result = integ(f, nitn=self.niter, neval=self.sample)
      return result

    int1 = norm1**(2-dim) * compute_integral_from_support(f1, [0, 1], [-100, 100])
    print()
    int2 = (2-dim) * np.log(norm2) + np.log(compute_integral_from_support(f2, [1, 12], [-100, 100]))

    print('p1', (int1 + np.exp(int2)).mean)




    # norm = (2-dim) * np.log(normalization)
    #
    # r1 = compute_integral_from_support(-10, 0)
    # r2 = compute_integral_from_support(0, +10)
    # print(r1, r2)
    # log_r = np.log(r1 + r2)
    # res = norm + log_r
    # print('p1', np.exp(res.mean))
    #
    # r1 = compute_integral_from_support(-4, 5)
    # log_r1 = np.log(r1)
    # r1 = norm + log_r1
    # print('p1', np.exp(r1.mean))
    #
    # r1 = compute_integral_from_support(-5, 5)
    # log_r1 = np.log(r1)
    # r1 = norm + log_r1
    # print('p1', np.exp(r1.mean))




    return p1


  def find_k2(self, eps, k1, log_k):

    sigma0, sigma1 = self.sigma0, self.sigma0

    def binary_search(start, end):
      while (end - start) > 0.0001:
        log_k2 = (start + end) / 2
        noise = self.make_noise(self.sigma1, 2)
        p2 = self.compute_from_integral(noise, eps, k1, log_k2, log_k) * (sigma0**2 / sigma1**2)
        if p2 <= self.pABar_sigma1:
          start = log_k2
        else:
          end = log_k2
      return start

    search_space = [-200000, 200000]
    log_k2 = search_space[0]
    while True:
      log_k2 = binary_search(*search_space)
      # p2 = self.compute_set(self.make_noise(self.sigma1, self.dim), eps, k1, log_k2)
      noise = self.make_noise(self.sigma1, 2)
      p2 = self.compute_from_integral(noise, eps, k1, log_k2, log_k) * (sigma0**2 / sigma1**2)
      if log_k2 not in search_space:
        break
      k1 -= 0.0001
    return k1, log_k2


  def compute_cert(self, eps):
    self.eps = eps
    dim = self.dim

    make_noise = self.make_noise
    pABar_sigma0 = self.pABar_sigma0
    pABar_sigma1 = self.pABar_sigma1

    k1 = self.find_k1_cohen_v1(eps)
    # print('k1', k1)

    noise = make_noise(self.sigma0, 2)
    p1 = self.compute_set(noise, eps, k1, -1000000)
    print('p1', p1)

    log_k = self.compute_log_k(self.sigma0)
    # print('log_k =', log_k)

    noise = self.make_noise(self.sigma0, 2)

    p1 = self.compute_from_integral(noise, eps, k1, -1000000, log_k)
    # normalization = (self.sigma0**2 / self.sigma1**2)
    # normalization = 1

    # print('p1', p1)
    # print()

    # norm = self.pABar_sigma0 / (integral * normalization)
    # p1 = integral * normalization * norm
    # print('norm', norm) 
    # print('p1', p1)


   

    # p1 = self.compute_from_integral(noise, eps, k1, -10000, log_k) * (self.sigma0**2 / self.sigma1**2)
    # print('p1', p1)


    # k1, log_k2 = self.find_k2(eps, k1, log_k)
    # print('k1', k1)
    # print('log_k2', log_k2)

    # noise = make_noise(self.sigma0, 2)
    # p1 = self.compute_set(noise, 2, eps, k1_cohen, -10000, log_k)
    # print('p1', p1)

    # k1 = k1_cohen - 0.01
    # k1, log_k2 = self.find_k_values(dim, eps, k1, K_sigma0, R)
    # print('log_k2', log_k2)
    
    # p1 = self.compute_set(make_noise(self.sigma0, self.dim), eps, k1, log_k2)
    # p2 = self.compute_set(make_noise(self.sigma1, self.dim), eps, k1, log_k2)



    # noise = self.make_noise(self.sigma0, 2)
    # p1 = self.compute_from_integral(noise, eps, k1, log_k2, log_k) * (sigma0**2 / sigma1**2)
    # noise = self.make_noise(self.sigma1, 2)
    # p2 = self.compute_from_integral(noise, eps, k1, log_k2, log_k) * (sigma0**2 / sigma1**2)
    # print(f'p1 = {p1}')
    # print(f'p2 = {p2}')


    return 0.3


def main():

  sigma0 = 0.25
  sigma1 = 0.30
  p1 = 0.70
  p2 = 0.50

  cert = Certificate(sigma0, p1, p2)
  radius = cert.compute(sigma1)
  # print('cert {:.4f}'.format(radius))



if __name__ == '__main__':
  main()




