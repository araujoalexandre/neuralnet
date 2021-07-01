

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
    self.dim = 1400
    self.sample = 10000000
    self.niter = 12
    self.device = 'cuda'
    self.close_below_tol = 0.0001
    self.binary_search_tol = 0.0001

    # print(f'dimension: {self.dim}')

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
    log_k = -(dim-2) * np.log(sigma) - ((dim-2)/2) * np.log(2) - loggamma((dim-1)/2) + 1/2 * np.log(pi)
    return log_k

  def integrand(self, x, k1, log_k2, sigma=0.25, a=1., b=0.):
    log_k = self.compute_log_k(sigma)
    x = torch.DoubleTensor(x).cuda()
    x[:, 0] = ( torch.abs(x[:, 0]) - b ) / a
    exp = torch.exp(-torch.norm(x, p=2, dim=1)**2 / (2*sigma**2))
    g = self.compute_set(x, self.eps, k1, log_k2, return_mean=False)
    r = x[:, 0]
    integral = exp * (r * np.exp(1/(self.dim-2) * log_k) * (g * 1.))**(self.dim-2)
    integral = torch.nan_to_num(integral)
    return np.double(integral.detach().cpu().numpy())


  def compute_from_integral(self, sigma, eps, k1, log_k2):

    dim = self.dim
    pi = np.pi

    center = ( np.sqrt((self.dim - 2)) / 4 )
    t = np.array([center, 0]).reshape(1, 2)
    max_ = self.integrand(t, k1, log_k2, sigma=self.sigma0, a=1., b=0.)
    self.max_ = max_
    print('max', max_)

    a = 1/max_
    b = -center
    print(f'a = {a}, b = {b}')

    bound = lambda r: a * r + b 
    b1 = bound(center - 0.75)
    b2 = bound(center + 0.75)
    support = [b1, b2]

    @vegas.batchintegrand
    def f1_vegas(x):
      return self.integrand(
        x, k1, log_k2, sigma=self.sigma0, a=a, b=b)


    def compute_integral_from_support(f, x1, x2):
      integ = vegas.Integrator([x1, x2])
      integ(f, nitn=self.niter, neval=self.sample)
      result = integ(f, nitn=self.niter, neval=self.sample)
      return result

    # int1 = norm1**(2-dim) * compute_integral_from_support(f1, [0, 1], [-100, 100])
    # int2 = (2-dim) * np.log(norm2) + np.log(compute_integral_from_support(f2, [1, 12], [-100, 100]))
    # return (int1 + np.exp(int2)).mean

    # int1 = norm1**(2-dim) * compute_integral_from_support(f1, [0, 10], [-100, 100])

    # print(norm1**(2-dim))

    print(f'dim: {self.dim}, support {support}')
    int1 = 1/a * 1/(np.pi*sigma**2) * compute_integral_from_support(f1_vegas, support, [-2, 2])
    # int1 = norm1**(2-dim) * compute_integral_from_support(f1_vegas, support, [-2, 2])
    # int1 = (2-dim) * np.log(norm1) + np.log(compute_integral_from_support(f1_vegas, support, [-2, 2]))
    # int1 = np.exp(int1)
    print(f'p1 = {int1.mean:.5f}')
  
    # print(f'dim: {self.dim}, support [2, 4]')
    # int1 = norm1**(2-dim) * compute_integral_from_support(f1_vegas, [2, 4], [-2, 2])
    # print(f'p1 = {int1.mean:.5f}')

    return None


  def find_k2(self, eps, k1, log_k):

    sigma0, sigma1 = self.sigma0, self.sigma0
    is_close_below = lambda x, y: y - self.close_below_tol <= x <= y


    print('-- start loop --')
    self.sample = 100000

    k1 -= 0.01
    log_k2 = -10000
    p2 = 0.
    while True:
      p2 = self.compute_from_integral(sigma1, eps, k1, log_k2) * (sigma0**2 / sigma1**2)
      print(f'k2 = {np.exp(log_k2):.4f}, p2 = {p2:.4f}')
      log_k2 += 500
      # k1 -= 0.01

      if is_close_below(p2, self.pABar_sigma1):
        break

    # def binary_search(start, end):
    #   while (end - start) > 0.001:
    #     log_k2 = (start + end) / 2
    #     p2 = self.compute_from_integral(sigma1, eps, k1, log_k2) * (sigma0**2 / sigma1**2)
    #     print(f'{start:.5f} / {end:.5f}, {p2:.4f}')
    #     if p2 <= self.pABar_sigma1:
    #       start = log_k2
    #     else:
    #       end = log_k2
    #   return start

    # search_space = [-2000, 2000]
    # log_k2 = search_space[0]
    # while True:
    #   k1 -= 0.1
    #   log_k2 = binary_search(*search_space)
    #   p2 = self.compute_from_integral(sigma1, eps, k1, log_k2) * (sigma0**2 / sigma1**2)
    #   print('p2', p2)
    #   if log_k2 not in search_space:
    #     break
    return k1, log_k2


  def compute_cert(self, eps):
    self.eps = eps
    dim = self.dim
    sigma0 = self.sigma0

    make_noise = self.make_noise
    pABar_sigma0 = self.pABar_sigma0
    pABar_sigma1 = self.pABar_sigma1

    k1 = self.find_k1_cohen_v1(eps)
    log_k2 = -1000000

    noise = make_noise(self.sigma0, 2)
    p1 = self.compute_set(noise, eps, k1, log_k2)
    print('p1', p1)

    log_k = self.compute_log_k(self.sigma0)
    print('log_k =', log_k)

    noise = self.make_noise(self.sigma0, 2)
    p1 = self.compute_from_integral(sigma0, eps, k1, log_k2)
    # print('p1', p1)
    # print()


    # log_k = self.compute_log_k(self.sigma0)
    # noise = self.make_noise(self.sigma0, 2)
    # g = self.compute_set(noise, eps, k1, log_k2, return_mean=False)
    # r = torch.abs(noise[:, 0]) * (g * 1.)
    # int1  = (r * np.exp(1/(dim-2) * log_k) )**(dim-2)
    # p1 = int1.mean().cpu().numpy()
    # print('old p1', p1)
    #
    #
    # log_k = self.compute_log_k(self.sigma1)
    # noise = self.make_noise(self.sigma1, 2)
    # g = self.compute_set(noise, eps, k1, log_k2, return_mean=False)
    # r = torch.abs(noise[:, 0]) * (g * 1.)
    # int1  = (r * np.exp(1/(dim-2) * log_k) )**(dim-2)
    # p2 = int1.mean().cpu().numpy()
    # print('old p2', p2)
    #
    # print('k1', k1)
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
  p2 = 0.60

  cert = Certificate(sigma0, p1, p2)
  radius = cert.compute(sigma1)
  # print('cert {:.4f}'.format(radius))



if __name__ == '__main__':
  main()




