

import numpy as np
import scipy as sp
import torch
from numpy import pi
from scipy.stats import norm as sp_norm
from scipy.special import gamma, loggamma

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import vegas


class Analyze:
  
  def begin(self, itn, integrator):
    print(integrator.map())
  
  def end(self, itn_result, result):
    pass




class Certificate:

  def __init__(self, sigma0, sigma1, p1, p2):

    # self.sigma0 = sigma0
    # self.sigma1 = sigma1
    # self.dim = 10
    self.pABar_sigma0 = p1
    self.pABar_sigma1 = p2
    self.sample = 100000
    self.niter = 12
    if torch.cuda.is_available():
      self.device = 'cuda'
    else:
      self.device = 'cpu'
    self.close_below_tol = 0.0001
    self.binary_search_tol = 0.0001

    # self.cohen_cert = self.sigma0 * sp_norm.ppf(self.pABar_sigma0)
    # print('cert cohen {:.4f}'.format(self.cohen_cert))

  def define_delta(self, dim, eps):
    delta = torch.zeros(dim, device=self.device)
    delta[-1] = eps
    return delta

  def make_noise(self, sigma, dim):
    return torch.randn(self.sample, dim, device=self.device) * sigma

  def compute(self):
    """ Binary search to find the maximum certificate radius """
    # we define the search space between [ cohen -0.02, cohen + 0.4 ]
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

  def X1_term(self, noise, eps, sigma0):
    dim = noise.shape[-1]
    delta = self.define_delta(dim, eps)
    return -torch.norm(noise + delta, p=2, dim=1)**2 / (2 * sigma0**2)

  def Y1_term(self, noise, sigma0, constant=None):
    sample = len(noise)
    if constant == None:
      constant = torch.zeros(sample, device=self.device)
    return -torch.norm(noise, p=2, dim=1)**2 / (2 * sigma0**2) - constant

  def Y2_term(self, noise, sigma1, log_k2, constant=None):
    sample = len(noise)
    if constant == None:
      constant = torch.zeros(sample, device=self.device)
    return -torch.norm(noise, p=2, dim=1)**2 / (2 * sigma1**2) + log_k2 - constant

  def compute_set(self, noise, eps, k1, log_k2, sigma0, sigma1, return_mean=True):

    X1 = self.X1_term(noise, eps, sigma0)
    Y1 = self.Y1_term(noise, sigma0).reshape(-1, 1)
    Y2 = self.Y2_term(noise, sigma1, log_k2).reshape(-1, 1)

    cst = torch.max(torch.cat([Y1, Y2], dim=1), dim=1)[0]
    Y1 = self.Y1_term(noise, sigma0, constant=cst)
    Y2 = self.Y2_term(noise, sigma1, log_k2, constant=cst)

    exp_Y1 = torch.exp(Y1)
    exp_Y2 = torch.exp(Y2)

    if return_mean:
      return torch.mean( ( X1 <= cst + torch.log( k1 * exp_Y1 + exp_Y2 ) ) * 1.)
    else:
      return X1 <= cst + torch.log( k1 * exp_Y1 + exp_Y2 )

  def compute_k1_cohen(self, eps, sigma0):
    """ Compute the constant k1 in the context of Cohen certificate """
    k1 = np.exp((eps / sigma0 * sp_norm.ppf(self.pABar_sigma0) ) - eps**2 / (2 * sigma0 **2))
    return k1

  def compute_log_k(self, sigma, dim):
    """ Compute the constant k """ 
    log_k = -(dim-2) * np.log(sigma) - ((dim-2)/2) * np.log(2) - loggamma((dim-1)/2) + 1/2 * np.log(pi)
    return log_k

  def integrand(self, x, dim, sigma0, sigma1, eps, k1, log_k2):
    log_k = self.compute_log_k(sigma0, dim)
    x = torch.DoubleTensor(x)
    if self.device == 'cuda':
      x = x.cuda()
    x[:, 0] = torch.abs(x[:, 0])
    exp = 1/(pi*sigma0**2) * torch.exp(-torch.norm(x, p=2, dim=1)**2 / (2*sigma0**2))
    g = self.compute_set(x, eps, k1, log_k2, sigma0, sigma1, return_mean=False)
    r = x[:, 0]
    integral = exp * (r * np.exp(1/(dim-2) * log_k) * (g * 1.))**(dim-2)
    integral = torch.nan_to_num(integral)
    return np.double(integral.detach().cpu().numpy())

  def compute_integral(self, dim, sigma0, sigma1, eps, k1, log_k2, support_r):

    @vegas.batchintegrand
    def integrand_vegas(x):
      return self.integrand(x, dim, sigma0, sigma1, eps, k1, log_k2)

    integ = vegas.Integrator([support_r, [-2, 2]])
    integ(integrand_vegas, nitn=self.niter, neval=self.sample)
    result = integ(integrand_vegas, nitn=self.niter, neval=self.sample)
    return result
  


  def find_mode_and_width_of_bell_curve(self, eps, k1, log_k2, sigma0, sigma1):

    print()
    points = 5000

    for i, dim in enumerate([5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]):

      center = sigma0 * np.sqrt(dim - 2)
      x = np.array([center, 0]).reshape(1, 2)
      support = [center - 0.75, center + 0.75]
      mode = self.integrand(x, dim, sigma0, sigma1, eps, k1, log_k2)

      support = [np.maximum(center - 0.55, 0), center + 0.55]
      width_support = support[1] - support[0]
      p1 = self.compute_integral(dim, sigma0, sigma1, eps, k1, log_k2, support)

      print(
        (f"dim: {dim:3d}, mode: {center:.4f}, {mode:.4f}, "
         f"support: [{support[0]:.4f}, {support[1]:.4f}], {width_support:.3f}, "
         f"p1: {p1.mean:.4f}"))



  def find_k2(self, eps, k1, log_k):

    # sigma0, sigma1 = sigma0, self.sigma0
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
    # sigma0 = self.sigma0
    # sigma1 = self.sigma1
    sigma0 = 0.25
    sigma1 = 0.30

    make_noise = self.make_noise
    pABar_sigma0 = self.pABar_sigma0
    pABar_sigma1 = self.pABar_sigma1

    k1 = self.compute_k1_cohen(eps, sigma0)
    # log_k = self.compute_log_k(self.sigma0, dim)
    log_k2 = -10000

    print(f'k1 = {k1:.4f}')
    # print(f'log_k = {log_k:.4f}')
    print()

    noise = make_noise(sigma0, 2)
    p1 = self.compute_set(noise, eps, k1, log_k2, sigma0, sigma1)
    print(f'p1 (with k1 from compute set): {p1.cpu().numpy():.4f}')

    # noise = self.make_noise(sigma0, 2)
    # p1 = self.compute_from_integral(sigma0, eps, k1, log_k2)
    # print(f'p1 (with k1 from integral): {p1:.4f}')
    # print()

    self.find_mode_and_width_of_bell_curve(eps, k1, log_k2, sigma0, sigma1)


    # log_k = self.compute_log_k(sigma0)
    # noise = self.make_noise(sigma0, 2)
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

    # noise = make_noise(sigma0, 2)
    # p1 = self.compute_set(noise, 2, eps, k1_cohen, -10000, log_k)
    # print('p1', p1)

    # k1 = k1_cohen - 0.01
    # k1, log_k2 = self.find_k_values(dim, eps, k1, K_sigma0, R)
    # print('log_k2', log_k2)
    
    # p1 = self.compute_set(make_noise(sigma0, dim), eps, k1, log_k2)
    # p2 = self.compute_set(make_noise(self.sigma1, dim), eps, k1, log_k2)

    # noise = self.make_noise(sigma0, 2)
    # p1 = self.compute_from_integral(noise, eps, k1, log_k2, log_k) * (sigma0**2 / sigma1**2)
    # noise = self.make_noise(self.sigma1, 2)
    # p2 = self.compute_from_integral(noise, eps, k1, log_k2, log_k) * (sigma0**2 / sigma1**2)
    # print(f'p1 = {p1}')
    # print(f'p2 = {p2}')

    return 0.3

  def plot_3d_integrand(self):

    fig = plt.figure(figsize=(10, 8))

    eps = 0.25
    sigma0 = 0.25
    sigma1 = 0.30
    
    n_points = 200
    dpi = 100
    format = 'pdf'
    
    eps = 0.25
    k1 = self.compute_k1_cohen(eps, sigma0)
    log_k2 = -100000
    
    x0 = np.linspace( 0, 10, n_points)
    x1 = np.linspace(-3, 3, n_points)
    grid = np.meshgrid(x0, x1)
    X, Y = grid
    x = np.hstack((grid[0].flatten().reshape(-1, 1), grid[1].flatten().reshape(-1, 1)))
    
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax = fig.add_subplot(1, 1, 1)
    
    for i, dim in enumerate([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]):
      # x, dim, sigma0, sigma1, eps, k1, log_k2)
      Z = self.integrand(x, dim, sigma0, sigma1, eps, k1, log_k2).reshape(n_points, n_points)
      Z[Z < 1e-4] = float('nan')
      Z = Z.reshape(n_points, n_points)

      # surf = ax.plot_surface(X, Y, Z, 
      #   rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=1, antialiased=False)

      # ax = fig.add_subplot(2, 4, i+1)
      ax.contourf(X, Y, Z, levels=50)
      # ax.title.set_text('Dimension {}'.format(dim))

    
    plt.tight_layout()
    plt.show()
    # fig.savefig('./integral_avec_log_k2_-5.{}'.format(format), dpi=dpi, format=format)





def main():

  sigma0 = 0.25
  sigma1 = 0.30
  p1 = 0.70
  p2 = 0.60

  cert = Certificate(sigma0, sigma1, p1, p2)
  # radius = cert.compute_cert(0.25)
  cert.plot_3d_integrand()
  
  # print('cert {:.4f}'.format(radius))



if __name__ == '__main__':
  main()




