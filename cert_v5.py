

import numpy as np
import scipy as sp
import torch
from numpy import pi
from scipy.stats import norm as sp_norm
from scipy.special import gamma, loggamma


import vegas


class Analyze:
  
  def begin(self, itn, integrator):
    print(integrator.map())
  
  def end(self, itn_result, result):
    pass




class Certificate:

  def __init__(self, sigma0, sigma1, p1, p2):

    self.sigma0 = sigma0
    self.sigma1 = sigma1
    self.pABar_sigma0 = p1
    self.pABar_sigma1 = p2
    self.dim = 10
    self.sample = 100000
    self.niter = 12
    if torch.cuda.is_available():
      self.device = 'cuda'
    else:
      self.device = 'cpu'
    self.close_below_tol = 0.0001
    self.binary_search_tol = 0.0001

    self.cohen_cert = self.sigma0 * sp_norm.ppf(self.pABar_sigma0)
    print('cert cohen {:.4f}'.format(self.cohen_cert))

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

  def compute_k1_cohen(self, eps):
    """ Compute the constant k1 in the context of Cohen certificate """
    k1 = np.exp((eps / self.sigma0 * sp_norm.ppf(self.pABar_sigma0) ) - eps**2 / (2 * self.sigma0 **2))
    return k1

  def compute_log_k(self, sigma):
    """ Compute the constant k """ 
    dim = self.dim
    log_k = -(dim-2) * np.log(sigma) - ((dim-2)/2) * np.log(2) - loggamma((dim-1)/2) + 1/2 * np.log(pi)
    return log_k

  def integrand(self, x, eps, k1, log_k2, sigma=0.25):
    log_k = self.compute_log_k(sigma)
    x = torch.DoubleTensor(x)
    if self.device == 'cuda':
      x = x.cuda()
    x[:, 0] = torch.abs(x[:, 0])
    exp = 1/(pi*sigma**2) * torch.exp(-torch.norm(x, p=2, dim=1)**2 / (2*sigma**2))
    g = self.compute_set(x, eps, k1, log_k2, return_mean=False)
    r = x[:, 0]
    integral = exp * (r * np.exp(1/(self.dim-2) * log_k) * (g * 1.))**(self.dim-2)
    integral = torch.nan_to_num(integral)
    return np.double(integral.detach().cpu().numpy())

  def compute_from_integral(self, sigma, eps, k1, log_k2):

    dim = self.dim

    center = sigma * np.sqrt(self.dim - 2)
    t = np.array([center, 0]).reshape(1, 2)
    mode = self.integrand(t, eps, k1, log_k2, sigma=self.sigma0)
    print(f'mode of curve: {center:.4f}, {mode:.4f}')

    # a = 1/max_
    # b = -center
    # a = 1.
    # b = 0.
    # print(f'a = {a}, b = {b}')

    # bound = lambda r: a * r + b 
    # b1 = bound(center - 0.75)
    # b2 = bound(center + 0.75)
    # support = [b1, b2]
    support = [center - 0.75, center + 0.75]

    @vegas.batchintegrand
    def f1_vegas(x):
      return self.integrand(
        x, eps, k1, log_k2, sigma=sigma)

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
    int1 = compute_integral_from_support(f1_vegas, support, [-2, 2])
    # int1 = norm1**(2-dim) * compute_integral_from_support(f1_vegas, support, [-2, 2])
    # int1 = (2-dim) * np.log(norm1) + np.log(compute_integral_from_support(f1_vegas, support, [-2, 2]))
    # int1 = np.exp(int1)
    print(f'p1 = {int1.mean:.5f}')
  
    # print(f'dim: {self.dim}, support [2, 4]')
    # int1 = norm1**(2-dim) * compute_integral_from_support(f1_vegas, [2, 4], [-2, 2])
    # print(f'p1 = {int1.mean:.5f}')

    return None


  def find_mode_and_width_of_bell_curve(self, sigma, eps, k1, log_k2):

    print()
    points = 5000

    # for i, dim in enumerate(np.arange(5, 60)):
    for i, dim in enumerate([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]):

      center = sigma * np.sqrt(dim - 2)
      t = np.array([center, 0]).reshape(1, 2)
      mode = self.integrand(t, eps, k1, log_k2, sigma=self.sigma0)

      # r = np.linspace(np.maximum(center - 2, 0), center + 2, num=points)
      # u = np.zeros(points)
      # x = np.hstack((r.reshape(-1, 1), u.reshape(-1, 1))).reshape(points, 2)
      # values = self.integrand(x, eps, k1, log_k2, sigma=sigma)

      # min_support = np.maximum(r[values > bound].min(), 0)
      # max_support = r[values > bound].max()
      # width_support = max_support - min_support

      @vegas.batchintegrand
      def f1_vegas(x):
        return self.integrand(
          x, eps, k1, log_k2, sigma=sigma)

      def compute_integral_from_support(f, x1, x2):
        integ = vegas.Integrator([x1, x2])
        integ(f, nitn=self.niter, neval=self.sample)
        result = integ(f, nitn=self.niter, neval=self.sample)
        return result

      support = [np.maximum(center - 0.75, 0), center + 0.75]

      # we increase the support if necessary 
      p1 = 0.
      while p1 < 0.69:
        p1 = compute_integral_from_support(f1_vegas, support, [-2, 2])
        support[0] -= 0.01
        support[1] += 0.01
 
      # print('support big enough')
 
      while True:
        # print(support, p1)
        p1 = compute_integral_from_support(f1_vegas, support, [-2, 2])
        if p1 < 0.698:
          support[0] -= 0.01
          support[1] += 0.01
          p1 = compute_integral_from_support(f1_vegas, support, [-2, 2])
          break
        support[0] += 0.01
        support[1] -= 0.01

      width_support = support[1] - support[0]
      print(
        (f"dim: {dim:3d}, mode: {center:.4f}, {mode:.4f}, "
         f"support: [{support[0]:.4f}, {support[1]:.4f}], {width_support:.3f}, "
         f"p1: {p1.mean:.4f}"))



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
    sigma1 = self.sigma1

    make_noise = self.make_noise
    pABar_sigma0 = self.pABar_sigma0
    pABar_sigma1 = self.pABar_sigma1

    k1 = self.compute_k1_cohen(eps)
    log_k = self.compute_log_k(self.sigma0)
    log_k2 = -1000000

    print(f'k1 = {k1:.4f}')
    print(f'log_k = {log_k:.4f}')
    print()

    noise = make_noise(self.sigma0, 2)
    p1 = self.compute_set(noise, eps, k1, log_k2)
    print(f'p1 (with k1 from compute set): {p1.cpu().numpy():.4f}')

    noise = self.make_noise(self.sigma0, 2)
    p1 = self.compute_from_integral(sigma0, eps, k1, log_k2)
    # print(f'p1 (with k1 from integral): {p1:.4f}')
    # print()

    self.find_mode_and_width_of_bell_curve(sigma0, eps, k1, log_k2)


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

  def plot_3d_integrand(self):

    fig = plt.figure(figsize=(32, 15))
    
    n_points = 200
    dpi = 100
    format = 'pdf'
    
    eps = 0.25
    k1 = self.compute_k1_cohen(eps)
    log_k = self.compute_log_k(self.sigma0)
    log_k2 = -100000
    
    x0 = np.linspace(  0, 10, n_points)
    x1 = np.linspace(-10, 10, n_points)
    grid = np.meshgrid(x0, x1)
    X, Y = grid
    x = np.hstack((grid[0].flatten().reshape(-1, 1), grid[1].flatten().reshape(-1, 1)))
    
    for i, dim in enumerate([20, 100, 120, 130, 150, 200, 300, 350]):
      self.dim = dim
      Z = self.f(x, eps, k1, log_k2, log_k).reshape(n_points, n_points)
      ax = fig.add_subplot(2, 4, i+1, projection='3d')
      surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
      ax.title.set_text('Dimension {}'.format(dim))
    plt.show()
    # fig.savefig('./integral_avec_log_k2_-5.{}'.format(format), dpi=dpi, format=format)





def main():

  sigma0 = 0.25
  sigma1 = 0.30
  p1 = 0.70
  p2 = 0.60

  cert = Certificate(sigma0, sigma1, p1, p2)
  radius = cert.compute_cert(0.25)
  
  # print('cert {:.4f}'.format(radius))



if __name__ == '__main__':
  main()




