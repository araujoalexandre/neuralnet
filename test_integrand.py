

import numpy as np
import scipy as sp
from numpy import pi
from scipy.special import gamma

import vegas

norm = lambda x: np.linalg.norm(x, ord=2, axis=1)


def define_delta(dim, eps):
  delta = np.zeros(dim)
  delta[-1] = eps
  return delta

def make_noise(sigma, dim):
  return np.random.randn(sample, dim) * sigma

def compute_set(noise, eps, k1, k2, sigma0, sigma1):
  dim = noise.shape[-1]
  delta = define_delta(dim, eps)
  left_term = np.exp(-norm(noise - delta)**2 / (2 * sigma0**2))
  k1_term = np.exp(-norm(noise)**2 / (2 * sigma0**2))
  k2_term = np.exp(-norm(noise)**2 / (2 * sigma1**2))
  return (left_term <= k1 * k1_term + k2 * k2_term) * 1.

def integrand(x, dim, sigma0, sigma1):
  eps = 0.25
  r, u = x[:, 0], x[:, 1]
  g = compute_set(x, eps, 2., 0.5, sigma0, sigma1)
  K = np.sqrt(pi) * sigma0**(2-dim) * 2**((2-dim)/2) * gamma((dim-1)/2)**(-1)
  integral = K/(pi*sigma0**2) * np.exp(-(r**2+u**2)/(2*sigma0**2)) * r**(dim-2) * g
  return np.double(integral)

def compute_integral(dim, sigma0, sigma1):
  @vegas.batchintegrand
  def integrand_vegas(x):
    return integrand(x, dim, sigma0, sigma1)
  integ = vegas.Integrator([[0, 10], [-3, 3]])
  integ(integrand_vegas, nitn=10, neval=10000000)
  result = integ(integrand_vegas, nitn=10, neval=10000000)
  return result.mean




def main():
  print(compute_integral( 3, 0.25, 0.30))
  print(compute_integral( 4, 0.25, 0.30))
  print(compute_integral( 5, 0.25, 0.30))
  print(compute_integral( 6, 0.25, 0.30))
  print(compute_integral(10, 0.25, 0.30))
  



if __name__ == '__main__':
  main()
