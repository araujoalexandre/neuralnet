
import logging
from math import ceil

from utils import Noise

import numpy as np
from scipy.stats import norm, beta, binom_test
from statsmodels.stats.proportion import proportion_confint
import torch
import torch.nn.functional as F


# Code from https://github.com/Hadisalman/smoothing-adversarial/blob/master/code/core.py

class Smooth(object):
  """A smoothed classifier g """

  # to abstain, Smooth returns this int
  ABSTAIN = -1

  def __init__(self, model, params, n_classes, dim):
    """
    :param model: maps from [batch x channel x height x width] to [batch x n_classes]
    :param n_classes:
    :param sigma: the noise level hyperparameter
    """
    self.model = model
    self.n_classes = n_classes
    self.params = params
    self.dim = dim
    self.noise_distribution = self.params.noise_distribution
    self.noise_scale = self.params.noise_scale
    self.noise = Noise(self.params)

    # param batch_size: batch size to use when evaluating the base classifier
    self.batch_size_sample = self.params.certificate['batch_size']
    # param n0: the number of Monte Carlo samples to use for selection
    self.sample_n0 = self.params.certificate['N0']
    # param n: the number of Monte Carlo samples to use for estimation
    self.sample_n = self.params.certificate['N']
    # param alpha: the failure probability
    self.alpha = self.params.certificate['alpha']

    if self.noise_distribution == 'uniform':
      self.beta_dist = beta(0.5 * (self.dim + 1), 0.5 * (self.dim + 1))
      self.binary_search_lower = self.params.certificate['binary_search_lower']
      self.binary_search_upper = self.params.certificate['binary_search_upper']
      self.binary_search_tol = self.params.certificate['binary_search_tol']

  def eval(self):
    self.model.eval()

  def load_state_dict(self, ckpt):
    self.model.load_state_dict(ckpt)

  def __call__(self, inputs):
    """ Monte Carlo algorithm for evaluating the prediction of g at x.
    :param inputs: the input [batch_size x channel x height x width]
    :return: the predicted class
    """
    batch_size, *img_size = inputs.shape
    x = inputs.repeat((self.sample_n0, 1, 1, 1, 1))
    x = x.reshape(self.sample_n0 * batch_size, *img_size)
    noise = self.noise(x) * self.noise_scale
    predictions = self.model(x + noise).argmax(axis=1)
    hard_preds = F.one_hot(predictions,  num_classes=self.n_classes)
    hard_preds = hard_preds.reshape(self.sample_n0, batch_size, self.n_classes).float()
    proba = hard_preds.mean(axis=0)
    return proba

  def certify(self, *args, **kwargs):
    if self.noise_distribution == 'normal':
      return self._certify_gaussian(*args, **kwargs)
    elif self.noise_distribution == 'uniform':
      return self._certify_uniform(*args, **kwargs)

  def _lower_confidence_bound(self, NA, N, alpha):
    """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
    This function uses the Clopper-Pearson method.
    :param NA: the number of "successes"
    :param N: the number of total draws
    :param alpha: the confidence level
    :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
    """
    return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

  def get_proba_lb(self, x):
    # draw samples of f(x+ epsilon)
    counts_selection = self._sample_noise(x, self.sample_n0, self.noise_scale)
    # use these samples to take a guess at the top class
    cAHat = counts_selection.argmax().item()
    # draw more samples of f(x + epsilon)
    counts_estimation = self._sample_noise(
      x, self.sample_n, self.noise_scale)
    # use these samples to estimate a lower bound on pA
    nA = counts_estimation[cAHat].item()
    pABar = self._lower_confidence_bound(nA, self.sample_n, self.alpha)
    proba = counts_estimation / self.sample_n
    return cAHat, pABar, proba

  def _certify_gaussian(self, x):
    """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
    With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
    robust within a L2 ball of radius R around x.
    :param x: the input [channel x height x width]
    :return: (predicted class, certified radius)
         in the case of abstention, the class will be ABSTAIN and the radius 0.
    """
    cAHat, pABar, proba = self.get_proba_lb(x)
    if pABar < 0.5:
      return Smooth.ABSTAIN, (0.0, 0.0), proba
    else:
      radius = self.noise_scale * norm.ppf(pABar)
      return cAHat, (radius, 0.), proba
  
  def _certify_uniform(self, x):
    """ Algorithm for certifying with Uniform noise form L2 ball"""
    cAHat, pABar, proba = self.get_proba_lb(x)
    if pABar < 0.5:
      return Smooth.ABSTAIN, (0.0, 0.0), proba
    else:
      radius1 = self._certify_uniform_worst_case(x, pABar)
      # radius2 = self._certify_uniform_with_grad(x, cAHat, radius1)
      radius2 = 0
      return cAHat, (radius1, radius2), proba

  def _certify_uniform_worst_case(self, x, pABar):
    """ UniformBall L2 certifcate from Yang et al. https://arxiv.org/pdf/2002.08118.pdf """
    lambda_ = self.noise_scale * np.sqrt(self.dim + 2)
    radius = lambda_ * (
        2 - 4 * self.beta_dist.ppf(0.75 - 0.5 * pABar))
    return radius

  def _certify_uniform_with_grad(self, x, cAHat, radius_worst_case):
    # binary search to find the certificate radius
    start = self.binary_search_lower
    end = self.binary_search_upper
    # if start == 0: start = 0.0001
    while (end - start) > self.binary_search_tol:
      # if end < radius_worst_case:
      #   # abort if we are bellow the worst case
      #   return 0
      mid = (start + end) / 2
      if self._eval_gradient(x, cAHat, self.sample_n, mid):
        # start = mid
        end = mid
      else:
        # end = mid
        start = mid
      logging.info('binary search: {}   {}'.format(start, end))
    # we return the lower value found by the binary search
    return start

  def _eval_gradient(self, x, cAHat, sample_n, eps):
    pr1 = self._sample_noise(x, sample_n, self.noise_scale)[cAHat] / sample_n
    pr2 = self._sample_noise(x, sample_n, self.noise_scale+eps)[cAHat] / sample_n
    radius = self.noise_scale * np.sqrt(self.dim + 2)
    eps_radius = eps * np.sqrt(self.dim + 2)
    k = (pr2 - pr1) / eps_radius
    # numerically unstable
    # ratio1 = (radius / (radius + eps_radius))**self.dim
    # ratio2 = ((radius + eps_radius) / radius)**self.dim
    # return k > ratio1 * (1/eps) * (ratio2 * (1 - pr1) - 1/2) 
    ratio1 = self.dim * np.log(radius / (radius + eps_radius))
    ratio2 = self.dim * np.log((radius + eps_radius) / radius)
    bound = np.exp(ratio1 - np.log(eps_radius) + ratio2 + np.log(1 - pr1)) - \
        np.exp(-np.log(2) + ratio1 - np.log(eps_radius))
    return k > bound


  def _sample_noise(self, x, num, noise_scale):
    """ Sample the base classifier's prediction under noisy corruptions of the input x.
    :param x: the input [channel x width x height]
    :param num: number of samples to collect
    :param batch_size:
    :param noise_scale: std to use
    :return: an ndarray[int] of length n_classes containing the per-class counts
    """
    batch_size = self.batch_size_sample
    with torch.no_grad():
      counts = np.zeros(self.n_classes)
      for _ in range(ceil(num / batch_size)):
        this_batch_size = min(batch_size, num)
        num -= this_batch_size
        batch = x.repeat((this_batch_size, 1, 1, 1))
        noise = self.noise(batch) * noise_scale
        predictions = self.model(batch + noise).argmax(axis=1)
        counts += F.one_hot(predictions, num_classes=self.n_classes).sum(axis=0).cpu().numpy()
      return counts





