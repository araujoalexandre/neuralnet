
import json
import time
import os
import re
import socket
import pprint
import logging
import glob
from os.path import join
from os.path import exists

import utils as global_utils
from dump_files import DumpFiles
from pkl_utils import pickle_load
from . import utils
from .models import model_config
from .dataset.readers import readers_config
from .randomized_smoothing import Smooth

import numpy as np
import torch
import torchvision.transforms.functional as F
import torch.backends.cudnn as cudnn


class Evaluator:
  """Evaluate a Pytorch Model."""

  def __init__(self, params):

    # Set up environment variables before doing any other global initialization to
    # make sure it uses the appropriate environment variables.
    utils.set_default_param_values_and_env_vars(params)

    self.params = params

    # Setup logging & log the version.
    global_utils.setup_logging(params.logging_verbosity)
    logging.info("Pytorch version: {}.".format(torch.__version__))
    logging.info("Hostname: {}.".format(socket.gethostname()))

    # print self.params parameters
    pp = pprint.PrettyPrinter(indent=2, compact=True)
    logging.info(pp.pformat(params.values()))

    self.train_dir = self.params.train_dir
    self.logs_dir = "{}_logs".format(self.train_dir)
    if self.train_dir is None:
      raise ValueError('Trained model directory not specified')
    self.num_gpus = self.params.num_gpus

    # create a mesage builder for logging
    self.message = global_utils.MessageBuilder()

    if self.params.cudnn_benchmark:
      cudnn.benchmark = True

    if self.params.num_gpus:
      self.batch_size = self.params.batch_size * self.num_gpus
    else:
      self.batch_size = self.params.batch_size

    if not self.params.data_pattern:
      raise IOError("'data_pattern' was not specified. "
        "Nothing to evaluate.")

    # load reader and model
    self.reader = readers_config[self.params.dataset](
      self.params, self.batch_size, self.num_gpus, is_training=False)
    self.model = model_config.get_model_config(
        self.params.model, self.params.dataset, self.params,
        self.reader.n_classes, is_training=False)

    self.model = torch.nn.DataParallel(self.model)
    self.model = self.model.cuda()
    self.model.eval()

    # add normalization as first layer of model
    if getattr(self.reader, 'normalize_mean', False):
      original_forward = self.model.forward
      def new_forward(inputs):
        inputs = F.normalize(
          inputs, self.reader.normalize_mean, self.reader.normalize_std)
        return original_forward(inputs)
      self.model.forward = new_forward

    # define Smooth classifier
    # we use scale as a common setting between normal and uniform 
    self.distribution = self.params.noise['distribution']
    if self.distribution == 'normal':
      self.noise_params = {
        'sigma': self.params.noise['normal']['sigma'],
        'scale': self.params.noise['normal']['sigma']
      }
    elif self.distribution == 'uniform':
      self.noise_params = {
        'scale': self.params.noise['uniform']['scale'],
        'low': self.params.noise['uniform']['low'],
        'high': self.params.noise['uniform']['high']
      }
    self.batch_size_sample = self.params.noise['batch_size']
    self.sample_n0= self.params.noise['N0']
    self.sample_n = self.params.noise['N']
    self.alpha = self.params.noise['alpha']
    self.smooth_model = Smooth(
      self.model, self.reader.n_classes, self.distribution, self.noise_params)

  def run(self):
    """Run evaluation of model with randomized smooting"""
    # normal evaluation has already been done
    # we get the best checkpoint of the model
    best_checkpoint, global_step = \
        global_utils.get_best_checkpoint(
          self.logs_dir, backend='pytorch')
    logging.info("Loading '{}'".format(best_checkpoint.split('/')[-1]))
    global_step, epoch = self.load_ckpt(best_checkpoint)
    self.eval(global_step, epoch)
    logging.info("Done evaluation with noise.")

  def load_ckpt(self, path):
    checkpoint = torch.load(path)
    global_step = checkpoint.get('global_step', 250)
    epoch = checkpoint.get('epoch', 1)
    if 'ema' in checkpoint.keys():
      logging.info("Loading ema state dict.")
      self.model.load_state_dict(checkpoint['ema'])
    else:
      if 'model_state_dict' not in checkpoint.keys():
        # if model_state_dict is not in checkpoint.keys() the 
        # checkpoint must be a pre-trained model
        # we should fix the problem of 'module'
        new_state_dict = {}
        for k, v in checkpoint.items():
          new_name = 'module.' + k
          new_state_dict[new_name] = v
        self.model.load_state_dict(new_state_dict)
      else:
        self.model.load_state_dict(checkpoint['model_state_dict'])
    self.model.eval()
    return global_step, epoch

  def _eval_with_noise_and_certify(self, data_loader): 

    logging.info("Eval with noise and Certify")
    logging.info("{} noise -- {}".format(self.distribution, self.noise_params['scale']))

    running_accuracy = 0
    running_accuracy_adv = 0
    running_inputs = 0

    for batch_n, (inputs, labels) in enumerate(data_loader):
      batch_start_time = time.time()
      inputs, labels = inputs.cuda(), labels.cuda()

      # predict without noise
      with torch.no_grad():
        outputs = self.model(inputs)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
      _, predicted = torch.max(outputs.data, 1)
      running_accuracy += predicted.eq(labels.data).cpu().sum().numpy()
      running_inputs += inputs.size(0)
      accuracy = running_accuracy / running_inputs

      # predict with noise
      for ii, x in enumerate(inputs):
        with torch.no_grad():
          cert_class, cert_radius, outputs_rs = self.smooth_model.certify(
            x, self.sample_n0, self.sample_n, self.alpha, self.batch_size_sample)

        results = f"{batch_n};{ii};{self.noise_params['scale']:.2f};{labels[ii]};"
        results += ";".join(["{:.2f}".format(x) for x in outputs[ii].data]) + ";"
        results += ";".join(["{:.2f}".format(x) for x in outputs_rs.data]) + ";"
        results += f"{cert_class};{cert_radius:.4f}\n"
        self.results_file.write(results)
        self.results_file.flush()
 
      seconds_per_batch = time.time() - batch_start_time
      examples_per_second = inputs.size(0) / seconds_per_batch
      self.message.add('accuracy', accuracy, format='.5f')
      self.message.add('imgs/sec', examples_per_second, format='.2f')
      logging.info(self.message.get_message())

  def _init_results_files(self, from_pickle):
    ext = '' if not from_pickle else '_from_pickle'
    filename = "stats_eval_noise_{}_{:.2f}_samples_{}{}.txt".format(
      self.distribution, self.noise_params['scale'], self.sample_n, ext)
    self.results_file = open(join(self.logs_dir, filename), 'w')

  def _data_loader_pickle(self):
    data_loader, _ = self.reader.load_dataset()
    # load path of pickle files
    logging.info("loading files from: {}".format(self.params.path_to_dump_files))
    files = glob.glob('{}/{}'.format(self.logs_dir, self.params.path_to_dump_files))
    sort_func = lambda x: int(x.split('_')[-1].replace('.pkl', ''))
    files = sorted(list(filter(lambda x: '/adv_' in x, files)), key=sort_func)
    for path, (_, labels) in zip(files, data_loader):
      inputs = pickle_load(path)
      inputs = torch.FloatTensor(inputs).cuda()
      yield inputs, labels

  def eval(self, global_step, epoch):
    """Run the evaluation loop once."""
    if not self.params.eval_with_pickle_files:
      self._init_results_files(False)
      data_loader, _ = self.reader.load_dataset()
      self._eval_with_noise_and_certify(data_loader)
    else:
      self._init_results_files(True)
      data_loader = self._data_loader_pickle()
      self._eval_with_noise_and_certify(data_loader)
    self.results_file.close()


