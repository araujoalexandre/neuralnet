
import os
import time
import datetime
import pprint
import socket
import logging
import warnings
import glob
from os import mkdir
from os.path import join
from os.path import exists

import utils
from .models import model_config
from .lipschitz import LipschitzRegularization
from .dataset.readers import readers_config

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn import SyncBatchNorm
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel
from torch.distributions import Categorical, kl_divergence


class Trainer:
  """A Trainer to train a PyTorch."""

  def __init__(self, params):
    """Creates a Trainer.
    """
    utils.set_default_param_values_and_env_vars(params)
    self.params = params

    # Setup logging & log the version.
    utils.setup_logging(params.logging_verbosity)

    self.job_name = self.params.job_name  # "" for local training
    self.is_distributed = bool(self.job_name)
    self.task_index = self.params.task_index
    self.local_rank = self.params.local_rank
    self.start_new_model = self.params.start_new_model
    self.train_dir = self.params.train_dir
    self.num_gpus = self.params.num_gpus
    if self.num_gpus and not self.is_distributed:
      self.batch_size = self.params.batch_size * self.num_gpus
    else:
      self.batch_size = self.params.batch_size

    # print self.params parameters
    if self.start_new_model and self.local_rank == 0:
      pp = pprint.PrettyPrinter(indent=2, compact=True)
      logging.info(pp.pformat(params.values()))

    if self.local_rank == 0:
      logging.info("PyTorch version: {}.".format(torch.__version__))
      logging.info("NCCL Version {}".format(torch.cuda.nccl.version()))
      logging.info("Hostname: {}.".format(socket.gethostname()))

    if self.is_distributed:
      self.num_nodes = len(params.worker_hosts.split(';'))
      self.world_size = self.num_nodes * self.num_gpus
      self.rank = self.task_index * self.num_gpus + self.local_rank
      dist.init_process_group(
        backend='nccl', init_method='env://',
        timeout=datetime.timedelta(seconds=30))
      if self.local_rank == 0:
        logging.info('World Size={} => Total batch size {}'.format(
          self.world_size, self.batch_size*self.world_size))
      self.is_master = bool(self.rank == 0)
    else:
      self.world_size = 1
      self.is_master = True

    # create a mesage builder for logging
    self.message = utils.MessageBuilder()

    # load reader and model
    self.reader = readers_config[self.params.dataset](
      self.params, self.batch_size, self.num_gpus, is_training=True)

    # load model
    self.model = model_config.get_model_config(
        self.params.model, self.params.dataset, self.params,
        self.reader.n_classes, is_training=True)
    # add normalization as first layer of model
    if self.params.add_normalization:
      # In order to certify radii in original coordinates rather than standardized coordinates, we
      # add the noise _before_ standardizing, which is why we have standardization be the first
      # layer of the classifier rather than as a part of preprocessing as is typical.
      normalize_layer = self.reader.get_normalize_layer()
      self.model = torch.nn.Sequential(normalize_layer, self.model)

    # define DistributedDataParallel job
    self.model = SyncBatchNorm.convert_sync_batchnorm(self.model)
    torch.cuda.set_device(params.local_rank)
    self.model = self.model.cuda()
    i = params.local_rank
    self.model = DistributedDataParallel(
      self.model, device_ids=[i], output_device=i)
    if self.local_rank == 0:
      logging.info('Model defined with DistributedDataParallel')

    # define set for saved ckpt
    self.saved_ckpts = set([0])

    # define optimizer
    self.optimizer = utils.get_optimizer(
                       self.params.optimizer,
                       self.params.optimizer_params,
                       self.params.init_learning_rate,
                       self.params.weight_decay,
                       self.model.parameters())

    # define learning rate scheduler
    self.scheduler = utils.get_scheduler(
      self.optimizer, self.params.lr_scheduler,
      self.params.lr_scheduler_params)

    # if start_new_model is False, we restart training
    if not self.start_new_model:
      if self.local_rank == 0:
        logging.info('Restarting training...')
      self._load_state()

    # define Lipschitz regularization module
    if self.params.lipschitz_regularization:
      if self.local_rank == 0:
        logging.info(
          "Lipschitz regularization with decay {}, start after epoch {}".format(
            self.params.lipschitz_decay, self.params.lipschitz_start_epoch))
      self.lipschitz = LipschitzRegularization(
        self.model, self.params, self.reader, self.local_rank)

    # exponential moving average
    self.ema = None
    if getattr(self.params, 'ema', False) > 0:
      self.ema = utils.EMA(self.params.ema)

    # if adversarial training, create the attack class
    if self.params.adversarial_training:
      if self.local_rank == 0:
        logging.info('Adversarial Training')
      attack_params = self.params.adversarial_training_params
      if 'eps_iter' in attack_params.keys() and attack_params['eps_iter'] == -1:
        eps = attack_params['eps']
        n_iter = attack_params['nb_iter']
        attack_params['eps_iter'] = eps / n_iter * 2
        if self.local_rank == 0:
          logging.info('Learning rate for attack: {}'.format(attack_params['eps_iter']))
      self.attack = utils.get_attack(
                      self.model,
                      self.reader.n_classes,
                      self.params.adversarial_training_name,
                      attack_params)

    # init noise
    if self.params.adaptive_noise and self.params.additive_noise:
      raise ValueError("Adaptive and Additive Noise should not be set together")
    if self.params.adaptive_noise:
      if self.local_rank == 0:
        logging.info('Training with Adaptive Noise: {} {}'.format(
          self.params.noise_distribution, self.params.noise_scale))
    elif self.params.additive_noise:
      if self.local_rank == 0:
        logging.info('Training with Noise: {} {}'.format(
          self.params.noise_distribution, self.params.noise_scale))
    if self.params.adaptive_noise or self.params.additive_noise:
      self.noise = utils.Noise(self.params)

    # stability training
    if self.params.stability_training:
      if self.local_rank == 0:
        logging.info("Training with Stability Training: {}".format(
          self.params.stability_training_lambda))
      if not any([
        self.params.adversarial_training,
        self.params.adaptive_noise,
        self.params.additive_noise]):
        raise ValueError("Adversarial Training or Adaptive Noise should be activated")


  def _load_state(self):
    # load last checkpoint
    checkpoints = glob.glob(join(self.train_dir, "model.ckpt-*.pth"))
    get_model_id = lambda x: int(x.strip('.pth').strip('model.ckpt-'))
    checkpoints = sorted(
      [ckpt.split('/')[-1] for ckpt in checkpoints], key=get_model_id)
    path_last_ckpt = join(self.train_dir, checkpoints[-1])
    self.checkpoint = torch.load(path_last_ckpt)
    self.model.load_state_dict(self.checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
    self.scheduler.load_state_dict(self.checkpoint['scheduler'])
    self.saved_ckpts.add(self.checkpoint['epoch'])
    epoch = self.checkpoint['epoch']
    if self.local_rank == 0:
      logging.info('Loading checkpoint {}'.format(checkpoints[-1]))


  def _save_ckpt(self, step, epoch, final=False):
    """Save ckpt in train directory."""
    freq_ckpt_epochs = self.params.save_checkpoint_epochs
    if (epoch % freq_ckpt_epochs == 0 and self.is_master \
        and epoch not in self.saved_ckpts) \
         or (final and self.is_master):
      ckpt_name = "model.ckpt-{}.pth".format(step)
      ckpt_path = join(self.train_dir, ckpt_name)
      if exists(ckpt_path): return 
      self.saved_ckpts.add(epoch)
      state = {
        'epoch': epoch,
        'global_step': step,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict()
      }
      if self.ema is not None:
        state['ema'] = self.ema.state_dict()
      logging.info("Saving checkpoint '{}'.".format(ckpt_name))
      torch.save(state, ckpt_path)


  def run(self):
    """Performs training on the currently defined Tensorflow graph.
    """
    # reset the training directory if start_new_model is True
    if self.is_master and self.start_new_model and exists(self.train_dir):
      utils.remove_training_directory(self.train_dir)
    if self.is_master and self.start_new_model:
      mkdir(self.train_dir)

    if self.params.torch_random_seed is not None:
      random.seed(self.params.torch_random_seed)
      torch.manual_seed(self.params.torch_random_seed)
      cudnn.deterministic = True
      warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')

    if self.params.cudnn_benchmark:
      cudnn.benchmark = True

    self._run_training()

  def _run_training(self):

    if self.params.lb_smooth == 0:
      self.criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
      if self.local_rank == 0:
        logging.info("Using CrossEntropyLoss with label smooth {}.".format(
          self.params.lb_smooth))
      self.criterion = utils.CrossEntropyLabelSmooth(
        self.reader.n_classes, self.params.lb_smooth)

    # if start_new_model is True, global_step = 0
    # else we get global step from checkpoint
    if self.start_new_model:
      start_epoch = 0
      global_step = 0
    else:
      start_epoch = self.checkpoint['epoch']
      global_step = self.checkpoint['global_step']

    data_loader, sampler = self.reader.load_dataset()
    if sampler is not None:
      assert sampler.num_replicas == self.world_size

    batch_size = self.batch_size
    if self.is_distributed:
      n_files = sampler.num_samples
    else:
      n_files = self.reader.n_train_files

    if self.local_rank == 0:
      logging.info("Number of files on worker: {}".format(n_files))
      logging.info("Start training")

    profile_enabled = False
    for epoch_id in range(start_epoch, self.params.num_epochs):
      if self.is_distributed:
        sampler.set_epoch(epoch_id)
      for n_batch, data in enumerate(data_loader):
        epoch = (int(global_step) * batch_size) / n_files
        with torch.autograd.profiler.profile(
            enabled=profile_enabled, use_cuda=True) as prof:
          self._training(data, epoch, global_step)
        if profile_enabled:
          logging.info(prof.key_averages().table(sort_by="self_cpu_time_total"))
          # prof.export_chrome_trace(join(
          #   self.train_dir+'_logs', 'trace_{}.json'.format(global_step)))
        self._save_ckpt(global_step, epoch_id)
        global_step += 1

      self.scheduler.step()
    self._save_ckpt(global_step, epoch_id, final=True)
    logging.info("Done training -- epoch limit reached.")


  def _process_gradient(self, step):
    if self.params.gradient_clip_by_norm:
      if step == 0 and self.local_rank == 0:
        logging.info("Clipping Gradient by norm: {}".format(
          self.params.gradient_clip_by_norm))
      torch.nn.utils.clip_grad_norm_(
        self.model.parameters(), self.params.gradient_clip_by_norm)
    elif self.params.gradient_clip_by_value:
      if step == 0 and self.local_rank == 0:
        logging.info("Clipping Gradient by value: {}".format(
          self.params.gradient_clip_by_value))
      torch.nn.utils.clip_grad_value_(
        self.model.parameters(), self.params.gradient_clip_by_value)


  def _check_memory_limit(self, inputs, n_sample):
    self.model.eval()
    with torch.no_grad():
      batch_size, *img_size = inputs.shape
      x = inputs.repeat(n_sample, 1, 1, 1)
      x = x.reshape(batch_size*n_sample, *img_size)
      _ = self.model(x)
    self.model.train()
    logging.info('Checking memory limit... ok')


  def _adaptive_noise(self, step, inputs, labels):

    self.mean_noise = 0.
    batch_size, *img_size = inputs.shape
    n_classes = self.reader.n_classes
    n_sample = self.params.n_sample

    self.noise_scale = self.params.noise_scale
    self.n_scales = self.params.n_scales
    self.noise_step = self.params.noise_step

    # check memory limit
    if step == 0 and self.local_rank == 0:
      self._check_memory_limit(inputs, n_sample)

    preds = self.model(inputs).argmax(axis=1)
    correct = (preds == labels)
    n_correct = correct.sum()

    # if no correct examples, we don't inject noise
    if n_correct == 0:
      return inputs

    idx1 = list(range(n_correct))
    idx2 = list(labels[correct])
    # generate the noise level to compute the gradient
    # scales = np.arange(self.noise_scale, 0., -self.noise_step)
    # scales = scales[:self.n_scales][::-1].copy()
    max_noise_scale = self.noise_scale * 2 - 0.05
    scales = np.linspace(0.05, max_noise_scale, num=self.n_scales)
    if step == 0 and self.local_rank == 0:
      scales_str = ', '.join(['{:.2f}'.format(x) for x in scales])
      logging.info('Scales for Adaptive Noise: [{}], with samples {}, mean {}'.format(
        scales_str, n_sample, np.mean(scales)))

    self.model.eval()
    with torch.no_grad():
      probabilities = torch.zeros((n_correct, self.n_scales+1))
      probabilities[:, 0] = 1
      x = inputs[correct, :, :, :].repeat(n_sample, 1, 1, 1)
      x = x.reshape(n_sample * n_correct, *img_size) 

      for i, scale in enumerate(scales):
        x_noise = x + self.noise(x) * scale
        logits = self.model(x_noise).cpu()
        predictions = torch.argmax(logits, axis=1)
        predictions_one_hot = F.one_hot(predictions, num_classes=n_classes).float()
        predictions_rs = predictions_one_hot.reshape(
          n_sample, n_correct, n_classes).mean(axis=0)
        proba_correct_class = predictions_rs[idx1, idx2]
        probabilities[:, i+1] = proba_correct_class

    self.model.train()

    # we take the index of the first value below 1
    idx_diff = ((probabilities < 1)*1).argmax(axis=1)
    noise_level = torch.FloatTensor([0] + list(scales))
    noise_level = noise_level.repeat(n_correct, 1)
    noise_level = noise_level[idx1, idx_diff] 

    if noise_level.sum() == 0:
      return inputs

    # inject noise to inputs
    adaptive_scales = (correct * 1.).to(inputs.device)
    adaptive_scales[correct] = noise_level.to(inputs.device)
    adaptive_scales = adaptive_scales.reshape(-1, 1)
    self.mean_noise = adaptive_scales.mean()

    inputs = inputs.reshape(batch_size, -1)
    noise = self.noise(inputs).reshape(batch_size, -1)
    inputs = inputs + noise * adaptive_scales
    inputs = inputs.reshape(batch_size, *img_size)
    return inputs


  def _training(self, data, epoch, step):

    batch_start_time = time.time()
    n_classes = self.reader.n_classes
    inputs, labels = data
    inputs = inputs.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)

    # save original inputs for stability training
    if self.params.stability_training:
      inputs_clean = inputs.clone()

    # generate adversarial attacks
    if self.params.adversarial_training:
      if inputs.min() < 0 or inputs.max() > 1:
        raise ValueError('Input values should be in the [0, 1] range.')
      inputs = self.attack.perturb(inputs)

    # Adaptive noise
    if self.params.adaptive_noise:
      inputs = self._adaptive_noise(step, inputs, labels)
    elif self.params.additive_noise:
      inputs = inputs + self.noise(inputs) * self.params.noise_scale
    

    total_loss = 0.
    outputs = self.model(inputs)
    loss_ce = self.criterion(outputs, labels)
    total_loss += loss_ce

    if self.params.stability_training:
      outputs_clean = self.model(inputs_clean)
      outputs_clean = Categorical(logits=outputs_clean)
      outputs = Categorical(logits=outputs)
      # the kl_divergence function convert the logits to softmax
      loss_stability_training = self.params.stability_training_lambda * \
          kl_divergence(outputs_clean, outputs).mean()
      total_loss += loss_stability_training

    if self.params.lipschitz_regularization and \
       epoch >= self.params.lipschitz_start_epoch:
      if epoch == self.params.lipschitz_start_epoch and self.local_rank == 0:
        logging.info("Start Lipschitz Regularization")
      loss_lip = self.lipschitz.get_loss(epoch, self.model)
      total_loss += loss_lip

    self.optimizer.zero_grad()
    total_loss.backward()
    self._process_gradient(step)
    self.optimizer.step()

    # update ema
    if self.ema is not None:
      self.ema(self.model, step)

    seconds_per_batch = time.time() - batch_start_time
    examples_per_second = self.batch_size / seconds_per_batch
    examples_per_second *= self.world_size
    
    if step == 10 and self.is_master:
      n_imgs_to_process = self.reader.n_train_files * self.params.num_epochs
      total_seconds = n_imgs_to_process / examples_per_second
      n_days = total_seconds // 86400
      n_hours = (total_seconds % 86400) / 3600
      logging.info(
        'Approximated training time: {:.0f} days and {:.1f} hours'.format(
          n_days, n_hours))

    local_rank = self.local_rank
    to_print = step % self.params.frequency_log_steps == 0
    if (to_print and local_rank == 0) or (step == 1 and local_rank == 0):
      lr = self.optimizer.param_groups[0]['lr']
      self.message.add("epoch", epoch, format="4.2f")
      self.message.add("step", step, width=5, format=".0f")
      self.message.add("lr", lr, format=".6f")
      self.message.add("ce", loss_ce, format=".4f")
      if self.params.adaptive_noise:
        self.message.add('m-noise', self.mean_noise, format=".2f")
      if self.params.lipschitz_regularization and \
         epoch >= self.params.lipschitz_start_epoch:
        self.message.add("lip", loss_lip, format=".4f")
      if self.params.stability_training:
        self.message.add("st", loss_stability_training, format=".4f")
      self.message.add("imgs/sec", examples_per_second, width=5, format=".0f")
      logging.info(self.message.get_message())




