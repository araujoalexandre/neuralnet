
import os
import pickle
from os.path import join, basename, exists, normpath


def pickle_dump(file, path):
  """function to dump picke object."""
  with open(path, 'wb') as f:
    pickle.dump(file, f, -1)


class DumpFiles:

  def __init__(self, params):
    self.batch_id = 0
    self.params = params
    self.logs_dir = "{}_logs".format(self.params.train_dir)

    attack_method = self.params.attack_method
    eot_samples = getattr(self.params, 'eot_samples', 1)

    self.img_filename = "dump_{}_img_{}_{}.pkl".format(
      attack_method, eot_samples, '{}')
    self.adv_filename = "dump_{}_adv_{}_{}.pkl".format(
      attack_method, eot_samples, '{}')
    self.preds_filename = "dump_{}_preds_img_{}_{}".format(
      attack_method, eot_samples, '{}')
    self.preds_adv_filename = "dump_{}_preds_adv_{}_{}".format(
      attack_method, eot_samples, '{}')

    # create main dump folder
    attack_folder = join(self.logs_dir, "dump_{}".format(attack_method))
    if not exists(attack_folder):
      os.mkdir(attack_folder)
    # create sub folder from sample params
    sample_folder = join(attack_folder, "sample_{}".format(eot_samples))
    if not exists(sample_folder):
      os.mkdir(sample_folder)
    # create final dump folder from params
    params = self.params.attack_params
    folder_name = []
    for k,v in params.items():
      folder_name.append('{}_{}'.format(k, str(v)))
    folder_name = '_'.join(folder_name)
    self.dump_folder = join(sample_folder, folder_name)
    if not exists(self.dump_folder):
      os.mkdir(self.dump_folder)

  def files(self, values):
    path_img = join(self.dump_folder, self.img_filename.format(self.batch_id))
    path_adv = join(self.dump_folder, self.adv_filename.format(self.batch_id))
    path_preds_img = join(self.dump_folder, self.preds_filename.format(self.batch_id))
    path_preds_adv = join(self.dump_folder,
                          self.preds_adv_filename.format(self.batch_id))
    pickle_dump(values['images'], path_img)
    pickle_dump(values['images_adv'], path_adv)
    pickle_dump(values['predictions'], path_preds_img)
    pickle_dump(values['predictions_adv'], path_preds_adv)
    self.batch_id += 1
