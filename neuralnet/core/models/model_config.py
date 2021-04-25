
from . import resnet_model_cifar
from . import resnet_model_imagenet
from . import efficientnet_model as efficientnet 
from . import wide_resnet_model

# from . import lenet_model
# from .efficientnet.model import create_efficientnet_model

_model_name_to_imagenet_model = {
    'resnet': resnet_model_imagenet.resnet,
    'wide_resnet': wide_resnet_model.WideResnetModel,
    'efficientnet-b0': efficientnet.create_efficientnet_model,
    'efficientnet-b1': efficientnet.create_efficientnet_model,
    'efficientnet-b2': efficientnet.create_efficientnet_model,
    'efficientnet-b3': efficientnet.create_efficientnet_model,
    'efficientnet-b4': efficientnet.create_efficientnet_model,
    'efficientnet-b5': efficientnet.create_efficientnet_model,
    'efficientnet-b6': efficientnet.create_efficientnet_model,
    'efficientnet-b7': efficientnet.create_efficientnet_model,
}

_model_name_to_cifar_model = {
    # 'lenet': lenet_model.LeNet,
    'resnet': resnet_model_cifar.ResNet,
    # 'wide_resnet': wide_resnet_model.WideResnetModel,
}

def _get_model_map(dataset_name):
  """Get name to model map for specified dataset."""
  if dataset_name in ('cifar10', 'cifar100'):
    return _model_name_to_cifar_model
  elif dataset_name == 'mnist':
    return _model_name_to_mnist_model
  elif dataset_name in ('imagenet'):
    return _model_name_to_imagenet_model
  else:
    raise ValueError('Invalid dataset name: {}'.format(dataset_name))


def get_model_config(model_name, dataset_name, params, nclass, is_training):
  """Map model name to model network configuration."""
  model_map = _get_model_map(dataset_name)
  if model_name not in model_map:
    raise ValueError("Invalid model name '{}' for dataset '{}'".format(
                     model_name, dataset_name))
  else:
    return model_map[model_name](params, nclass, is_training)



