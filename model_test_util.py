
"""Common utils for tests for object detection tflearn model."""

from __future__ import absolute_import

import os
import tempfile
import tensorflow as tf


from object_detection import model
from object_detection import model_hparams

FLAGS = tf.flags.FLAGS

FASTER_RCNN_MODEL_NAME = 'faster_rcnn_resnet50_pets'
SSD_INCEPTION_MODEL_NAME = 'ssd_inception_v2_pets'


def GetPipelineConfigPath(model_name):
  """Returns path to the local pipeline config file."""
  return os.path.join(tf.resource_loader.get_data_files_path(), 'samples',
                      'configs', model_name + '.config')


def InitializeFlags(model_name_for_test):
  FLAGS.model_dir = tempfile.mkdtemp()
  FLAGS.pipeline_config_path = GetPipelineConfigPath(model_name_for_test)


def BuildExperiment():
  """Builds an Experiment object for testing purposes."""
  run_config = tf.contrib.learn.RunConfig()
  hparams = model_hparams.create_hparams(
      hparams_overrides='load_pretrained=false')

  # pylint: disable=protected-access
  experiment_fn = model.build_experiment_fn(10, 10)
  # pylint: enable=protected-access
  return experiment_fn(run_config, hparams)
