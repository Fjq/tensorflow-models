# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Converts the ilsvrc2014 detection dataset to something usable in TFRecords
Note that it doesn't download
Inspired from the flowers download_and_convert script.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import tarfile
from glob import glob

import tensorflow as tf

from datasets import dataset_utils

# The URL where the Flowers data can be downloaded.
# _DATA_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
_DATA_URLS = [
    'http://image-net.org/image/ilsvrc2014/ILSVRC2014_DET_train.tar',
    'http://image-net.org/image/ilsvrc2013/ILSVRC2013_DET_val.tar',
    'http://image-net.org/image/ilsvrc2014/ILSVRC2014_DET_bbox_train.tgz',
    'http://image-net.org/image/ilsvrc2013/ILSVRC2013_DET_bbox_val.tgz',

    # FIXME there seems to be missing bounding boxes for the
    # ILSVRC2014_DET_train.tar data, namely the ILSVRC2013_DET_train_extra*.tar
    # ones... maybe in there?
    # 'http://image-net.org/image/ilsvrc2013/ILSVRC2013_DET_bbox_train.tgz'
]

# The number of images in the validation set.
# _NUM_VALIDATION = 350

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _recursive_get_files(folder, ext='.JPEG'):
  """Returns all the files with extention `ext` from `folder` (recursively)."""
  print('recursive get files in ' + folder)
  out_files = []
  for root, dirs, files in os.walk(folder):
      print(root)
      for f in files:
          if f.endswith(ext):
              out_files.append(os.path.join(root, f))
  return out_files


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'detection_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, examples, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    examples: List of (path, class) tuples
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation', 'test']

  num_per_shard = int(math.ceil(len(examples) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(examples))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(examples), shard_id))
            sys.stdout.flush()
            filename, class_id = examples[i]
            # Read the filename:
            image_data = tf.gfile.FastGFile(filename, 'r').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            example = dataset_utils.image_to_tfexample(
                image_data, 'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _clean_up_temporary_files(dataset_dir):
  """Removes temporary files used to create the dataset.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  raise RuntimeError('hey!')
  filename = _DATA_URL.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)
  tf.gfile.Remove(filepath)

  tmp_dir = os.path.join(dataset_dir, 'flower_photos')
  tf.gfile.DeleteRecursively(tmp_dir)


def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation', 'test']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def run(dataset_dir, train_dir_pos, train_dir_neg, valid_dir_pos, valid_dir_neg, test_folder):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset will be written.
    rest: image folders for train/valid and pos/neg
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  # Download the data
  # TODO fix the folders for that
  # for url in _DATA_URLS:
  #   dataset_utils.download_and_uncompress_tarball(url, dataset_dir)

  # TODO intermerdiary steps are done by hand right now, using the scripts in
  # the `detection` folder
  # here we are assuming we have a folder with .JPEGs with positive examples
  # and another with negative examples, for both train and validation

  for split_name, pos_dir, neg_dir in [
          ('train', train_dir_pos, train_dir_neg),
          ('validation', valid_dir_pos, valid_dir_neg),
          ('test', test_folder, None)]:
      files_pos = _recursive_get_files(pos_dir)
      n = len(files_pos)
      classes_pos = [1] * len(files_pos)

      if neg_dir is not None:
          files_neg = _recursive_get_files(neg_dir)

          # just because we have so many positive negative examples
          if split_name == 'train':
              random.shuffle(files_neg)
              files_neg = files_neg[:n]

          classes_neg = [0] * len(files_neg)
      else:
          files_neg = []
          classes_neg = []

      files = files_pos + files_neg
      classes = classes_pos + classes_neg

      examples = zip(files, classes)
      random.seed(_RANDOM_SEED)
      random.shuffle(examples)

      _convert_dataset(split_name, examples, dataset_dir)

if __name__ == '__main__':
    run(*sys.argv[1:])

  # _clean_up_temporary_files(dataset_dir)
  # print('\nFinished converting the Flowers dataset!')
