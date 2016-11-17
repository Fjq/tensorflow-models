# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 3, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])
    label -= FLAGS.labels_offset

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    #####################################
    # Select the preprocessing function #
    #####################################
    # I set `is_training` to True because I'm suspecting that the unprocessed
    # examples are too easy
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=True)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    # print(tf.random_crop(image, [ru,ru,3]).dtype)
    # image = tf.cond(tf.equal(label, 1),
                    # lambda: image,
                    # lambda: tf.random_crop(image, [ru,ru,3])
                    # )

    from common import prepare_neg

    image = prepare_neg(image)
    eval_iage_size = FLAGS.eval_image_size or network_fn.default_image_size
    image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=20,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=20)
    # images = tf.expand_dims(image, 0)

    tf.image_summary('input', images, max_images=20)
    # tf.image_summary('labels', tf.reshape(tf.cast(labels[:20], tf.uint8) * 255, [1,1,20,1]))

    ####################
    # Define the model #
    ####################
    logits, endpoints = network_fn(images)
    logits = endpoints['Predictions']

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    # tf.image_summary('logits', tf.reshape(tf.cast(logits[:20], tf.uint8) * 255, [1,1,20,1]))
    # predictions = tf.argmax(logits, 1)
    tf.image_summary('logits', tf.reshape(logits, [-1,1,2,1]), max_images=20)

    prepool = endpoints['PrePool']
    shape = prepool.get_shape().as_list()
    nb, height, width, chan = shape
    prepool = tf.reshape(prepool, [shape[0] * shape[1] * shape[2], shape[3]])
    with tf.variable_scope('InceptionResnetV2', 'InceptionResnetV2', [prepool], reuse=True), tf.variable_scope('Logits'):
        logits2 = slim.fully_connected(prepool, dataset.num_classes, activation_fn=None,
                                       scope='Logits')
        # logits2 = tf.nn.softmax(logits2)
        logits2 = tf.reshape(logits2, shape[:3] + [dataset.num_classes])
        logits2 = tf.pad(logits2, [[0,0], [0,1], [0,1], [0,0]])
        logits2 = tf.transpose(logits2, perm=[0,1,3,2])
        # logits2 = tf.reshape(logits2, [shape[0], shape[1] , -1, 1])
        logits2 = tf.reshape(logits2, [1, nb * (height+1), -1, 1])
        # logits2 = tf.cast(logits2 * 255., tf.uint8)

    tf.image_summary('logits2', logits2, max_images=20)
    # predictions = tf.reshape(logits, [-1,2])
    # prepool = tf.Print(prepool, [prepool.get_shape()], 'prepool')

    # fc_weights =
    # slim.get_unique_variable('InceptionResnetV2/Logits/Logits/weights')
    # print(shape)

    # labels = tf.squeeze(labels)

    # print(images.get_shape())
    # print(labels.get_shape())
    # images_trans = images[:20]
    # labels_trans = tf.cast(tf.reshape(labels[:20], [-1,1,1,1]), tf.float32)
    # labels_trans = tf.tile(labels_trans, [1,299,10,3])

    # pred_trans = tf.reshape(tf.cast(predictions[:20], tf.float32), [-1,1,1,1])
    # pred_trans = tf.tile(pred_trans, [1,299,10,3])
    # im_lb = tf.concat(2, [images_trans, labels_trans, pred_trans])
    # tf.image_summary('image_batch', im_lb, max_images=20)

    images_trans = images[:20]
    labels_trans = tf.cast(tf.reshape(logits[:20], [-1,1,2,1]), tf.float32)
    labels_trans = tf.tile(labels_trans, [1,299,1,3])
    im_lb = tf.concat(2, [images_trans, labels_trans])
    tf.image_summary('image_batch', im_lb, max_images=20)

    # Define the metrics:
    # names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        # 'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        # 'Recall@5': slim.metrics.streaming_recall_at_k(
            # logits, labels, 5),
    # })

    # Print the summaries to screen.
    # for name, value in names_to_values.iteritems():
    #   summary_name = 'eval/%s' % name
    #   op = tf.scalar_summary(summary_name, value, collections=[])
    #   op = tf.Print(op, [value], summary_name)
    #   tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    slim.evaluation.evaluate_once(
        master=FLAGS.master,
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.eval_dir,
        eval_op=[logits, logits2],
        num_evals=1,
        final_op=[logits, logits2],
        variables_to_restore=variables_to_restore)


if __name__ == '__main__':
  tf.app.run()
