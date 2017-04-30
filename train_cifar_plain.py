from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import time
from input_data_cifar  import create_train_datasets
from input_data_cifar import create_test_datasets
import data_helpers
import tensorflow as tf

FLAGS = None

NUM_IMAGES = 5000

# one ps and four workers

def main(_):
 
#  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
#  list_ = []
#  for line in open("/mnt/ds3lab/litian/input_data/cifar10/label3.txt"):
#      list_.append(['a', line.strip('\n')])
#  classes = np.array(list_)
#  print (len(classes))


#  train_dataset, mean, std = create_train_datasets(classes[:, 1], num_samples=NUM_IMAGES)
#  val_dataset = create_test_datasets(classes[:, 1], mean, std, num_samples=NUM_IMAGES)

#  val_images, val_labels = val_dataset.next_batch(20)

#  num_classes = len(classes)
#  print (num_classes)
  data_sets=data_helpers.load_data()

#  with tf.device('/gpu:0'):

  # Create the model
  x = tf.placeholder(tf.float32, shape = [None, 3072])
  y_ = tf.placeholder(tf.int64, shape=[None])

  w1 = tf.get_variable(name='w1',shape=[3072,240], initializer=tf.truncated_normal_initializer(stddev=1.0/np.sqrt(float(3072))),regularizer=tf.contrib.layers.l2_regularizer(0.1))
  b1 = tf.Variable(tf.zeros([240]))
  h1 = tf.nn.relu(tf.matmul(x, w1)+b1)

  w2 = tf.get_variable(name='w2',shape=[240,10], initializer=tf.truncated_normal_initializer(stddev=1.0/np.sqrt(float(240))),
          regularizer=tf.contrib.layers.l2_regularizer(0.1))
  b2 = tf.Variable(tf.zeros([10]))
  y = tf.matmul(h1, w2) + b2


  cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))

  train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(cross_entropy)

  correct_prediction = tf.equal(tf.argmax(y, 1), y_)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  sess = tf.Session()
  sess.run(tf.initialize_all_variables())

  zipped_data = zip(data_sets['images_train'], data_sets['labels_train'])
  batches = data_helpers.gen_batch(list(zipped_data), 128, 50000)
  for i in range(50000):
  #              batch_xs, batch_ys = mnist.train.next_batch(100)
     #   image_batch, label_batch = train_dataset.next_batch(60, random_crop=True)   
        batch = next(batches)
        image_batch, label_batch=zip(*batch)
        sess.run(train_step, feed_dict={x: image_batch, y_: label_batch})

        if i % 50 == 0:
              train_accuracy = sess.run(accuracy,feed_dict={x: image_batch, y_: label_batch})
              train_loss = sess.run(cross_entropy, feed_dict={x: image_batch, y_: label_batch})
              localtime = time.asctime(time.localtime(time.time()))
              print (localtime)
              print("step %d, training accuracy %g, training loss %g" % (i, train_accuracy, train_loss))
	if i % 500 == 0 :
	      val_accuracy = sess.run(accuracy, feed_dict={x: data_sets['images_test'], y_: data_sets['labels_test']})
              print("validation set accuracy %g" % val_accuracy)
            #print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
#  parser.add_argument('--job_name', type=str, help='either ps or worker')
#  parser.add_argument('--task_index', type=int, help='task index, starting from 0')

  FLAGS, unparsed = parser.parse_known_args() 
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


