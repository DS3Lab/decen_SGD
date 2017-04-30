from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import time
import data_helpers

import tensorflow as tf

FLAGS = None

NUM_IMAGES = 5000

# one ps and four workers

def main(_):

  # cluster specification
  parameter_servers = ["spaceml1:2222"]
  workers = ["spaceml1:2223", "spaceml1:2224", "spaceml1:2225", "spaceml1:2226"]
  num_workers = len(workers)

  cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

  #local server, either ps or worker
  server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
  data_sets = data_helpers.load_data() 

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
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

        correct_prediction = tf.equal(tf.argmax(y, 1),y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #init_token_op = opt.get_init_tokens_op()
        #chief_queue_runner = opt.get_chief_queue_runner()

        saver = tf.train.Saver()     
        init_op = tf.global_variables_initializer()


        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/mnt/ds3lab/litian/logs",
                             init_op=init_op, 
                             saver=saver)

        zipped_data = zip(data_sets['images_train'], data_sets['labels_train'])
        batches = data_helpers.gen_batch(list(zipped_data), 128, 40000)

        with sv.managed_session(server.target) as sess:
            begin = time.time()
            for i in range(40000):
                batch = next(batches)
                image_batch, label_batch = zip(*batch) 
                sess.run(train_step, feed_dict={x: image_batch, y_: label_batch})

                if i % 50 == 0:
                    train_accuracy = sess.run(accuracy,feed_dict={x: image_batch, y_: label_batch})
                    train_loss = sess.run(cross_entropy, feed_dict={x: image_batch, y_: label_batch})
                    localtime = time.asctime(time.localtime(time.time()))
                    print (localtime)
                    tmp = time.time()
                    print ((tmp - begin)/60.0)
                    print("step %d, training accuracy %g, training loss %g" % (i, train_accuracy, train_loss))
            #print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

        sv.stop()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--job_name', type=str, help='either ps or worker')
  parser.add_argument('--task_index', type=int, help='task index, starting from 0')

  FLAGS, unparsed = parser.parse_known_args() 
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


# the best learning rate should be chosen in centralized setting

# in order to compare the real training time, we don't do testing in the experiments
# (test will add the communication cost for decentralized setting, since we need to get variables from n ps)
# however, we only need to get variables from one server in the centralized setting
# in order to see the performance, we compute the testing accuracy during training



