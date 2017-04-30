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
  # in order to prevent ps from occupying GPUs, first start workers, then start parameter servers

  parameter_servers = ["sgs-gpu-02:2222"]
  workers = ["sgs-gpu-02:2223", "sgs-gpu-02:2224", "sgs-gpu-03:2222", "sgs-gpu-03:2223"]
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
        global_step = tf.get_variable('global_step', [], initializer = tf.constant_initializer(0), trainable = False)

        x = tf.placeholder(tf.float32, shape = [None, 3072])
        y_ = tf.placeholder(tf.int64, shape=[None])

        w1 = tf.get_variable(name='w1',shape=[3072,240], initializer=tf.truncated_normal_initializer(stddev=1.0/np.sqrt(float(3072))),
          regularizer=tf.contrib.layers.l2_regularizer(0.1))
        b1 = tf.Variable(tf.zeros([240]))
        h1 = tf.nn.relu(tf.matmul(x, w1)+b1)

        w2 = tf.get_variable(name='w2', shape=[240,10], initializer=tf.truncated_normal_initializer(stddev=1.0/np.sqrt(float(120))),
          regularizer=tf.contrib.layers.l2_regularizer(0.1))
        b2 = tf.Variable(tf.zeros([10]))
        y = tf.matmul(h1, w2) + b2


        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))

        opt = tf.train.GradientDescentOptimizer(0.0005)
        opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate = num_workers, total_num_replicas = num_workers)

        train_step = opt.minimize(cross_entropy, global_step = global_step)

        correct_prediction = tf.equal(tf.argmax(y, 1),y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #init_token_op = opt.get_init_tokens_op()
        #chief_queue_runner = opt.get_chief_queue_runner()

        saver = tf.train.Saver()     
        init_op = tf.global_variables_initializer()

        init_token_op = opt.get_init_tokens_op()
        chief_queue_runner = opt.get_chief_queue_runner()

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/mnt/ds3lab/litian/logs",
                             init_op=init_op, 
                             saver=saver, global_step = global_step)

        zipped_data = zip(data_sets['images_train'], data_sets['labels_train'])
        batches = data_helpers.gen_batch(list(zipped_data), 128, 50000)

        # start a session
        sess = sv.prepare_or_wait_for_session(server.target)

        if FLAGS.task_index == 0:
          sv.start_queue_runners(sess, [chief_queue_runner])
          sess.run(init_token_op)

        for i in range(50000):
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

        sv.stop()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--job_name', type=str, help='either ps or worker')
  parser.add_argument('--task_index', type=int, help='task index, starting from 0')

  FLAGS, unparsed = parser.parse_known_args() 
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


