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
#  parameter_servers = ["sgs-gpu-02:2222", "sgs-gpu-02:2223", "sgs-gpu-03:2222", "sgs-gpu-03:2223"]
#  workers = ["sgs-gpu-02:2224", "sgs-gpu-02:2225", "sgs-gpu-03:2224", "sgs-gpu-03:2225"]
  parameter_servers = ["spaceml1:2222","spaceml1:2223","spaceml1:2224","spaceml1:2225"]
  workers = ["spaceml1:2226", "spaceml1:2227", "spaceml1:2228", "spaceml1:2229"]

  num_ps = len(parameter_servers)
  num_workers = len(workers)

  cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

  #local server, either ps or worker
  server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

  data_sets=data_helpers.load_data()

  W1=[0,0,0,0]
  b1=[0,0,0,0]
  W2=[0,0,0,0]
  b2=[0,0,0,0]



  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    with tf.device("/job:ps/task:0"):
        W1[0] = tf.get_variable(name='w10',shape=[3072,240], initializer=tf.truncated_normal_initializer(stddev=1.0/np.sqrt(float(3072))),
          regularizer=tf.contrib.layers.l2_regularizer(0.1))
#        W1[0] = tf.Variable(tf.random_normal([3072,240]))
        b1[0] = tf.Variable(tf.zeros([240]))
        W2[0] = tf.get_variable(name='w20', shape=[240,10], initializer=tf.truncated_normal_initializer(stddev=1.0/np.sqrt(float(120))),
          regularizer=tf.contrib.layers.l2_regularizer(0.1))
        #W2[0] = tf.Variable(tf.random_normal([240,10]))
        b2[0] = tf.Variable(tf.zeros([10]))
    with tf.device("/job:ps/task:1"):
        W1[1] = tf.get_variable(name='w11',shape=[3072,240], initializer=tf.truncated_normal_initializer(stddev=1.0/np.sqrt(float(3072))),
          regularizer=tf.contrib.layers.l2_regularizer(0.1))
        #W1[1] = tf.Variable(tf.random_normal([3072,240]))
        b1[1] = tf.Variable(tf.zeros([240]))
        W2[1] = tf.get_variable(name='w21', shape=[240,10], initializer=tf.truncated_normal_initializer(stddev=1.0/np.sqrt(float(120))),
          regularizer=tf.contrib.layers.l2_regularizer(0.1))
       # W2[1] = tf.Variable(tf.random_normal([240,10]))
        b2[1] = tf.Variable(tf.zeros([10]))
    with tf.device("/job:ps/task:2"):
        W1[2] = tf.get_variable(name='w12',shape=[3072,240], initializer=tf.truncated_normal_initializer(stddev=1.0/np.sqrt(float(3072))),
          regularizer=tf.contrib.layers.l2_regularizer(0.1))
        
        #W1[2] = tf.Variable(tf.random_normal([3072,240]))
        b1[2] = tf.Variable(tf.zeros([240]))
        W2[2] = tf.get_variable(name='w22', shape=[240,10], initializer=tf.truncated_normal_initializer(stddev=1.0/np.sqrt(float(120))),
          regularizer=tf.contrib.layers.l2_regularizer(0.1))
        
        #W2[2] = tf.Variable(tf.random_normal([240,10]))
        b2[2] = tf.Variable(tf.zeros([10]))
    with tf.device("/job:ps/task:3"):
        W1[3] = tf.get_variable(name='w13',shape=[3072,240], initializer=tf.truncated_normal_initializer(stddev=1.0/np.sqrt(float(3072))),
          regularizer=tf.contrib.layers.l2_regularizer(0.1))
       # W1[3] = tf.Variable(tf.random_normal([3072,240]))
        b1[3] = tf.Variable(tf.zeros([240]))
        W2[3] = tf.get_variable(name='w23', shape=[240,10], initializer=tf.truncated_normal_initializer(stddev=1.0/np.sqrt(float(120))),
          regularizer=tf.contrib.layers.l2_regularizer(0.1))
        #W2[3] = tf.Variable(tf.random_normal([240,10]))
        b2[3] = tf.Variable(tf.zeros([10]))

    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):

      
  # Create the model
        x = tf.placeholder(tf.float32, shape = [None, 3072])
        y_ = tf.placeholder(tf.int64, shape=[None])

       
        h1 = tf.nn.relu(tf.matmul(x, W1[FLAGS.task_index])+b1[FLAGS.task_index])      
        y = tf.matmul(h1, W2[FLAGS.task_index]) + b2[FLAGS.task_index]

        avg_W1 = tf.assign(W1[FLAGS.task_index], (W1[0]+W1[1]+W1[2]+W1[3])/4.0)
        avg_b1 = tf.assign(b1[FLAGS.task_index], (b1[0]+b1[1]+b1[2]+b1[3])/4.0)
        avg_W2 = tf.assign(W2[FLAGS.task_index], (W2[0]+W2[1]+W2[2]+W2[3])/4.0)
        avg_b2 = tf.assign(b2[FLAGS.task_index], (b2[0]+b2[1]+b2[2]+b2[3])/4.0)


        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))

        opt = tf.train.GradientDescentOptimizer(FLAGS.lr)

        grads_and_vars = opt.compute_gradients(cross_entropy, [W1[FLAGS.task_index], b1[FLAGS.task_index], W2[FLAGS.task_index], b2[FLAGS.task_index]])

#	w = W2[FLAGS.task_index]
#	b = b2[FLAGS.task_index]
        new_gv0 = (grads_and_vars[0][0]-(W1[(FLAGS.task_index - 1) % num_ps] + W1[(FLAGS.task_index + 1) % num_ps] - 2*W1[FLAGS.task_index])/(3*FLAGS.lr*1.0), grads_and_vars[0][1])
        new_gv1 = (grads_and_vars[1][0]-(b1[(FLAGS.task_index - 1) % num_ps] + b1[(FLAGS.task_index + 1) % num_ps] - 2*b1[FLAGS.task_index])/(3*FLAGS.lr*1.0), grads_and_vars[1][1]) 
        new_gv2 = (grads_and_vars[2][0]-(W2[(FLAGS.task_index - 1) % num_ps] + W2[(FLAGS.task_index + 1) % num_ps] - 2*W2[FLAGS.task_index])/(3*FLAGS.lr*1.0), grads_and_vars[2][1])
        new_gv3 = (grads_and_vars[3][0]-(b2[(FLAGS.task_index - 1) % num_ps] + b2[(FLAGS.task_index + 1) % num_ps] - 2*b2[FLAGS.task_index])/(3*FLAGS.lr*1.0), grads_and_vars[3][1])

        #print b1[FLAGS.task_index]
        g=grads_and_vars[1][0]
        new_gv=list()
        new_gv.append(new_gv0)
        new_gv.append(new_gv1) 
        new_gv.append(new_gv2)
        new_gv.append(new_gv3)

        train_step = opt.apply_gradients(new_gv)

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
            for i in range(40000):
                batch = next(batches)
                image_batch, label_batch=zip(*batch) 
                if i % 400 == 0 and (i / 400) % num_workers == FLAGS.task_index:
                        val_accuracy = sess.run(accuracy, feed_dict={x: data_sets['images_test'], y_: data_sets['labels_test']})
                        print("step %d, validation set accuracy %g" % (i, val_accuracy))
                        localtime = time.asctime(time.localtime(time.time()))
                        print (localtime)

                sess.run(train_step, feed_dict={x: image_batch, y_: label_batch})

                if i % 50 == 0:
                        train_accuracy = sess.run(accuracy,feed_dict={x: image_batch, y_: label_batch})
                        train_loss = sess.run(cross_entropy, feed_dict={x: image_batch, y_: label_batch})
                        localtime = time.asctime(time.localtime(time.time()))
                        print (localtime)
                        print("step %d, training accuracy %g, training loss %g" % (i, train_accuracy, train_loss))
        sv.stop()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--job_name', type=str, help='either ps or worker')
  parser.add_argument('--task_index', type=int, help='task index, starting from 0')
  parser.add_argument('--lr', type=float, help='learning rate')
  FLAGS, unparsed = parser.parse_known_args() 
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

'''
export CUDA_VISIBLE_DEVICES=0
python train_cifar10_decentralized.py --job_name=ps --task_index=0 --lr=0.0005

export CUDA_VISIBLE_DEVICES=1
python train_cifar10_decentralized.py --job_name=ps --task_index=1 --lr=0.0005

export CUDA_VISIBLE_DEVICES=2
python train_cifar10_decentralized.py --job_name=ps --task_index=2 --lr=0.0005

export CUDA_VISIBLE_DEVICES=3
python train_cifar10_decentralized.py --job_name=ps --task_index=3 --lr=0.0005


export CUDA_VISIBLE_DEVICES=4
python train_cifar10_decentralized.py --job_name=worker --task_index=0 --lr=0.0005

export CUDA_VISIBLE_DEVICES=5
python train_cifar10_decentralized.py --job_name=worker --task_index=1 --lr=0.0005

export CUDA_VISIBLE_DEVICES=6
python train_cifar10_decentralized.py --job_name=worker --task_index=2 --lr=0.0005

export CUDA_VISIBLE_DEVICES=7
python train_cifar10_decentralized.py --job_name=worker --task_index=3 --lr=0.0005


python train_cifar10_decentralized.py --job_name=worker --task_index=4 --lr=0.0005

'''


