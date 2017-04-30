from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import time
from input_data_mnist  import create_datasets
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

NUM_IMAGES = 1000

def main(_):

  # cluster specification
  parameter_servers = ["spaceml1:2222","spaceml1:2223","spaceml1:2224","spaceml1:2225"]
  workers = ["spaceml1:2226", "spaceml1:2227", "spaceml1:2228", "spaceml1:2229"]

  num_ps = len(parameter_servers)
  num_worker = num_ps

  cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

  #local server, either ps or worker
  server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
 
#  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  list_ = []
  for line in open("/home/litian/data/label2.txt"):
      list_.append(['a', line.strip('\n')])
  classes = np.array(list_)
  print (len(classes))


  train_dataset, val_dataset, test_dataset = create_datasets(classes[:, 1], num_samples=NUM_IMAGES, val_fraction=0.2,
                                                           test_fraction=0.2)
  val_images, val_labels = val_dataset.next_batch(20)
  num_classes = len(classes)
  print (num_classes)

#  W=[tf.Variable(tf.zeros([784, 10])),tf.Variable(tf.zeros([784, 10])),tf.Variable(tf.zeros([784, 10])),tf.Variable(tf.zeros([784, 10]))]
#  b=[tf.Variable(tf.zeros([10])),tf.Variable(tf.zeros([10])),tf.Variable(tf.zeros([10])),tf.Variable(tf.zeros([10]))]
  W=[0,0,0,0]
  b=[0,0,0,0]
  if FLAGS.job_name == "ps":
    #server.join()
#    with tf.device("/job:ps/task:0"):
#        W[0] = tf.Variable(tf.zeros([784, 10],dtype=tf.float32))
#        b[0] = tf.Variable(tf.zeros([10], dtype=tf.float32))
#    with tf.device("/job:ps/task:1"):
#        W[1] = tf.Variable(tf.zeros([784, 10], dtype=tf.float32))
#        b[1] = tf.Variable(tf.zeros([10], dtype=tf.float32))
#    with tf.device("/job:ps/task:2"):
#        W[2] = tf.Variable(tf.zeros([784, 10], dtype=tf.float32))
#        b[2] = tf.Variable(tf.zeros([10], dtype=tf.float32))
#    with tf.device("/job:ps/task:3"):
#        W[3] = tf.Variable(tf.zeros([784, 10], dtype=tf.float32))
#        b[3] = tf.Variable(tf.zeros([10], dtypw=tf.float32))
    server.join()
  elif FLAGS.job_name == "worker":
    with tf.device("/job:ps/task:0"):
        W[0] = tf.Variable(tf.zeros([784, 10]))
        b[0] = tf.Variable(tf.zeros([10]))
    with tf.device("/job:ps/task:1"):
        W[1] = tf.Variable(tf.zeros([784, 10])) 
        b[1] = tf.Variable(tf.zeros([10]))
    with tf.device("/job:ps/task:2"):
        W[2] = tf.Variable(tf.zeros([784, 10]))    
        b[2] = tf.Variable(tf.zeros([10]))
    with tf.device("/job:ps/task:3"):
        W[3] = tf.Variable(tf.zeros([784, 10]))    
        b[3] = tf.Variable(tf.zeros([10]))

    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):

  # Create the model
        x = tf.placeholder(tf.float32, [None, 784])
        y = tf.matmul(x, W[FLAGS.task_index]) + b[FLAGS.task_index]

  # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 10])

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))



        w = (W[(FLAGS.task_index - 1) % num_ps] + W[(FLAGS.task_index + 1) % num_ps] - W[FLAGS.task_index])
        #loss =  w * w / 12.0 / FLAGS.lr + cross_entropy

        opt = tf.train.GradientDescentOptimizer(FLAGS.lr)
        #opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate = num_worker, total_num_replicas = num_worker+2)
        #train_step = opt.minimize(loss, var_list=[W[FLAGS.task_index], b[FLAGS.task_index]])
        # all_variables()

        grads_and_vars = opt.compute_gradients(cross_entropy, [W[FLAGS.task_index], b[FLAGS.task_index]])
        
        new_gv0 = (grads_and_vars[0][0]-(W[(FLAGS.task_index - 1) % num_ps] + W[(FLAGS.task_index + 1) % num_ps] - 2*W[FLAGS.task_index])/(3*FLAGS.lr*1.0), grads_and_vars[0][1])
        new_gv1 = (grads_and_vars[1][0]-(b[(FLAGS.task_index - 1) % num_ps] + b[(FLAGS.task_index + 1) % num_ps] - 2*b[FLAGS.task_index])/(3*FLAGS.lr*1.0), grads_and_vars[1][1]) 
	new_gv=list()
	new_gv.append(new_gv0)
	new_gv.append(new_gv1) 
        train_step = opt.apply_gradients(new_gv)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #init_token_op = opt.get_init_tokens_op()
        #chief_queue_runner = opt.get_chief_queue_runner()

        saver = tf.train.Saver()     
        init_op = tf.global_variables_initializer()


        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/mnt/ds3lab/litian/logs",
                             init_op=init_op, 
                             saver=saver)

        with sv.managed_session(server.target) as sess:
            for i in range(10000):
  #              batch_xs, batch_ys = mnist.train.next_batch(100)
                image_batch, label_batch = train_dataset.next_batch(60, random_crop=True)   
                sess.run(train_step, feed_dict={x: image_batch, y_: label_batch})

                if i % 5 == 0:
                    train_accuracy = sess.run(accuracy,feed_dict={x: image_batch, y_: label_batch})
                    train_loss = sess.run(cross_entropy, feed_dict={x: image_batch, y_: label_batch})
                    localtime = time.asctime(time.localtime(time.time()))
                    print (localtime)
                    print("step %d, training accuracy %g, training loss %g" % (i, train_accuracy, train_loss))
	        if i % 25 == 0 :
		    val_accuracy = sess.run(accuracy, feed_dict={x: val_images, y_: val_labels})
                    print("validation set accuracy %g" % val_accuracy)
            #print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

        sv.stop()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/mnt/ds3lab/litian/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--job_name', type=str, help='either ps or worker')
  parser.add_argument('--task_index', type=int, help='task index, starting from 0')
  parser.add_argument('--lr', type=float, help='learning rate for SGD')

  FLAGS, unparsed = parser.parse_known_args() 
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


