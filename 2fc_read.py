from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import time
from input_data_cifar  import create_train_datasets
from input_data_cifar import create_test_datasets
import tensorflow as tf

FLAGS = None

NUM_IMAGES = 5000
num_classes=10


def main(_):

  # cluster specification
  
  parameter_servers = ["sgs-gpu-03:2225"]
  #workers = ["spaceml1:2223", "spaceml1:2224", "spaceml1:2225", "spaceml1:2226"]
  
#  parameter_servers = ["sgs-gpu-03:2224"]
  #workers = ["sgs-gpu-02:2223", "sgs-gpu-02:2224","sgs-gpu-03:2222", "sgs-gpu-03:2223"]
  workers = ["sgs-gpu-03:2226"]
  num_workers = len(workers)

  cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

  #local server, either ps or worker
  server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
  #data_sets = data_helpers.load_data() 

  num_classes=10

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
  # Create the model
        x = tf.placeholder(tf.float32, shape=[None, 224* 224, 3])
        x_reshape = tf.reshape(x, [-1, 224*224*3]) 
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
        

        w1 = tf.get_variable(name='w1',shape=[224*224*3,40], initializer=tf.truncated_normal_initializer(stddev=1.0/np.sqrt(float(224*224*3))),regularizer=tf.contrib.layers.l2_regularizer(0.1))
        print (w1)
	b1 = tf.Variable(tf.zeros([40]))
        h1 = tf.nn.relu(tf.matmul(x_reshape, w1)+b1)
	print (h1)

        
        w2 = tf.get_variable(name='w2',shape=[40,10], initializer=tf.truncated_normal_initializer(stddev=1.0/np.sqrt(float(4096))),
          regularizer=tf.contrib.layers.l2_regularizer(0.1))
        print (w2)
	b2 = tf.Variable(tf.zeros([10]))
        y = tf.matmul(h1, w2) + b2   
        print (y)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
     
        train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
     
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
     
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
#        saver = tf.train.Saver()     
        init_op = tf.global_variables_initializer()

        list_ = []
        for line in open("/mnt/ds3lab/litian/input_data/cifar10/label3.txt"):
            list_.append(['a', line.strip('\n')])
        classes = np.array(list_)
        print (len(classes))

        train_dataset, mean = create_train_datasets(FLAGS.task_index, classes[:, 1], num_samples=50)
#        val_dataset = create_test_datasets(classes[:, 1], mean)

 #       val_images, val_labels = val_dataset.next_batch(10000)

        num_classes = len(classes)
        print (num_classes)

        #sess = tf.Session()
        #if FLAGS.task_index==0:
            #sess.run(tf.initialize_all_variables())
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/mnt/ds3lab/litian/logs_7",
                             init_op=init_op)

        with sv.managed_session(server.target) as sess:
            begin = time.time()
            for i in range(50000):
		print (i)
		t1 = time.time()
		image_batch, label_batch = train_dataset.next_batch(64, random_crop=True)
                t2 = time.time()
		print ("time for loading a batch is %g" % (t2-t1))
                sess.run(w1, feed_dict={x: image_batch, y_: label_batch})
                t3 = time.time()
		print ("time for training a batch is %g" % (t3-t2))
		if i % 50 == 0 :
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

'''
export CUDA_VISIBLE_DEVICES=2
python alexnet_centralized_without_test.py --job_name=ps --task_index=0 

export CUDA_VISIBLE_DEVICES=0
python alexnet_centralized_without_test.py --job_name=worker --task_index=0 

export CUDA_VISIBLE_DEVICES=1
python alexnet_centralized_without_test.py --job_name=worker --task_index=1 

export CUDA_VISIBLE_DEVICES=3
python alexnet_centralized_without_test.py --job_name=worker --task_index=2 

export CUDA_VISIBLE_DEVICES=4
python alexnet_centralized_without_test.py --job_name=worker --task_index=3 
--------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0
python alexnet_centralized_without_test.py --job_name=ps --task_index=0 

export CUDA_VISIBLE_DEVICES=0
python alexnet_centralized_without_test.py --job_name=worker --task_index=0 

export CUDA_VISIBLE_DEVICES=1
python alexnet_centralized_without_test.py --job_name=worker --task_index=1 

export CUDA_VISIBLE_DEVICES=0
python alexnet_centralized_without_test.py --job_name=worker --task_index=2 

export CUDA_VISIBLE_DEVICES=1
python alexnet_centralized_without_test.py --job_name=worker --task_index=3 

'''



