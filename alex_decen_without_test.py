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
import data_helpers

FLAGS = None

NUM_IMAGES = 5000
num_classes=10


# one ps and four workers

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def relu_weight_variable(shape):
    assert len(shape) is 2
    input_size = shape[0]
    initial = tf.truncated_normal(shape, stddev=np.sqrt(2.0 / input_size))
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, strides):
    return conv_batch_normalization(tf.nn.conv2d(x, W, strides=strides, padding='SAME'))

def conv_batch_normalization(x):
    mean, variance = tf.nn.moments(x, axes=[0, 1, 2])
    return tf.nn.batch_normalization(x, mean, variance, None, None, 0.0001)

def fc_batch_normalization(x):
    mean, variance = tf.nn.moments(x, axes=[0])
    return tf.nn.batch_normalization(x, mean, variance, None, None, 0.0001)

def main(_):

  # cluster specification
  
  #parameter_servers = ["spaceml1:2222", "spaceml1:2223", "spaceml1:2224", "spaceml1:2225"]
  #workers = ["spaceml1:2226", "spaceml1:2227", "spaceml1:2228", "spaceml1:2229"]
  
  parameter_servers = ["spaceml1:2222", "spaceml1:2223", "spaceml1:2224", "spaceml1:2225"]
  workers = ["sgs-gpu-02:2222", "sgs-gpu-02:2223","sgs-gpu-03:2222", "sgs-gpu-03:2223"]

  num_workers = len(workers)
  num_servers = num_workers

  cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

  #local server, either ps or worker
  server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
  #data_sets = data_helpers.load_data() 

  W_conv1=[]
  b_conv1=[]
  W_conv2=[]
  b_conv2=[]
  W_conv3=[]
  b_conv3=[]
  W_conv4=[]
  b_conv4=[]
  W_conv5=[]
  b_conv5=[]
  W_fc1=[]
  b_fc1=[]
  W_fc2=[]
  b_fc2=[]
  W_fc3=[]
  b_fc3=[]
  for n in range(0, num_servers):
      W_conv1.append(0)
      b_conv1.append(0)
      W_conv2.append(0)
      b_conv2.append(0)
      W_conv3.append(0)
      b_conv3.append(0)
      W_conv4.append(0)
      b_conv4.append(0)
      W_conv5.append(0)
      b_conv5.append(0)
      W_fc1.append(0)
      b_fc1.append(0)
      W_fc2.append(0)
      b_fc2.append(0)
      W_fc3.append(0)
      b_fc3.append(0)



  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    for i in range(num_servers):
        with tf.device("/job:ps/task:%d" % i):
            W_conv1[i] = weight_variable([11,11,3,96])
            b_conv1[i] = bias_variable([96])
            W_conv2[i] = weight_variable([5,5,96,256])
            b_conv2[i] = bias_variable([256])
            W_conv3[i] = weight_variable([3,3,256,384])
            b_conv3[i] = bias_variable([384])
            W_conv4[i] = weight_variable([3,3,384,384])
            b_conv4[i] = bias_variable([384])
            W_conv5[i] = weight_variable([3,3,384,256])
            b_conv5[i] = bias_variable([256])
  
            W_fc1[i] = relu_weight_variable([7*7*256, 4096])
            b_fc1[i] = bias_variable([4096])
            W_fc2[i] = relu_weight_variable([4096,4096])
            b_fc2[i] = bias_variable([4096])
            W_fc3[i] = relu_weight_variable([4096,10])
            b_fc3[i] = bias_variable([10])

    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
  # Create the model
        num_classes=10
        x = tf.placeholder(tf.float32, shape=[None, 224* 224, 3])
        y_ = tf.placeholder(tf.float32, shape=[None,10])
        keep_prob = tf.placeholder(tf.float32)
        #x_ = tf.image.resize_images(x,[224,224], method=tf.image.ResizeMethod.BICUBIC)
        x_reshaped = tf.reshape(x, [-1, 224, 224, 3])  
    
        h_conv1 = tf.nn.relu(conv2d(x_reshaped, W_conv1[FLAGS.task_index], [1, 4, 4, 1]) + b_conv1[FLAGS.task_index])   
        norm1 = tf.nn.lrn(h_conv1, 5, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)   
        max_pool1 = tf.nn.max_pool(norm1, ksize = [1,3,3,1], strides = [1,2,2,1], padding='SAME')
 
        h_conv2 = tf.nn.relu(conv2d(max_pool1, W_conv2[FLAGS.task_index], [1, 1, 1, 1]) + b_conv2[FLAGS.task_index]) 
        norm2 = tf.nn.lrn(h_conv2, 5, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)    
        h_pool2 = tf.nn.max_pool(norm2, ksize = [1,3,3,1], strides = [1,2,2,1], padding='SAME') # 
  
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3[FLAGS.task_index], [1, 1, 1, 1]) + b_conv3[FLAGS.task_index])

        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4[FLAGS.task_index], [1, 1, 1, 1]) + b_conv4[FLAGS.task_index]) # 
      
        h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5[FLAGS.task_index], [1, 1, 1, 1]) + b_conv5[FLAGS.task_index])     
        max_pooling5 = tf.nn.max_pool(h_conv5, ksize = [1,3,3,1], strides = [1,2,2,1], padding='SAME') # 
   
        h_conv5_flat = tf.reshape(max_pooling5, [-1, 7*7*256])  
  
        h_fc1 = tf.nn.relu(fc_batch_normalization(tf.matmul(h_conv5_flat, W_fc1[FLAGS.task_index]) + b_fc1[FLAGS.task_index]))     
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # 
     
        h_fc2 = tf.nn.relu(fc_batch_normalization(tf.matmul(h_fc1_drop, W_fc2[FLAGS.task_index]) + b_fc2[FLAGS.task_index]))   
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob) # 
    
        y_score = fc_batch_normalization(tf.matmul(h_fc2_drop, W_fc3[FLAGS.task_index]) + b_fc3[FLAGS.task_index])   
        y_logit = tf.nn.softmax(y_score)
     
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_score, labels=y_))

        opt = tf.train.GradientDescentOptimizer(FLAGS.lr)
        grads_and_vars = opt.compute_gradients(cross_entropy,
          [W_conv1[FLAGS.task_index], b_conv1[FLAGS.task_index], 
           W_conv2[FLAGS.task_index], b_conv2[FLAGS.task_index],
           W_conv3[FLAGS.task_index], b_conv3[FLAGS.task_index],
           W_conv4[FLAGS.task_index], b_conv4[FLAGS.task_index],
           W_conv5[FLAGS.task_index], b_conv5[FLAGS.task_index],
           W_fc1[FLAGS.task_index], b_fc1[FLAGS.task_index],
           W_fc2[FLAGS.task_index], b_fc2[FLAGS.task_index],
           W_fc3[FLAGS.task_index], b_fc3[FLAGS.task_index]])

        new_gv=[]
        new_gv.append((grads_and_vars[0][0]-(W_conv1[(FLAGS.task_index - 1) % num_servers] + W_conv1[(FLAGS.task_index - 1) % num_servers]
          - 2 * W_conv1[FLAGS.task_index])/(3*FLAGS.lr*1.0), grads_and_vars[0][1]))

        new_gv.append((grads_and_vars[1][0]-(b_conv1[(FLAGS.task_index - 1) % num_servers] + b_conv1[(FLAGS.task_index - 1) % num_servers]
          - 2 * b_conv1[FLAGS.task_index])/(3*FLAGS.lr*1.0), grads_and_vars[1][1]))

        new_gv.append((grads_and_vars[2][0]-(W_conv2[(FLAGS.task_index - 1) % num_servers] + W_conv2[(FLAGS.task_index - 1) % num_servers]
          - 2 * W_conv2[FLAGS.task_index])/(3*FLAGS.lr*1.0), grads_and_vars[2][1]))

        new_gv.append((grads_and_vars[3][0]-(b_conv2[(FLAGS.task_index - 1) % num_servers] + b_conv2[(FLAGS.task_index - 1) % num_servers]
          - 2 * b_conv2[FLAGS.task_index])/(3*FLAGS.lr*1.0), grads_and_vars[3][1]))

        new_gv.append((grads_and_vars[4][0]-(W_conv3[(FLAGS.task_index - 1) % num_servers] + W_conv3[(FLAGS.task_index - 1) % num_servers]
          - 2 * W_conv3[FLAGS.task_index])/(3*FLAGS.lr*1.0), grads_and_vars[4][1]))

        new_gv.append((grads_and_vars[5][0]-(b_conv3[(FLAGS.task_index - 1) % num_servers] + b_conv3[(FLAGS.task_index - 1) % num_servers]
          - 2 * b_conv3[FLAGS.task_index])/(3*FLAGS.lr*1.0), grads_and_vars[5][1]))

        new_gv.append((grads_and_vars[6][0]-(W_conv4[(FLAGS.task_index - 1) % num_servers] + W_conv4[(FLAGS.task_index - 1) % num_servers]
          - 2 * W_conv4[FLAGS.task_index])/(3*FLAGS.lr*1.0), grads_and_vars[6][1]))

        new_gv.append((grads_and_vars[7][0]-(b_conv4[(FLAGS.task_index - 1) % num_servers] + b_conv4[(FLAGS.task_index - 1) % num_servers]
          - 2 * b_conv4[FLAGS.task_index])/(3*FLAGS.lr*1.0), grads_and_vars[7][1]))

        new_gv.append((grads_and_vars[8][0]-(W_conv5[(FLAGS.task_index - 1) % num_servers] + W_conv5[(FLAGS.task_index - 1) % num_servers]
          - 2 * W_conv5[FLAGS.task_index])/(3*FLAGS.lr*1.0), grads_and_vars[8][1]))

        new_gv.append((grads_and_vars[9][0]-(b_conv5[(FLAGS.task_index - 1) % num_servers] + b_conv5[(FLAGS.task_index - 1) % num_servers]
          - 2 * b_conv5[FLAGS.task_index])/(3*FLAGS.lr*1.0), grads_and_vars[9][1]))

        new_gv.append((grads_and_vars[10][0]-(W_fc1[(FLAGS.task_index - 1) % num_servers] + W_fc1[(FLAGS.task_index - 1) % num_servers]
          - 2 * W_fc1[FLAGS.task_index])/(3*FLAGS.lr*1.0), grads_and_vars[10][1]))

        new_gv.append((grads_and_vars[11][0]-(b_fc1[(FLAGS.task_index - 1) % num_servers] + b_fc1[(FLAGS.task_index - 1) % num_servers]
          - 2 * b_fc1[FLAGS.task_index])/(3*FLAGS.lr*1.0), grads_and_vars[11][1]))

        new_gv.append((grads_and_vars[12][0]-(W_fc2[(FLAGS.task_index - 1) % num_servers] + W_fc2[(FLAGS.task_index - 1) % num_servers]
          - 2 * W_fc2[FLAGS.task_index])/(3*FLAGS.lr*1.0), grads_and_vars[12][1]))
        
        new_gv.append((grads_and_vars[13][0]-(b_fc2[(FLAGS.task_index - 1) % num_servers] + b_fc2[(FLAGS.task_index - 1) % num_servers]
          - 2 * b_fc2[FLAGS.task_index])/(3*FLAGS.lr*1.0), grads_and_vars[13][1]))

        new_gv.append((grads_and_vars[14][0]-(W_fc3[(FLAGS.task_index - 1) % num_servers] + W_fc3[(FLAGS.task_index - 1) % num_servers]
          - 2 * W_fc3[FLAGS.task_index])/(3*FLAGS.lr*1.0), grads_and_vars[14][1]))
        
        new_gv.append((grads_and_vars[15][0]-(b_fc3[(FLAGS.task_index - 1) % num_servers] + b_fc3[(FLAGS.task_index - 1) % num_servers]
          - 2 * b_fc3[FLAGS.task_index])/(3*FLAGS.lr*1.0), grads_and_vars[15][1]))
     
        #train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

        train_step = opt.apply_gradients(list(new_gv))
     
        correct_prediction = tf.equal(tf.argmax(y_logit, 1), tf.argmax(y_,1))
     
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()     
        init_op = tf.global_variables_initializer()

        #sess = tf.Session()
        #if FLAGS.task_index==0:
            #sess.run(tf.initialize_all_variables())
        list_ = []
        for line in open("/mnt/ds3lab/litian/input_data/cifar10/label3.txt"):
            list_.append(['a', line.strip('\n')])
        classes = np.array(list_)
        print (len(classes))

        train_dataset, mean = create_train_datasets(FLAGS.task_index, classes[:, 1], num_samples=5000)
        
        num_classes = len(classes)
        print (num_classes)

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/mnt/ds3lab/litian/logs_4",
                             init_op=init_op)



        with sv.managed_session(server.target) as sess:
            begin = time.time()
            for i in range(50000):
                print (i)    
                t1 = time.time()               
                image_batch, label_batch = train_dataset.next_batch(64, random_crop=True)
                t2 = time.time()
                print ("time for loading a batch is %g" % (t2-t1))
                
                sess.run(train_step, feed_dict={x: image_batch, y_: label_batch, keep_prob: 0.5})
                if i % 50 == 0:
                    train_accuracy = sess.run(accuracy,feed_dict={x: image_batch, y_: label_batch, keep_prob: 1.0})
                    train_loss = sess.run(cross_entropy, feed_dict={x: image_batch, y_: label_batch, keep_prob: 1.0})
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
  parser.add_argument('--lr', type=float, help='learning rate')

  FLAGS, unparsed = parser.parse_known_args() 
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


# the best learning rate should be chosen in centralized setting

'''
export CUDA_VISIBLE_DEVICES=5
python alex_decen_without_test.py --job_name=ps --task_index=0 --lr=0.001

export CUDA_VISIBLE_DEVICES=2
python alex_decen_without_test.py --job_name=ps --task_index=1 --lr=0.001

export CUDA_VISIBLE_DEVICES=6
python alex_decen_without_test.py --job_name=ps --task_index=2 --lr=0.001

export CUDA_VISIBLE_DEVICES=7
python alex_decen_without_test.py --job_name=ps --task_index=3 --lr=0.001

export CUDA_VISIBLE_DEVICES=0
python alex_decen_without_test.py --job_name=worker --task_index=0 --lr=0.001

export CUDA_VISIBLE_DEVICES=1
python alex_decen_without_test.py --job_name=worker --task_index=1 --lr=0.001

export CUDA_VISIBLE_DEVICES=0
python alex_decen_without_test.py --job_name=worker --task_index=2 --lr=0.001

export CUDA_VISIBLE_DEVICES=1
python alex_decen_without_test.py --job_name=worker --task_index=3 --lr=0.001


'''



