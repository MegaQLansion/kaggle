import tensorflow.examples.tutorials.mnist.input_data as input_data 
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True) 

import tensorflow as tf
import skflow


x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)#predict result with softmax kernel
y_ = tf.placeholder("float", [None,10])#real result
cross_entropy = -tf.reduce_sum(y_*tf.log(y))#cost function
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)#gradient descent to minimize cost function
#initialization
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
#train with each batch of 100 datas
for i in range(1000):
  batch_xs, batch_ys =mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
#evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))#compare y and y_

print ('The accuracy is',sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))