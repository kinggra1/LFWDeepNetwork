from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()



OUTPUT_SIZE = 10575 # number of faces we are classifying
INPUT_WIDTH = 100
INPUT_HEIGHT = 100
DROPOUT_RATE = 0.4



# initialize weight variables with random, small, positive values
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def avg_pool_7x7(x):
  return tf.nn.avg_pool(x, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='SAME')

x = tf.placeholder(tf.float32, shape=[None, INPUT_WIDTH*INPUT_HEIGHT])
y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])

#W = tf.Variable(tf.zeros([784,10]))
#b = tf.Variable(tf.zeros([10]))

#sess.run(tf.initialize_all_variables())
#y = tf.nn.softmax(tf.matmul(x,W) + b)
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#for i in range(1000):
#  batch = mnist.train.next_batch(100)
#  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# shape represents patch size x/y, num input channels, num output channels
#W_conv1 = weight_variable([5, 5, 1, 32])
W_conv11 = weight_variable([3, 3, 1, 32])
b_conv11 = bias_variable([32])
W_conv12 = weight_variable([3, 3, 1, 64])
b_conv12 = bias_variable([64])

x_image = tf.reshape(x, [-1, INPUT_WIDTH, INPUT_HEIGHT, 1])

h_conv11 = tf.nn.relu(conv2d(x_image, W_conv11) + b_conv11)
h_conv12 = tf.nn.relu(conv2d(h_conv11, W_conv12) + b_conv12)
h_pool1 = max_pool_2x2(h_conv12)

W_conv21 = weight_variable([3, 3, 64, 64])
b_conv21 = bias_variable([64])
W_conv22 = weight_variables([3, 3, 64, 128])
b_conv22 = bias_variable([128])

h_conv21 = tf.nn.relu(conv2d(h_pool1, W_conv21) + b_conv21)
h_conv22 = tf.nn.relu(conv2d(h_conv21, W_conv22) + b_conv22)
h_pool2 = max_pool_2x2(h_conv12)

W_conv31 = weight_variable([3, 3, 128, 96])
b_conv31 = bias_variable([96])
W_conv32 = weight_variables([3, 3, 96, 192])
b_conv32 = bias_variable([192])

h_conv31 = tf.nn.relu(conv2d(h_pool2, W_conv31) + b_conv31)
h_conv32 = tf.nn.relu(conv2d(h_conv31, W_conv32) + b_conv32)
h_pool3 = max_pool_2x2(h_conv32)

W_conv41 = weight_variable([3, 3, 192, 128])
b_conv41 = bias_variable([128])
W_conv42 = weight_variables([3, 3, 128, 256])
b_conv42 = bias_variable([256])

h_conv41 = tf.nn.relu(conv2d(h_pool3, W_conv41) + b_conv41)
h_conv42 = tf.nn.relu(conv2d(h_conv41, W_conv42) + b_conv42)
h_pool4 = max_pool_2x2(h_conv42)

W_conv51 = weight_variable([3, 3, 256, 160])
b_conv51 = bias_variable([160])
W_conv52 = weight_variables([3, 3, 160, 320])
b_conv52 = bias_variable([320])

h_conv51 = tf.nn.relu(conv2d(h_pool4, W_conv51) + b_conv51)
h_conv52 = tf.nn.relu(conv2d(h_conv51, W_conv52) + b_conv52)
h_pool5 = avg_pool_7x7(h_conv52)

keep_prob = 1-DROPOUT_RATE
h_dropout = tf.nn.dropout(h_pool5, keep_prob)



#W_fc1 = weight_variable(



#W_fc1 = weight_variable([7 * 7 * 64, 1024])
#b_fc1 = bias_variable([1024])

#h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
#h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#keep_prob = tf.placeholder(tf.float32)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#W_fc2 = weight_variable([1024, 10])
#b_fc2 = bias_variable([10])

#y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

logfile = open("teststats.txt", 'w')

for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    logfile.write("step %d, training accuracy %g\n"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


test_result = accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

print("test accuracy %g"%test_result)

logfile.write("test accuracy %g\n"%test_result)

logfile.close()
