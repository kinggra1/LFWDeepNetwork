import build_image_data as builder
import tensorflow as tf
sess = tf.InteractiveSession()

import numpy as np
import time

import directory

#files = directory.get_cropped_CASIA_files()

logfile = '10575train_log_newqsub.txt'
trainingfile = './test/train-00000-of-00001'
testingfile = './26faces_split/validation-00000-of-00001'
graph_dir = './10575trained_graph_distorted_qsub'

OUTPUT_SIZE = 10575  # number of faces we are classifying
INPUT_WIDTH = 110
INPUT_HEIGHT = 110
CHANNELS = 3
RANDOM_DISTORTIONS = True
DROPOUT_KEEP_RATE = 0.6

ENQUEUE_THREADS = 4
TRAINING_STEPS = 300000
BATCH_SIZE = 100
MIN_AFTER_DEQUE = 1000
CAPACITY = MIN_AFTER_DEQUE + 3*BATCH_SIZE


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
  return tf.nn.avg_pool(x, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID')

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


def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def random_distortions(image, thread_id=0):
  # Randomly flip horizontally.
  with tf.name_scope("flip_horizontal", values=[image]):
    image = tf.image.random_flip_left_right(image)

  # Randomly distort the colors based on thread id.
  color_ordering = thread_id
  with tf.name_scope("distort_color", values=[image]):
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=16. / 255.)
      image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
      image = tf.image.random_hue(image, max_delta=0.032)
      image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
    elif color_ordering == 1:
      image = tf.image.random_brightness(image, max_delta=16. / 255.)
      image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
      image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
      image = tf.image.random_hue(image, max_delta=0.032)

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def read_and_decode(filename_queue, thread_id=0):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  feature_map = {

      'image/height': tf.FixedLenFeature([], dtype=tf.int64),
      'image/width': tf.FixedLenFeature([], dtype=tf.int64),
      'image/filename': tf.FixedLenFeature([], dtype=tf.string),
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([], dtype=tf.int64,
                                              default_value=1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }

  features = tf.parse_single_example(
    serialized_example,
    feature_map,
    #dense_keys=['image_raw', 'label'],
    # Defaults are not specified since both keys are required.
    #dense_types=[tf.string, tf.int64]
    )

  # Convert from a scalar string tensor (whose single string has
  #image = tf.decode_raw(features['image/encoded'], tf.uint8)
  image = tf.image.decode_jpeg(features['image/encoded'], channels=CHANNELS)

  #print(image.get_shape().as_list())
  image = tf.reshape(image, [INPUT_WIDTH, INPUT_HEIGHT,CHANNELS])
  #image.set_shape([INPUT_WIDTH*INPUT_HEIGHT*CHANNELS])

  # OPTIONAL: Could reshape into a 28x28 image and apply distortions
  # here.  Since we are not applying any distortions in this
  # example, and the next step expects the image to be flattened
  # into a vector, we don't bother.

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  image = tf.cast(image, tf.float32) * (1. / 255)

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  #label = tf.cast(features['image/class/label'], tf.int32)
  label = tf.one_hot(features['image/class/label'], OUTPUT_SIZE, axis=-1)
  #label = tf.reshape(label, [OUTPUT_SIZE])
  if (RANDOM_DISTORTIONS):
    image = random_distortions(image, thread_id)

  return image-0.5, label


# training data to plug into graph
filename_queue = tf.train.string_input_producer(
    [trainingfile], num_epochs=None)


image_queue = tf.RandomShuffleQueue(CAPACITY, MIN_AFTER_DEQUE, [tf.float32, tf.float32], shapes=[[INPUT_WIDTH, INPUT_HEIGHT, CHANNELS], [OUTPUT_SIZE]])

#image, label = read_and_decode(filename_queue)
#enqueue_op = image_queue.enqueue([image, label])

	#tf.train.shuffle_batch(
      #[image, label], batch_size=BATCH_SIZE, capacity=CAPACITY,
      #min_after_dequeue=MIN_AFTER_DEQUE))

enqueue_ops = []
for i in range(ENQUEUE_THREADS):
  image, label = read_and_decode(filename_queue, i)
  enqueue_op = image_queue.enqueue([image, label])
  enqueue_ops.append(enqueue_op)

qr = tf.train.QueueRunner(image_queue, enqueue_ops)

# are we training or testing?
#training = tf.placeholder(tf.bool)

# set up an operator to grab training or validation images
#x, y_ = tf.cond(training, lambda: image_queue.dequeue_many(BATCH_SIZE), lambda: tf.train.shuffle_batch([image, label], batch_size=100, capacity=200, min_after_dequeue=50))


#images_train, labels_train = image_queue.dequeue_many(BATCH_SIZE)
#images_test, labels_test = tf.train.shuffle_batch([image, label], batch_size=100, capacity=200, min_after_dequeue=50)
#x = tf.placeholder_with_default(images_train, shape=[BATCH_SIZE, INPUT_WIDTH, INPUT_HEIGHT, CHANNELS])
#y_ = tf.placeholder_with_default(labels_train, shape=[BATCH_SIZE, OUTPUT_SIZE])


# Before we used placeholders and a feed dict, lets update to just plug the tensors right in now
#x = tf.placeholder(tf.float32, shape=[None, INPUT_WIDTH*INPUT_HEIGHT])
#y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])
#image, label = read_and_decode(filename_queue)
#x, y_ = tf.train.batch([image, label], 1)

x, y_ = image_queue.dequeue_many(BATCH_SIZE)

#x, y_  = tf.train.shuffle_batch(
#      [image, label], batch_size=BATCH_SIZE, capacity=CAPACITY,
#      min_after_dequeue=MIN_AFTER_DEQUE)

#shape represents patch size x/y, num input channels, num output channels
W_conv11 = weight_variable([3, 3, 3, 32])
b_conv11 = bias_variable([32])
W_conv12 = weight_variable([3, 3, 32, 64])
b_conv12 = bias_variable([64])

x_image = tf.reshape(x, [-1, INPUT_WIDTH, INPUT_HEIGHT, CHANNELS])

h_conv11 = tf.nn.relu(conv2d(x_image, W_conv11) + b_conv11)
h_conv12 = tf.nn.relu(conv2d(h_conv11, W_conv12) + b_conv12)
h_pool1 = max_pool_2x2(h_conv12)

W_conv21 = weight_variable([3, 3, 64, 64])
b_conv21 = bias_variable([64])
W_conv22 = weight_variable([3, 3, 64, 128])
b_conv22 = bias_variable([128])

h_conv21 = tf.nn.relu(conv2d(h_pool1, W_conv21) + b_conv21)
h_conv22 = tf.nn.relu(conv2d(h_conv21, W_conv22) + b_conv22)
h_pool2 = max_pool_2x2(h_conv22)

W_conv31 = weight_variable([3, 3, 128, 96])
b_conv31 = bias_variable([96])
W_conv32 = weight_variable([3, 3, 96, 192])
b_conv32 = bias_variable([192])

h_conv31 = tf.nn.relu(conv2d(h_pool2, W_conv31) + b_conv31)
h_conv32 = tf.nn.relu(conv2d(h_conv31, W_conv32) + b_conv32)
h_pool3 = max_pool_2x2(h_conv32)

W_conv41 = weight_variable([3, 3, 192, 128])
b_conv41 = bias_variable([128])
W_conv42 = weight_variable([3, 3, 128, 256])
b_conv42 = bias_variable([256])

h_conv41 = tf.nn.relu(conv2d(h_pool3, W_conv41) + b_conv41)
h_conv42 = tf.nn.relu(conv2d(h_conv41, W_conv42) + b_conv42)
h_pool4 = max_pool_2x2(h_conv42)

W_conv51 = weight_variable([3, 3, 256, 160])
b_conv51 = bias_variable([160])
W_conv52 = weight_variable([3, 3, 160, 320])
b_conv52 = bias_variable([320])

h_conv51 = tf.nn.relu(conv2d(h_pool4, W_conv51) + b_conv51)
h_conv52 = conv2d(h_conv51, W_conv52) + b_conv52
h_pool5 = avg_pool_7x7(h_conv52)
h_pool5_reshape = tf.reshape(h_pool5, [-1, 320])

#W_fc1 = weight_variable([320, 320])
#b_fc1 = bias_variable([320])

#h_fc1 = tf.nn.relu(tf.matmul(h_pool5_reshape, W_fc1) + b_fc1)

# END OF FEATURE GENERATION

# variable dropout for training vs testing
keep_prob = tf.placeholder(tf.float32)
h_dropout = tf.nn.dropout(h_pool5_reshape, keep_prob, name='features')

W_fc2 = weight_variable([320, OUTPUT_SIZE])
b_fc2 = bias_variable([OUTPUT_SIZE])

#y_conv = tf.nn.softmax(tf.matmul(h_dropout, W_fc2) + b_fc2)
y_conv = tf.matmul(h_dropout, W_fc2) + b_fc2

print(y_conv.get_shape().as_list())
print(y_.get_shape().as_list())

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.to_float(y_) * tf.log(y_conv), 1))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=tf.to_float(y_)))

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#sess.run(tf.initialize_all_variables())



def datalog(message):
  with open(logfile, 'a') as f:
    f.write(message)


# Create saver for saving graph variables/weights
saver = tf.train.Saver(max_to_keep=200)

# Create the graph, etc.
init_op = tf.global_variables_initializer()

# Initialize the variables (like the epoch counter).
sess.run(init_op)
sess.run(tf.local_variables_initializer())



if __name__ == "__main__":

  # Start input enqueue threads.
  coord = tf.train.Coordinator()
  enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)


  datalog("step, training accuracy, validation accuracy\n")


  print("Let's get ready to traaaaaaain!")
  try:
    if 1: #while not coord.should_stop():
      # Run training steps or whatever

      for i in range(TRAINING_STEPS):
        train_step.run(feed_dict={keep_prob: DROPOUT_KEEP_RATE})
        #print(tf.argmax(y_,1).eval(feed_dict={keep_prob: 0.5}))
        #print(tf.argmax(y_conv,1).eval(feed_dict={keep_prob: 0.5}))
        if i%100 == 0:
          train_accuracy = accuracy.eval(feed_dict={
             keep_prob: 1.0})#, x:images_test.eval(), y_:labels_test.eval()})
          print("step %d, training accuracy %g"%(i, train_accuracy))
          datalog("%d, %g\n"%(i, train_accuracy))
        if i%10000 == 0:
          saver.save(sess, graph_dir+'/network.ckpt', global_step=i)

  except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
  finally:
      # When done, ask the threads to stop.
      coord.request_stop()
  
  # Wait for threads to finish.
  coord.join(threads)
  
  tf.train.write_graph(sess.graph_def, graph_dir, 'network.pbtxt')
  saver.save(sess, graph_dir+'/network.ckpt', global_step=TRAINING_STEPS)
  #tf.train.export_meta_graph('./trained_graph/network.meta')

  test_result = accuracy.eval(feed_dict={
      keep_prob: 1.0})

  print("test accuracy %g"%test_result)

  datalog("test accuracy %g\n"%test_result)



sess.close()
