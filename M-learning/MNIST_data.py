import tensorflow as tf
print(tf.__version__)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

print(type(mnist))
# numbers of traing images
print(mnist.train.num_examples)

import matplotlib.pyplot as plt

single_image = mnist.train.images[1].reshape(28,28)

plt.imshow(single_image)
# plt.show()

#placeholders
x = tf.placeholder(tf.float32,shape=[None,784])
#vareibles
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#create graph operations
y = tf.matmul(x,W) + b

#loss function
y_true = tf.placeholder(tf.float32,[None,10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y))
#optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)
#creat session
init = tf.global_variables_initializer()

#session
with tf.Session() as sess:
    sess.run(init)

    for step in range(1000):

        batch_x, batch_y = mnist.train.next_batch(100)

        sess.run(train,feed_dict={x:batch_x,y_true:batch_y})

    #Evaluate the model
    correct_pridiction = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))
    
    #bool data to float
    acc = tf.reduce_mean(tf.cast(correct_pridiction,tf.float32))

    #predicted [3,4]
    print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))