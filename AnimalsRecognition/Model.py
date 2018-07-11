import tensorflow as tf


class Model():

    def __init__(self):
        return

    def createCompGraph(self):
        self.imageSize = 100
        self.numClasses = 2
        self.input = tf.placeholder(dtype=tf.int32, shape=[None, self.imageSize, self.imageSize], name="input")
        self.output = tf.placeholder(dtype=tf.int32, shape=[None, self.numClasses], name="output")

        # Todo 2 specify the model
        inputImage = tf.reshape(self.input, shape=[-1, self.imageSize, self.imageSize, 1])

        conv1 = tf.layers.conv2d(inputImage, 32, kernel_constraint=[5, 5], strides=(2, 2), padding="SAME")
        pool1 = tf.layers.average_pooling2d(conv1, pool_size=[2, 2], strides=[2, 2])

        conv2 = tf.layers.conv2d(pool1, 64, kernel_size=[5, 5], strides=[2, 2])
        pool2 = tf.layers.average_pooling2d(conv2, pool_size=[2, 2], strides=[2, 2])

        dense = tf.layers.Dense(pool2, units=self.numClasses)

        self.logits = tf.nn.softmax(dense)
        self.loss = tf.losses.softmax_cross_entropy(self.logits, self.output)
        self.accuracy=tf.reduce_mean(
            tf.case(
                    tf.equal(tf.arg_max(self.logits),tf.arg_max(self.output))
                ,tf.int8
            )
        )

        self.optimizer=tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)

        return

    def intializeModel(self):
        self.sess=tf.InteractiveSession()
        tf.initialize_all_variables().run()
        return

    def train(self,images,output):

        _,acc,lo=self.sess.run([self.optimizer,self.accuracy,self.loss],feed_dict={
            self.input:images,
            self.output:output
        })

        return acc,lo

    def test(self,images,output):
        acc, lo = self.sess.run([ self.accuracy, self.loss], feed_dict={
            self.input: images,
            self.output: output
        })

        return acc, lo

    def predict(self,images):
        labels=self.sess.run([self.logits],feed_dict={
            self.input: images
        })
        return labels;