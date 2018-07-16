import tensorflow as tf
import json


class Model():

    def __init__(self):
        with open("Conf/dataset.json") as file:
            self.config = json.load(file)
            self.imageSize = self.config["imageSize"]
            self.numClasses = len(self.config["labels"])
            self.channels = self.config["channels"]
        return

    def createCompGraph(self):
        # todo improve the model. model has staggering accuracy at 50 percent
        self.input = tf.placeholder(tf.float32, shape=[None, self.imageSize, self.imageSize, self.channels],
                                    name="input")
        self.output = tf.placeholder(tf.float32, shape=[None, self.numClasses], name="output")

        inputImage = tf.reshape(self.input, shape=[-1, self.imageSize, self.imageSize, self.channels])

        conv1 = tf.layers.conv2d(inputImage, 8, kernel_size=[5, 5], strides=(2, 2), padding="SAME",activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=[2, 2])

        conv2=tf.layers.conv2d(pool1,64,5,activation=tf.nn.relu,padding="SAME")
        pool2=tf.layers.max_pooling2d(conv2,[2,2],strides=2)

        conv3=tf.layers.conv2d(pool2,128,5,strides=2,activation=tf.nn.relu)
        pool3=tf.layers.max_pooling2d(conv3,pool_size=[2,2],strides=[2,2])


        flat = tf.layers.flatten(pool3)

        dropout = tf.layers.dropout(flat, self.config["model"]["dropout"])

        dense = tf.layers.dense(dropout, units=self.numClasses)

        self.logits = tf.nn.softmax(dense)

        self.loss = tf.losses.softmax_cross_entropy(logits=self.logits, onehot_labels=self.output)

        self.accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.arg_max(self.logits, 1), tf.arg_max(self.output, 1))
                , tf.float32
            )
        )

        self.learningRate = self.config["model"]["learningRate"]

        self.optimizer = tf.train.GradientDescentOptimizer(self.learningRate).minimize(self.loss)

        #summary items
        tf.summary.histogram("conv1", conv1)
        tf.summary.histogram("pool1", pool1)
        tf.summary.histogram("conv2", conv2)
        tf.summary.histogram("pool2", pool2)
        tf.summary.histogram("conv3", conv3)
        tf.summary.histogram("pool3", pool3)

        return

    def intializeModel(self):
        self.sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()
        self.trainWriter=tf.summary.FileWriter("./logs/1/train",self.sess.graph)
        self.trainCount=0
        return

    def train(self, images, output):
        merge=tf.summary.merge_all()
        summary,_, acc, lo = self.sess.run([merge,self.optimizer, self.accuracy, self.loss], feed_dict={
            self.input: images,
            self.output: output
        })
        self.trainWriter.add_summary(summary,self.trainCount)
        self.trainCount+=1
        return acc, lo

    def test(self, images, output):
        acc, lo = self.sess.run([self.accuracy, self.loss], feed_dict={
            self.input: images,
            self.output: output
        })

        return acc, lo

    def predict(self, images):
        labels = self.sess.run([self.logits], feed_dict={
            self.input: images
        })
        return labels

    def visualize(self, images):
        layer = self.sess.run([self.pool1], feed_dict={
            self.input: images
        })
        return layer
