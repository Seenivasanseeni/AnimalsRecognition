import tensorflow as tf
import json
from .tools import *

class Model():

    def __init__(self):
        with open("Conf/dataset.json") as file:
            self.config = json.load(file)
            self.imageSize = self.config["imageSize"]
            self.numClasses = self.config["numClasses"]
            self.channels = self.config["channels"]
        return

    def createCompGraph(self):
        # todo improve the model. model has staggering accuracy at 50 percent
        self.input = tf.placeholder(tf.float32, shape=[None, self.imageSize, self.imageSize, self.channels],
                                    name="input")
        self.output = tf.placeholder(tf.float32, shape=[None, self.numClasses], name="output")

        inputImage = tf.reshape(self.input, shape=[-1, self.imageSize, self.imageSize, self.channels])

        self.conv1=tf.layers.conv2d(inputImage,32,kernel_size=[5,5],strides=2,activation=tf.nn.relu)

        self.pool1=tf.layers.max_pooling2d(self.conv1,pool_size=[5,5],strides=2)


        flat=tf.layers.flatten(self.pool1)

        self.dense=tf.layers.dense(flat,units=self.numClasses,activation=tf.nn.relu)

        self.logits = tf.nn.softmax(self.dense)

        self.loss = tf.losses.softmax_cross_entropy(logits=self.logits, onehot_labels=self.output)



        self.accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.arg_max(self.logits, 1), tf.arg_max(self.output, 1))
                , tf.float32
            )
        )

        self.learningRate = self.config["model"]["learningRate"]

        self.optimizer = tf.train.GradientDescentOptimizer(self.learningRate).minimize(self.loss)

        # summary items
        tf.summary.histogram("loss", self.loss)
        tf.summary.histogram("accuracy", self.accuracy)
        return

    def intializeModel(self):
        self.sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()
        makeLogDir()
        self.trainWriter = tf.summary.FileWriter("./logs/1/train", self.sess.graph)
        self.trainCount = 0
        self.testWriter = tf.summary.FileWriter("./logs/1/test", self.sess.graph)
        self.testCount = 0

        return

    def train(self, images, output):
        merge = tf.summary.merge_all()
        summary, _, acc, lo = self.sess.run([merge, self.optimizer, self.accuracy, self.loss], feed_dict={
            self.input: images,
            self.output: output
        })
        self.trainWriter.add_summary(summary, self.trainCount)
        self.trainCount += 1
        return acc, lo


    def test(self, images, output):
        merge=tf.summary.merge_all()
        summary,acc, lo = self.sess.run([merge,self.accuracy, self.loss], feed_dict={
            self.input: images,
            self.output: output
        })
        self.testWriter.add_summary(summary,self.testCount)
        self.testCount+=1
        return acc, lo

    def predict(self, images):
        labels = self.sess.run([self.logits], feed_dict={
            self.input: images
        })
        return labels

    def visualize(self, images):
        layer = self.sess.run([self.visualizeMark], feed_dict={
            self.input: images
        })
        return layer
