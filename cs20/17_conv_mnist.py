import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import tensorflow as tf
import utils


def conv_relu(inputs, filters, k_size, stride, padding, scope_name):
    '''
    A method that does convolution + relu on inputs
    '''
    with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
        in_channels = inputs.shape[-1]
        kernel = tf.compat.v1.get_variable('kernel',
                                 [k_size, k_size, in_channels, filters],
                                 initializer=tf.truncated_normal_initializer())
        biases = tf.compat.v1.get_variable('biases',
                                 [filters],
                                 initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding=padding)
    return tf.nn.relu(conv + biases, name=scope.name)


def maxpool(inputs, ksize, stride, padding='VALID', scope_name='pool'):
    '''A method that does max pooling on inputs'''
    with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
        pool = tf.nn.max_pool2d(inputs,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding=padding)
    return pool


def fully_connected(inputs, out_dim, scope_name='fc'):
    '''
    A fully connected linear layer on inputs
    '''
    with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]
        w = tf.compat.v1.get_variable('weights', [in_dim, out_dim],
                            initializer=tf.truncated_normal_initializer())
        b = tf.compat.v1.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer(0.0))
        out = tf.matmul(inputs, w) + b
    return out


class ConvNet(object):
    def __init__(self):
        self.lr = 0.001
        self.batch_size = 128
        self.keep_prob = tf.constant(0.75)
        self.gstep = tf.Variable(0, dtype=tf.int32,
                                 trainable=False, name='global_step')
        self.n_classes = 10
        self.skip_step = 20
        self.n_test = 10000
        self.training = True

    def get_data(self):
        with tf.name_scope('data'):
            train_data, test_data = utils.get_mnist_dataset(self.batch_size)
            iterator = tf.compat.v1.data.Iterator.from_structure(
                tf.compat.v1.data.get_output_types(train_data),
                tf.compat.v1.data.get_output_shapes(train_data))
            img, self.label = iterator.get_next()
            self.img = tf.reshape(img, shape=[-1, 28, 28, 1])
            # reshape the image to make it work with tf.nn.conv2d

            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
            self.test_init = iterator.make_initializer(test_data)  # initializer for train_data

    def inference(self):
        conv1 = conv_relu(inputs=self.img,
                          filters=32,
                          k_size=5,
                          stride=1,
                          padding='SAME',
                          scope_name='conv1')
        pool1 = maxpool(conv1, 2, 2, 'VALID', 'pool1')
        conv2 = conv_relu(inputs=pool1,
                          filters=64,
                          k_size=5,
                          stride=1,
                          padding='SAME',
                          scope_name='conv2')
        pool2 = maxpool(conv2, 2, 2, 'VALID', 'pool2')
        feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
        pool2 = tf.reshape(pool2, [-1, feature_dim])
        fc = fully_connected(pool2, 1024, 'fc')
        dropout = tf.nn.dropout(tf.nn.relu(fc), rate=1-self.keep_prob, name='relu_dropout')
        self.logits = fully_connected(dropout, self.n_classes, 'logits')

    def loss(self):
        '''
        define loss function
        use softmax cross entropy with logits as the loss function
        compute mean cross entropy, softmax is applied internally
        '''
        #
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name='loss')

    def optimize(self):
        '''
        Define training op
        using Adam Gradient Descent to minimize cost
        '''
        self.opt = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss,
                                                            global_step=self.gstep)
    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('summaries'):
            tf.compat.v1.summary.scalar('loss', self.loss)
            tf.compat.v1.summary.scalar('accuracy', self.accuracy)
            tf.compat.v1.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.compat.v1.summary.merge_all()

    def eval(self):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    def build(self):
        '''
        Build the computation graph
        '''
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.opt, self.loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, 'data/checkpoints/convnet_mnist/mnist-convnet', step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds / self.n_test))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, n_epochs):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        utils.safe_mkdir('data/checkpoints')
        utils.safe_mkdir('data/checkpoints/convnet_mnist')
        writer = tf.compat.v1.summary.FileWriter('./data/graphs/convnet', tf.compat.v1.get_default_graph())

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('data/checkpoints/convnet_mnist/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)
        writer.close()


if __name__ == '__main__':
    model = ConvNet()
    model.build()
    model.train(n_epochs=30)
