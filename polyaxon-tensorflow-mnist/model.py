import argparse
import gzip
import numpy as np
import os
import tensorflow as tf
import logging
import time
import os

from six.moves.urllib.request import urlretrieve

# Polyaxon
from polyaxon_client.tracking import Experiment

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

ACTIVATIONS = {
    'relu': tf.nn.relu,
    'sigmoid': tf.sigmoid,
    'tanh': tf.tanh,
}

OPTIMIZERS = {
    'sgd': tf.train.GradientDescentOptimizer,
    'rmsprop': tf.train.RMSPropOptimizer,
    'adam': tf.train.AdamOptimizer,
}

MNIST_HOST = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
IMAGE_WIDTH = 28
OUTPUT_CLASSES = 10


def load_image_data(filename):
    with gzip.open(filename, 'rb') as unzipped_file:
        data = np.frombuffer(unzipped_file.read(), dtype=np.uint8)
    images = data[16:].reshape((-1, IMAGE_WIDTH ** 2)).astype(np.float32)
    images /= 255
    return images


def load_onehot_data(filename):
    with gzip.open(filename, 'rb') as unzipped_file:
        data = np.frombuffer(unzipped_file.read(), dtype=np.uint8)
    labels = data[8:]
    length = len(labels)
    onehot = np.zeros((length, OUTPUT_CLASSES), dtype=np.float32)
    onehot[np.arange(length), labels] = 1
    return onehot


def load_mnist_data(path='/tmp/mnist'):
    if not os.path.isdir(path):
        os.makedirs(path)
    for data_file in [
        TRAIN_IMAGES,
        TRAIN_LABELS,
        TEST_IMAGES,
        TEST_LABELS,
    ]:
        destination = os.path.join(path, data_file)
        if not os.path.isfile(destination):
            urlretrieve("{}{}".format(MNIST_HOST, data_file), destination)
    return (
        (load_image_data(os.path.join(path, TRAIN_IMAGES)),
         load_onehot_data(os.path.join(path, TRAIN_LABELS))),
        (load_image_data(os.path.join(path, TEST_IMAGES)),
         load_onehot_data(os.path.join(path, TEST_LABELS))),
    )


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv_layer(x, filter_size, out_features, activation, pool_size):
    W = weight_variable([filter_size, filter_size, x.get_shape()[3].value, out_features])
    b = bias_variable([out_features])
    conv = ACTIVATIONS[activation](tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='SAME') + b)
    pool = tf.nn.max_pool(conv, ksize=[1, pool_size, pool_size, 1],
                          strides=[1, pool_size, pool_size, 1], padding='SAME')
    return pool


def fully_connected_layer(x, out_size, name):
    W = weight_variable([x.get_shape()[1].value, out_size])
    b = bias_variable([out_size])
    return tf.add(tf.matmul(x, W),b, name=name)


def create_model(conv1_size,
                 conv1_out,
                 conv1_activation,
                 pool1_size,
                 conv2_size,
                 conv2_out,
                 conv2_activation,
                 pool2_size,
                 fc1_activation,
                 fc1_size,
                 optimizer,
                 log_learning_rate):
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH ** 2], name="input_data_tensor")
    y = tf.placeholder(tf.float32, shape=[None, OUTPUT_CLASSES], name="input_labels_tensor")
    keep_prob = tf.placeholder(tf.float32, name="keep_probability")
    input_image = tf.reshape(x, [-1, IMAGE_WIDTH, IMAGE_WIDTH, 1])

    conv1 = conv_layer(input_image, conv1_size, conv1_out, conv1_activation, pool1_size)

    conv2 = conv_layer(conv1, conv2_size, conv2_out, conv2_activation, pool2_size)

    _, conv2_height, conv2_width, conv2_features = conv2.get_shape()
    flattened = tf.reshape(conv2,
                           [-1, conv2_height.value * conv2_width.value * conv2_features.value])

    fc_1 = ACTIVATIONS[fc1_activation](fully_connected_layer(flattened, fc1_size, name='first_fully_connected'))
    fc_1_drop = tf.nn.dropout(fc_1, keep_prob)

    y_conv = fully_connected_layer(fc_1_drop, OUTPUT_CLASSES, name='output_tensor')
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
    train_step = OPTIMIZERS[optimizer](10 ** log_learning_rate).minimize(cross_entropy, name="train_step_tensor")
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy_tensor")

    return x, y, y_conv, keep_prob, train_step, accuracy


def train_model(model, x_train, y_train, batch_size, dropout, epochs, x_test, y_test, experiment):
    evaluation_steps = 10
    accuracy_early_stopping = 0.9
    max_iterations = 150
    x, y, y_conv, keep_prob, train_step, _ = model
    train_length = len(x_train)
    step = 0
    for i in range(epochs):
        logger.info("New epoch")
        indices = np.arange(train_length)
        np.random.shuffle(indices)
        for start in range(0, train_length, batch_size):
            if step % evaluation_steps == 0:
                test_set_accuracy = evaluate_model(model=model, x_test=x_test, y_test=y_test)
                print(f"Evaluated at step: {str(step)}, accuracy: {test_set_accuracy}")
                experiment.log_metrics(accuracy=test_set_accuracy)
                if accuracy_early_stopping <= test_set_accuracy:
                    break
            if step > max_iterations:
                break
            end = min(start + batch_size, train_length)
            batch_indices = indices[start:end]
            x_batch, y_batch = x_train[batch_indices], y_train[batch_indices]
            train_step.run(feed_dict={x: x_batch, y: y_batch, keep_prob: dropout})
            step += 1


def evaluate_model(model, x_test, y_test, batch_size=124):
    x, y, y_conv, keep_prob, _, accuracy = model
    test_length = len(x_test)
    indices = np.arange(test_length)
    measured_accuracy = 0.0
    for start in range(0, test_length, batch_size):
        end = min(start + batch_size, test_length)
        batch_indices = indices[start:end]
        current_batch_size = len(batch_indices)
        x_batch, y_batch = x_test[batch_indices], y_test[batch_indices]
        batch_accuracy = accuracy.eval(feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0})
        measured_accuracy = measured_accuracy + (batch_accuracy * current_batch_size)
    measured_accuracy = measured_accuracy / test_length
    return measured_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--conv1_size',
        type=int,
        default=5)
    parser.add_argument(
        '--conv1_out',
        type=int,
        default=32)
    parser.add_argument(
        '--conv1_activation',
        type=str,
        default='relu')
    parser.add_argument(
        '--pool1_size',
        type=int,
        default=2)
    parser.add_argument(
        '--conv2_size',
        type=int,
        default=5
    )
    parser.add_argument(
        '--conv2_out',
        type=int,
        default=64)
    parser.add_argument(
        '--conv2_activation',
        type=str,
        default='relu')
    parser.add_argument(
        '--pool2_size',
        type=int,
        default=2)
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2
    )
    parser.add_argument(
        '--fc1_size',
        type=int,
        default=1024
    )
    parser.add_argument(
        '--fc1_activation',
        type=str,
        default='sigmoid')
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam'
    )
    parser.add_argument(
        '--log_learning_rate',
        type=int,
        default=-3
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1
    )
    args = parser.parse_args()
    experiment = Experiment()

    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    logger.info("loaded data (info)")
    experiment.log_data_ref(data=x_train, data_name='x_train')
    experiment.log_data_ref(data=y_train, data_name='y_train')
    experiment.log_data_ref(data=x_test, data_name='x_test')
    experiment.log_data_ref(data=y_test, data_name='y_test')


    with tf.Session() as sess:
        model = create_model(
            conv1_size=args.conv1_size,
            conv1_out=args.conv1_out,
            conv1_activation=args.conv1_activation,
            pool1_size=args.pool1_size,
            conv2_size=args.conv2_size,
            conv2_out=args.conv2_out,
            conv2_activation=args.conv2_activation,
            pool2_size=args.pool2_size,
            fc1_activation=args.fc1_activation,
            fc1_size=args.fc1_size,
            optimizer=args.optimizer,
            log_learning_rate=args.log_learning_rate)
        logger.info("model created")
        if os.environ.get('POLYAXON_NO_OP', "0") != "1":
            saver = tf.train.Saver(save_relative_paths=False)
            saver.save_path = os.path.join(experiment.get_outputs_path(), 'model.ckpt')
            logger.info(f"saving model at {str(saver.save_path)}")
        else:
            saver = tf.train.Saver(save_relative_paths=True)
            saver.save_path = os.path.join('log_dir', 'model.ckpt')
            logger.info(f"saving model at {str(saver.save_path)}")
        sess.run(tf.global_variables_initializer())
        logger.info("model initialized")
        train_model(model,
                    x_train,
                    y_train,
                    batch_size=args.batch_size,
                    dropout=args.dropout,
                    epochs=args.epochs,
                    x_test=x_test,
                    y_test=y_test,
                    experiment=experiment)
        logger.info("model trained")
        model_save_path = saver.save(sess=sess, save_path=saver.save_path)
        logger.info(f"model saved at {str(model_save_path)}")
        accuracy = evaluate_model(model, x_test, y_test)
        logger.info(f"model evaluated acc:{accuracy}")
        # Polyaxon
        experiment.log_metrics(accuracy=accuracy)
    logger.info("done")


if __name__ == '__main__':
    main()
