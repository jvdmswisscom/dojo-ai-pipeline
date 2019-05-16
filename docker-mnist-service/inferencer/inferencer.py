import argparse
import gzip
import numpy as np
import os
import tensorflow as tf
import logging
import time

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

MODEL_FOLDER = "/outputs"

from PIL import Image


def imageprepare(image):
    im = image.convert('L')
    im = im.resize((28, 28))
    im = list(im.getdata())
    im = [(255 - x) * 1.0 / 255.0 for x in im]
    return im


def load_model(model_folder, checkpoint_filename=None):
    """
    Loads an already trained model. If you specify a checkpoint_filename then the specific checkpoint is loaded,
    otherwise the latest checkpoint is loaded.
    :param model_folder:
    :param checkpoint_filename:
    :return:
    """
    # load graph and session
    tf.reset_default_graph()
    if checkpoint_filename is None:
        checkpoint_filename = tf.train.latest_checkpoint(model_folder)
    saver = tf.train.import_meta_graph(checkpoint_filename + '.meta')
    predict_session = tf.Session()
    saver.restore(predict_session, checkpoint_filename)
    # restore tensors
    y = predict_session.graph.get_tensor_by_name("input_labels_tensor" + ':0')
    x = predict_session.graph.get_tensor_by_name("input_data_tensor" + ':0')
    keep_prob = predict_session.graph.get_tensor_by_name("keep_probability" + ':0')
    # train_step = predict_session.graph.get_tensor_by_name("train_step_tensor" + ':0')
    y_conv = predict_session.graph.get_tensor_by_name("output_tensor" + ':0')
    accuracy = predict_session.graph.get_tensor_by_name("accuracy_tensor" + ':0')
    return predict_session, (x, y, y_conv, keep_prob, None, accuracy)


def evaluate_model(model, x_test, y_test, session, batch_size=124):
    x, y, y_conv, keep_prob, _, accuracy = model
    test_length = len(x_test)
    indices = np.arange(test_length)
    measured_accuracy = 0.0
    prediction_list = []
    for start in range(0, test_length, batch_size):
        end = min(start + batch_size, test_length)
        batch_indices = indices[start:end]
        current_batch_size = len(batch_indices)
        x_batch, y_batch = x_test[batch_indices], y_test[batch_indices]
        prediction = y_conv.eval(feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0}, session=session)
        prediction_list = prediction_list + list(np.argmax(prediction, 1))
        batch_accuracy = accuracy.eval(feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0}, session=session)
        measured_accuracy = measured_accuracy + (batch_accuracy * current_batch_size)
    measured_accuracy = measured_accuracy / test_length
    return measured_accuracy, prediction_list


def classify(img):
    img = Image.open(img)
    x_test = imageprepare(img)
    x_test = np.array(x_test).reshape(1, 784)
    y_test = np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]).reshape(1, 10)  # Random class. Has not effect.
    model_path = find_model_path(MODEL_FOLDER)
    predict_session, model = load_model(model_folder=model_path)
    model_accuracy, prediction_list = evaluate_model(model, x_test, y_test, session=predict_session)
    logger.info(f"predictions: {prediction_list}")
    return prediction_list[0]


def find_model_path(p):
    input_path = p
    for dir_path, subdir_list, file_list in os.walk(input_path):
        for fname in file_list:
            if fname == "checkpoint":
                return dir_path


def main():
    classify(None)
    print('done')


if __name__ == '__main__':
    main()
    print('done')
