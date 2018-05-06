import datetime
import logging
import os

import cv2
import numpy as np
import tensorflow as tf
from FNN.TrainModel import TrainModel
from class_mappings import *
from config import *

logging.basicConfig(filename=TRAIN_LOG_FILE, level=logging.DEBUG)


def normalize_image(image):
    return image / 255


def normalized2rgb(normalized_image):
    return normalized_image * 255


def get_classes_from_rgb(rgb):
    tuple_rgb = tuple(rgb)
    classes = np.zeros(NO_CLASSES)
    if tuple_rgb in rgb2class:
        classes[rgb2class[tuple_rgb]] = 1
    else:
        classes[30] = 1  # TMP! TODO: do sth with unprocessed pixels
    return classes


def rgb_image2class_image(rgb_image):
    new_image = []
    for x in range(rgb_image.shape[0]):
        new_image.append([])
        for y in range(rgb_image.shape[1]):
            rgb = rgb_image[x, y]
            classes = get_classes_from_rgb(tuple(rgb))
            new_image[x].append(list(classes))
    return np.array(new_image)


def preprocess_image(path):
    raw_data = os.listdir(path)
    raw_data = [RAW_IMAGES_PATH + "\\" + s for s in raw_data]

    raw_train = raw_data[: int(len(raw_data) * .95)]
    raw_validate = raw_data[len(raw_train):]

    raw_train_tensor = get_image_tensor(raw_train)
    raw_validate_tensor = get_image_tensor(raw_validate)

    return raw_train_tensor, raw_validate_tensor


def get_image_tensor(images_list):
    image_files_prod = tf.train.string_input_producer(images_list, shuffle=False, seed=1)
    reader = tf.WholeFileReader()
    image_file_name, image = reader.read(image_files_prod)
    image = tf.to_float(tf.image.decode_png(image, channels=3))
    image = tf.image.resize_images(
        image,
        [IMAGE_WIDTH, IMAGE_HEIGHT]
    )
    return image


def load_data(max_load, curr_idx):
    # x = []
    # y = []
    # x_v = []
    # y_v = []
    # skip = 10

    raw_train, raw_validate = preprocess_image(RAW_IMAGES_PATH)
    labeled_train, labeled_validate = preprocess_image(LABELED_IMAGES_PATH)

    raw_train, raw_validate, labeled_train, labeled_validate = tf.train.batch(
        [raw_train, raw_validate, labeled_train, labeled_validate],
        batch_size=20,
        capacity=1000)

    model = TrainModel(MODEL_DIRECTORY + '\\trained-model')

    model = model.get_model()

    with tf.Session() as sess:
        # initialize
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        i = 0
        while i < 255:
            im_lab = sess.run([raw_train, raw_validate, labeled_train, labeled_validate])

            labeled_images = rgb_image2class_image(im_lab[2])
            val_labeled_imagages = rgb_image2class_image(im_lab[3])
            np.expand_dims(labeled_images, axis=0)
            np.expand_dims(val_labeled_imagages, axis=0)
            model.fit(
                im_lab[0], labeled_images, epochs=EPOCHS_COUNT, verbose=1,
                validation_data=(im_lab[1], val_labeled_imagages)
            )
            i += 1
        coord.request_stop()
        coord.join(threads)

    model_json = model.to_json()

    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")

    # raw_data = os.listdir(RAW_IMAGES_PATH)
    # raw_data = [RAW_IMAGES_PATH + "\\" + s for s in raw_data]
    # labaled_data = os.listdir(LABELED_IMAGES_PATH)
    # labaled_data = [LABELED_IMAGES_PATH + "\\" + s for s in labaled_data]
    #
    # raw_train = labaled_data[: int(len(labaled_data) * .95)]
    # raw_validate = labaled_data[len(raw_train):]
    #
    # raw_train = labaled_data[: int(len(labaled_data) * .95)]
    # raw_validate = labaled_data[len(raw_train):]
    #
    # image_files_prod = tf.train.string_input_producer(raw_data, shuffle=False, seed=1)
    # label_files_prod = tf.train.string_input_producer(labaled_data, shuffle=False, seed=1)
    #
    # reader = tf.WholeFileReader()
    #
    # image_file, image = reader.read(image_files_prod)
    # label_file, label = reader.read(label_files_prod)
    #
    # image = tf.to_float(tf.image.decode_png(image, channels=3)) / 256
    # label = tf.to_float(tf.image.decode_png(label, channels=3))

    # labeled_image = rgb_image2class_image(labeled_image)

    # image = tf.image.resize_images(
    #     image,
    #     [IMAGE_WIDTH, IMAGE_HEIGHT]
    # )
    #
    # label = tf.image.resize_images(
    #     label,
    #     [IMAGE_WIDTH, IMAGE_HEIGHT]
    # )

    # image_batch, label_batch = tf.train.batch([image, label],
    #                                           batch_size=1,
    #                                           capacity=1)

    # return image_batch, label_batch

    # for raw_image_path, labaled_image_path, count in zip(os.listdir(RAW_IMAGES_PATH),
    #                                                      os.listdir(LABELED_IMAGES_PATH),
    #                                                      range(1, MAX_LOAD_IMAGES + 1)):
    #     if count <= curr_idx:  # I know it is shitty solution, TODO: fix it ~AM
    #         continue
    #     print("{} {} {}".format(raw_image_path, labaled_image_path, count))
    #
    #     raw_image = cv2.imread(os.path.join(RAW_IMAGES_PATH, raw_image_path))
    #     labeled_image = cv2.imread(os.path.join(LABELED_IMAGES_PATH, labaled_image_path))
    #
    #     raw_image = cv2.resize(raw_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    #     labeled_image = cv2.resize(labeled_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    #     raw_image = normalize_image(raw_image)
    #     labeled_image = rgb_image2class_image(labeled_image)
    #
    #     if count % skip == 0:
    #         x_v.append(raw_image)
    #         y_v.append(labeled_image)
    #     else:
    #         x.append(raw_image)
    #         y.append(labeled_image)
    #
    #     if count >= curr_idx + max_load:  # I know it is shitty solution, TODO: fix it ~AM
    #         break
    #
    # return np.asarray(x), np.asarray(y), np.asarray(x_v), np.asarray(y_v)


def main():
    iterations = 1
    load_by_iter = int(MAX_LOAD_IMAGES / iterations)

    for iteration in range(iterations):
        logging.info("START ITER {}".format(iteration + 1))
        x, y, x_v, y_v = load_data(load_by_iter, iteration * load_by_iter)
        print(x.dtype)
        logging.info("[{}]".format(datetime.datetime.now()))
        logging.info("Lengths: x: {}, y: {}, x_v: {}, y_v: {}".format(len(x), len(y), len(x_v), len(y_v)))
        logging.info("Trainig data shape: x: {}, y: {}".format(x.shape, y.shape))
        logging.info("Validation data shape: x_v: {}, y_v: {}".format(x_v.shape, y_v.shape))

        model = TrainModel(MODEL_DIRECTORY + '\\trained-model')
        # print("Input shape: {}".format(x.shape))
        model.train(x, y, x_v, y_v)
        del x
        del y
        del x_v
        del y_v

    logging.info("FINISH")


if __name__ == '__main__':
    main()
