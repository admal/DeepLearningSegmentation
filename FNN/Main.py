import os

import cv2
import glob
import random

from FNN import LoadData
from FNN.VggSegNet import VGGSegnet, np, Sequential, load_model
from config import *


def train():
    unwanted_width = IMAGE_WIDTH
    unwanted_height = IMAGE_HEIGHT
    train_images_path = "C:\\Users\\wpiot\\PycharmProjects\\DeepLearningSegmentation\\FNN\\data\\dataset1" \
                        "\\images_prepped_train\\"
    train_segs_path = "C:\\Users\\wpiot\\PycharmProjects\\DeepLearningSegmentation\\FNN\\data\\dataset1" \
                      "\\annotations_prepped_train\\"

    val_images_path = "C:\\Users\\wpiot\\PycharmProjects\\DeepLearningSegmentation\\FNN\\data\\dataset1\\images_prepped_test\\"
    val_segs_path = "C:\\Users\\wpiot\\PycharmProjects\\DeepLearningSegmentation\\FNN\\data\\dataset1" \
                    "\\annotations_prepped_test\\"
    save_weights_path = "C:\\Users\\wpiot\\PycharmProjects\\DeepLearningSegmentation\\FNN\\models\\weights"
    newest_model = save_weights_path + '.newest'

    train_batch_size = BATCH_SIZE
    val_batch_size = BATCH_SIZE
    n_classes = NO_CLASSES

    output_height = IMAGE_HEIGHT
    output_width = IMAGE_WIDTH

    # m = load_model(model_json,weights)
    # if not m:
    m = VGGSegnet(n_classes, unwanted_height, unwanted_width)

    if os.path.isfile(newest_model):
        m.load_weights(newest_model)

    m.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

    G = LoadData.imageSegmentationGenerator(train_images_path, train_segs_path, train_batch_size, n_classes,
                                            unwanted_height, unwanted_width, output_height, output_width)

    G2 = LoadData.imageSegmentationGenerator(val_images_path, val_segs_path, val_batch_size, n_classes, unwanted_height,
                                             unwanted_width, output_height, output_width)

    ep = 1
    while True:
        m.fit_generator(G, 512, validation_data=G2, validation_steps=200, epochs=1)
        m.save_weights(save_weights_path + "." + str(ep))
        m.save_weights(save_weights_path + ".newest")
        predict(m, ep)
        ep += 1


def predict(m, epoch):
    images_path = "C:\\Users\\wpiot\\PycharmProjects\\DeepLearningSegmentation\\FNN\\predict_photos\\"
    images_out_path = "C:\\Users\\wpiot\\PycharmProjects\\DeepLearningSegmentation\\FNN\\predictions\\"
    images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
    images.sort()

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(NO_CLASSES)]

    for imgName in images:
        out_name = imgName.replace(images_path, images_out_path)
        path, name = out_name.rsplit('\\', 1)
        out_name = path + '\\' + str(epoch) + '__' + name
        x = LoadData.getImageArr(imgName, IMAGE_WIDTH, IMAGE_HEIGHT)
        pr = m.predict(np.array([x]))[0]
        pr = pr.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, NO_CLASSES)).argmax(axis=2)
        seg_img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        for c in range(NO_CLASSES):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
        seg_img = cv2.resize(seg_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        cv2.imwrite(out_name, seg_img)


if __name__ == '__main__':
    train()
    # m = VGGSegnet(NO_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH)
    # save_weights_path = "C:\\Users\\wpiot\\PycharmProjects\\DeepLearningSegmentation\\FNN\\models\\weights"
    # newest_model = save_weights_path + '.newest'
    # if os.path.isfile(newest_model):
    #     m.load_weights(newest_model)
    # predict(m,0)
