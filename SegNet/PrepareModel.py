import glob
import os

import cv2
from keras.callbacks import CSVLogger

from SegNet import LoadData
from SegNet.VggSegNet import VGGSegnet, np
from class_mappings import cmap
from config import *


def train(save_weights_path, train_images_path, train_segs_path, val_images_path, val_segs_path):
    unwanted_width = IMAGE_WIDTH
    unwanted_height = IMAGE_HEIGHT

    newest_model = save_weights_path + '.newest'

    train_batch_size = BATCH_SIZE
    val_batch_size = BATCH_SIZE
    n_classes = NO_CLASSES

    output_height = IMAGE_HEIGHT
    output_width = IMAGE_WIDTH

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
    csv_logger = CSVLogger('log.csv', append=True, separator=';')
    ep = 1
    while True:
        m.fit_generator(G, 25, validation_data=G2, validation_steps=5, epochs=1, callbacks=[csv_logger], verbose=2)
        # list all data in history

        m.save_weights(save_weights_path + "." + str(ep))
        m.save_weights(save_weights_path + ".newest")
        ep += 1


def predict(m, epoch):
    images_path = TO_PREDICT_DIR
    images_out_path = PREDICTED_OUTPUT_DIR
    images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
    images.sort()

    colors = cmap

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
