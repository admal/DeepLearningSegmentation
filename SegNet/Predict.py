from SegNet.PrepareModel import predict
from SegNet.VggSegNet import VGGSegnet, os
from config import NO_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH, SAVE_WEIGHTS_PATH

if __name__ == '__main__':
    m = VGGSegnet(NO_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH)

    newest_model = SAVE_WEIGHTS_PATH + '.newest'
    if os.path.isfile(newest_model):
        m.load_weights(newest_model)
    predict(m, -1)
