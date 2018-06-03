from SegNet.PrepareModel import train
from config import *

if __name__ == '__main__':
    train(SAVE_WEIGHTS_PATH, TRAIN_IMAGES_PATH, TRAIN_SEGS_PATH, VAL_IMAGES_PATH, VAL_SEGS_PATH)
