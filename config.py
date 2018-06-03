MAX_LOAD_IMAGES = 10

RAW_IMAGES_PATH = r"C:\Users\wpiot\PycharmProjects\DeepLearningSegmentation\raw_image"
SEGMENTED_IMAGES_PATH = r"C:\Users\wpiot\PycharmProjects\DeepLearningSegmentation\segmented"
LABELED_IMAGES_PATH = r"C:\Users\wpiot\PycharmProjects\DeepLearningSegmentation\test_segmented"
NO_CLASSES = 10
BATCH_SIZE = 1
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480
IMAGE_CHANNELS = 1
LEARNING_RATE = 0.005
MODEL_DIRECTORY = r"C:\Users\wpiot\PycharmProjects\DeepLearningSegmentation"
EPOCHS_COUNT = 10

TRAIN_LOG_FILE = "train.log"

# SEGNET

# Train params (dont neet to specify if predict)
TRAIN_IMAGES_PATH = "C:\\Users\\wpiot\\PycharmProjects\\DeepLearningSegmentation\\FNN\\data\\dataset1" \
                    "\\images_prepped_train\\"

TRAIN_SEGS_PATH = "C:\\Users\\wpiot\\PycharmProjects\\DeepLearningSegmentation\\FNN\\data\\dataset1" \
                  "\\annotations_prepped_train\\"

VAL_IMAGES_PATH = "C:\\Users\\wpiot\\PycharmProjects\\DeepLearningSegmentation\\FNN\\data\\dataset1\\images_prepped_test\\"

VAL_SEGS_PATH = "C:\\Users\\wpiot\\PycharmProjects\\DeepLearningSegmentation\\FNN\\data\\dataset1" \
                "\\annotations_prepped_test\\"

# Need to be specified if predict (optional for train - for resuming training)
SAVE_WEIGHTS_PATH = "C:\\Users\\wpiot\\PycharmProjects\\DeepLearningSegmentation\\FNN\\models\\weights"
TO_PREDICT_DIR = "C:\\Users\\wpiot\\PycharmProjects\\DeepLearningSegmentation\\FNN\\predict_photos\\"
PREDICTED_OUTPUT_DIR = "C:\\Users\\wpiot\\PycharmProjects\\DeepLearningSegmentation\\FNN\\predictions\\"
