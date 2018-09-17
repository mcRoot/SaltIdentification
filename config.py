import os

config = {
    "base_path": "./input",
    "train_dir": "train",
    "train_persisted": "train.pck",
    "test_persisted": "test.pck",
    "mask_persisted": "mask.pck",
    "test_id_persisted": "test_id.pck",
    "train_id_persisted": "train_id.pck",
    "test_dir": "test",
    "image_dir_name": "images",
    "mask_dir_name": "masks",
    "image_ext": "png",
    "train_file": "train.csv",
    "depth_file": "depth:csv",
    "augment": True
}

CACHE_PATH = os.path.abspath("./generated_data")
MODEL_FILENAME = "nn_model.tf"

img_size = 101
img_shape = (img_size, img_size)
n_out_layers = 1
n_channels = 1
kernel_size = [3, 3]
epochs = 4000
batch_size = 80
display_steps = 50
learning_rate = 0.01
validation_perc = 0.1
#dropout_rate = 0.2

thresholds = [0.5, 0.55, 0.6, 0.65, 0.7]

pred_step = 2000
