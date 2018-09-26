import os

config = {
    "base_path": "./input",
    "train_dir": "train",
    "augmented_dir": "augmented",
    "train_persisted": "train.pck",
    "test_persisted": "test.pck",
    "test0_persisted": "test0.pck",
    "test1_persisted": "test1.pck",
    "mask_persisted": "mask.pck",
    "test_id_persisted": "test_id.pck",
    "train_id_persisted": "train_id.pck",
    "test_dir": "test",
    "image_dir_name": "images",
    "mask_dir_name": "masks",
    "image_ext": "png",
    "train_file": "train.csv",
    "depth_file": "depth:csv",
    "augment": True,
    "persist": True
}

CACHE_PATH = os.path.abspath("./generated_data")
MODEL_FILENAME = "nn_model.tf"

lovasz_epochs = 3
use_lovasz_loss = True
conv_to_rgb = False
resize_image = False
tta = True
img_size = 101
img_shape = (img_size, img_size)
n_out_layers = 1
n_channels = 1
kernel_size = [3, 3]
epochs = 4400
batch_size = 100
display_steps = 100
learning_rate = 0.001
validation_perc = 0.06
#dropout_rate = 0.2

thresholds = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
kaggle_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

pred_step = 2000
