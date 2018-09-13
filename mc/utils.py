import cv2
import numpy as np
import time
import os
import glob
import dill
import tensorflow as tf
from config import config
from config import MODEL_FILENAME, CACHE_PATH


def get_img_cv2(path):
    img = cv2.imread(path, 0)
    return img

def img_flip(img):
    return cv2.flip(img, 0), cv2.flip(img, 1)

def load_set(base_path, trainset=True):
    X_train = []
    X_train_id = []
    X_train_mask = []
    start_time = time.time()
    setpath = None
    if trainset:
        setpath = config["train_dir"]
    else:
        setpath = config["test_dir"]
    base = os.path.join(base_path, setpath, config["image_dir_name"])
    base_mask = os.path.join(base_path, setpath, config["mask_dir_name"])
    print("start loading images - basepath: {}, maskpath: {}".format(base, base_mask))
    category_path = os.path.join(base, "*.{}".format(config["image_ext"]))
    files = glob.glob(category_path)
    for file in files:
        fname = os.path.basename(file)
        image = get_img_cv2(file)
        assert image.shape == (101, 101)
        X_train.append(image.reshape(101, 101, 1))
        X_train_id.append(fname.split(".png")[0])
        if trainset:
            if config["augment"]:
                img1, img2 =  img_flip(image)
                X_train.append(img1.reshape(101, 101, 1))
                X_train_id.append(fname.split(".")[0])
                X_train.append(img2.reshape(101, 101, 1))
                X_train_id.append(fname.split(".")[0])
            maskfname = os.path.join(base_mask, fname)
            image = get_img_cv2(maskfname)
            X_train_mask.append(image.reshape(101, 101, 1))
            if config["augment"]:
                img1, img2 =  img_flip(image)
                X_train_mask.append(img1.reshape(101, 101, 1))
                X_train_mask.append(img2.reshape(101, 101, 1))

    print("Load images complete - total time {0:.2f} sec".format((time.time() - start_time)))
    return X_train, X_train_id, X_train_mask

def convert_for_submission(m):
    counting = False
    count = 0
    seq = []
    for i, c in enumerate(m):
        if c == 1 and counting:
            count = count + 1
        if c == 1 and not counting:
            seq.append(i + 1)
            count = 1
            counting = True
        if c == 0 and counting:
            seq.append(count)
            counting = False
    return  " ".join(map(str, seq))

def normalize_set(dataset=[]):
    print("convert to numpy array...")
    dataset = np.array(dataset, dtype=np.uint8)
    print("dataset shape {}".format(dataset.shape))
    #print("Reshape...")
    #dataset = dataset.transpose((0, 2, 3, 1))
    #dataset = dataset.transpose((0, 1, 3, 2))
    #print("new dataset shape {}".format(dataset.shape))
    print("Convert to float...")
    dataset = dataset.astype(np.float32)
    dataset = dataset / 255
    #target = LabelBinarizer().fit_transform(target)
    #print("new target shape {}".format(target.shape))
    return dataset

def save_tf_model(sess):
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(CACHE_PATH, MODEL_FILENAME))

def persist(filename, obj):
    with open(filename, 'wb') as file:
        dill.dump(obj, file)

def load_cache(filename):
    return dill.load(open(filename, 'rb'))

def reset_vars(sess):
    sess.run(tf.global_variables_initializer())

def reset_tf(sess):
    if sess:
        sess.close()
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess
