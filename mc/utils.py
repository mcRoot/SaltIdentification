import cv2
import imutils
import numpy as np
import time
import os
import glob
import dill
import tensorflow as tf
import Augmentor
from config import config, resize_image, img_size, tta
from config import MODEL_FILENAME, CACHE_PATH

def metric_iou(pred, mask):
    both = pred + mask
    union = (both >= 1).sum()
    intr = (both == 2).sum()
    IoU = intr / float(union)
    return IoU

def calc_IoUs(pred, mask):
    ious = []
    for i, p in enumerate(pred):
        ious.append(metric_iou(p, mask[i]))
    return np.array(ious)

def full_IoU_metric(pred, mask, th=[]):
    ious_mask = {}
    for t in th:
        pred_def = (pred > t) * 1
        ious = calc_IoUs(pred_def, mask)
        ious_mask[t] = ious
    return ious_mask

def kaggle_iou_metric(complete_pred, masks, kaggle_th):
    positives = (masks.sum(axis=1) > 0) * 1.0
    res = {}
    for k in complete_pred:
        curr = complete_pred[k]
        prec = []
        for kt in kaggle_th:
            pred_pos = (curr > kt) * 1.0
            tp = ((pred_pos * positives) > 0).sum()
            fp = ((pred_pos - positives) > 0).sum()
            fn = ((pred_pos - positives) < 0).sum()
            p = float(tp) / (tp + fn + fp)
            prec.append(p)
        res[k] = [np.array(prec).mean()]
    return res

def devise_complete_iou_results(pred, mask, image_th=[], kaggle_th=[]):
    res = full_IoU_metric(pred, mask, image_th)
    return kaggle_iou_metric(res, mask, kaggle_th)

def tta_augment(X):
    res = {}
    res_flip_0 = []
    res_flip_1 = []
    res_rot_90 = []
    res_rot_180 = []
    res_rot_270 = []
    for img in X:
        res_flip_0.append(flip(img, 0).reshape(img_size, img_size, 1))
        res_flip_1.append(flip(img, 1).reshape(img_size, img_size, 1))
        res_rot_90.append(imutils.rotate_bound(img, angle=90).reshape(img_size, img_size, 1))
        res_rot_180.append(imutils.rotate_bound(img, angle=180).reshape(img_size, img_size, 1))
        res_rot_270.append(imutils.rotate_bound(img, angle=270).reshape(img_size, img_size, 1))
    res["flip_0"] = np.array(res_flip_0)
    res["flip_1"] = np.array(res_flip_1)
    res["rot_90"] = np.array(res_rot_90)
    res["rot_180"] = np.array(res_rot_180)
    res["rot_270"] = np.array(res_rot_270)
    return res


def transformations(img, trans):
    f = None
    if trans == "flip_0":
        f = lambda img: cv2.flip(img, 0)
    if trans == "flip_1":
        f = lambda img: cv2.flip(img, 1)
    if trans == "rot_90":
        f = lambda img: imutils.rotate_bound(img, -90)
    if trans == "rot_180":
        f = lambda img: imutils.rotate_bound(img, 180)
    if trans == "rot_270":
        f = lambda img: imutils.rotate_bound(img, 90)
    return f

def deaugment(X, trans):
    res = []
    for img in X:
        res.append(transformations(img, trans)(img).reshape(img_size, img_size, 1))
    return np.array(res)

def augment(X, mask, X_id):
    res_X = []
    res_mask = []
    res_id = []
    for i, img in enumerate(X):
        res_X.append(flip(img, 0).reshape(img_size, img_size, 1))
        res_X.append(flip(img, 1).reshape(img_size, img_size, 1))
        res_id.extend([X_id[i], X_id[i]])
        rot, rot_id = rotate(img, X_id[i])
        res_X.extend(rot)
        res_id.extend(rot_id)
        rot, rot_id = translate(img, X_id[i])
        res_X.extend(rot)
        res_id.extend(rot_id)
        res_mask.append(flip(mask[i], 0).reshape(img_size, img_size, 1))
        res_mask.append(flip(mask[i], 1).reshape(img_size, img_size, 1))
        res_mask.extend(rotate(mask[i], X_id[i])[0])
        res_mask.extend(translate(mask[i], X_id[i])[0])
    return res_X, res_mask, res_id

def rotate(img, img_id):
    res = []
    res_id = []
    for a in xrange(0, 350, 90):
        res.append(imutils.rotate_bound(img, a).reshape(img_size, img_size, 1))
        res_id.append(img_id)
    return res, res_id

def translate(img, img_id):
    res = []
    res_id = []
    res.append(imutils.translate(img, img_size // 2, 0).reshape(img_size, img_size, 1))
    res_id.append(img_id)
    res.append(imutils.translate(img, - (img_size // 2), 0).reshape(img_size, img_size, 1))
    res_id.append(img_id)
    res.append(imutils.translate(img, 0, img_size // 2).reshape(img_size, img_size, 1))
    res_id.append(img_id)
    res.append(imutils.translate(img, 0, - (img_size // 2)).reshape(img_size, img_size, 1))
    res_id.append(img_id)
    return res, res_id

def flip(img, kind):
    return cv2.flip(img, kind)

def get_img_cv2(path, mask=False):
    img = cv2.imread(path, 0)
    if not mask and resize_image:
        img = cv2.resize(img, (img_size, img_size))
    return img

def load_set(base_path, trainset=True, augmented=False):
    X_train = []
    X_train_id = []
    X_train_mask = []
    X_train_flip_0 = []
    X_train_flip_1 = []
    start_time = time.time()
    setpath = None
    if trainset:
        setpath = config["train_dir"]
    else:
        setpath = config["test_dir"]
    if augmented:
        setpath = config["augmented_dir"]
    base = os.path.join(base_path, setpath, config["image_dir_name"])
    base_mask = os.path.join(base_path, setpath, config["mask_dir_name"])
    print("start loading images - basepath: {}, maskpath: {}".format(base, base_mask))
    category_path = os.path.join(base, "*.{}".format(config["image_ext"]))
    files = glob.glob(category_path)
    for file in files:
        fname = os.path.basename(file)
        image = get_img_cv2(file)
        assert image.shape == (img_size, img_size)
        X_train.append(image.reshape(img_size, img_size, 1))
        X_train_id.append(fname.split(".png")[0])
        if not trainset:
            if tta:
                img1 = flip(image, 0)
                img2 = flip(image, 1)
                X_train_flip_0.append(img1.reshape(img_size, img_size, 1))
                X_train_flip_1.append(img2.reshape(img_size, img_size, 1))
        if trainset:
            #if config["augment"]:
            #    img1, img2 =  img_flip(image)
            #    X_train.append(img1.reshape(img_size, img_size, 1))
            #    X_train_id.append(fname.split(".")[0])
            #    X_train.append(img2.reshape(img_size, img_size, 1))
            #    X_train_id.append(fname.split(".")[0])
            maskfname = os.path.join(base_mask, fname)
            image = get_img_cv2(maskfname, mask=True)
            X_train_mask.append(image.reshape(101, 101, 1))
            #if config["augment"]:
            #    img1, img2 =  img_flip(image)
            #    X_train_mask.append(img1.reshape(101, 101, 1))
            #    X_train_mask.append(img2.reshape(101, 101, 1))

    print("Load images complete - total time {0:.2f} sec".format((time.time() - start_time)))
    return X_train, X_train_id, X_train_mask, X_train_flip_0, X_train_flip_1

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
            count = 0
            counting = False
    if counting:
        seq.append(count)
    assert len(seq) % 2 == 0
    return " ".join(map(str,seq))

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
