import mc.utils as util
import config as config
import time
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

def preprocess():
    print("Preprocessing...")
    preprocessed_file = os.path.join(config.CACHE_PATH, config.config['train_persisted'])
    if os.path.isfile(preprocessed_file):
        X_train = util.load_cache(os.path.join(config.CACHE_PATH, config.config['train_persisted']))
        X_train_id = util.load_cache(os.path.join(config.CACHE_PATH, config.config['train_id_persisted']))
        X_train_mask = util.load_cache(os.path.join(config.CACHE_PATH, config.config['mask_persisted']))
    else:
        X_train, X_train_id, X_train_mask = util.load_set(config.config["base_path"], True)
        X_train = util.normalize_set(X_train)
        X_train_mask = util.normalize_set(X_train_mask)
        util.persist(os.path.join(config.CACHE_PATH, config.config['train_persisted']), X_train)
        util.persist(os.path.join(config.CACHE_PATH, config.config['train_id_persisted']), X_train_id)
        util.persist(os.path.join(config.CACHE_PATH, config.config['mask_persisted']), X_train_mask)

    preprocessed_file = os.path.join(config.CACHE_PATH, config.config['test_persisted'])
    if os.path.isfile(preprocessed_file):
        X_test = util.load_cache(os.path.join(config.CACHE_PATH, config.config['test_persisted']))
        X_test_id = util.load_cache(os.path.join(config.CACHE_PATH, config.config['test_id_persisted']))
    else:
        X_test, X_test_id, _ = util.load_set(config.config["base_path"], False)
        X_test = util.normalize_set(X_test)
        util.persist(os.path.join(config.CACHE_PATH, config.config['test_persisted']), X_test)
        util.persist(os.path.join(config.CACHE_PATH, config.config['test_id_persisted']), X_test_id)
    return X_train, np.array(X_train_id), X_train_mask, X_test, np.array(X_test_id, dtype=np.object)

def build_net():
    initializer = tf.contrib.layers.xavier_initializer(uniform=False,dtype=tf.float32)
    x = tf.placeholder(tf.float32, shape=[None, config.img_size, config.img_size, config.n_channels], name="x")
    y = tf.placeholder(tf.float32, shape=[None, config.img_size, config.img_size, config.n_out_layers], name="y")
    training = tf.placeholder(tf.bool, name="training")
    print(x)
    prev_layer = x
    conv_layers = []
    for i, c in enumerate(config.conv_layers):
        conv_layers.append(tf.layers.conv2d(prev_layer, c, config.kernel_size, kernel_initializer=initializer, padding="same", activation=tf.nn.relu, name="conv-{}".format(i)))
        prev_layer = tf.layers.dropout(conv_layers[-1], config.dropout_rate, training=training)
    out_layer = tf.layers.conv2d(prev_layer, 1, 1, kernel_initializer=initializer, name="out")
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(out_layer, shape=(-1, 1)),
                                                               labels= tf.reshape(y, shape=(-1, 1)))
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss)
    return loss, optimizer, tf.nn.sigmoid(out_layer)

def train_validation(X_train, X_train_mask, X_train_id):
    assert len(X_train) == len(X_train_mask)
    assert len(X_train) == len (X_train_id)
    p = np.random.RandomState(seed=42).permutation(len(X_train))
    X_train, X_train_mask, X_train_id =  X_train[p], X_train_mask[p], X_train_id[p]
    sep = np.int16(len(X_train) * config.validation_perc)
    val_border = len(X_train) - sep
    return X_train[:val_border,:,:,:], X_train_mask[:val_border,:,:,:], X_train_id[:val_border], \
           X_train[val_border:,:,:,:], X_train_mask[val_border:,:,:,:], X_train_id[val_border:]

def choose_batch(X, mask, id, rnd):
    i = rnd.choice(len(X), config.batch_size)
    return np.copy(X[i, :, :, :]), np.copy(mask[i, :, :, :]), np.copy(id[i])

def train_net(X, mask, id, X_val, mask_val, X_test, loss, optimizer, out, sess):
    print("Training...")
    rnd = np.random.RandomState(seed=42)
    util.reset_vars(sess)
    start_t = time.time()
    for i in range(config.epochs):
        batch, mask_batch, id_batch = choose_batch(X, mask, id, rnd)
        sess.run(optimizer, feed_dict={"x:0": batch, "y:0": mask_batch, "training:0": True})
        if(i % config.display_steps == 0):
            cost = "skipped"#cost = sess.run(loss, feed_dict={"x:0": X, "y:0": mask, "training:0": False})
            cost_test = sess.run(loss, feed_dict={"x:0": X_val, "y:0": mask_val, "training:0": False})
            print("Iteration {}".format(i))
            print("Loss -> train: {:.4f}, test: {:.4f}".format(cost, cost_test))
    cost = "skipped" #cost = sess.run(loss, feed_dict={"x:0": X, "y:0": mask, "training:0": False})
    cost_test = sess.run(loss, feed_dict={"x:0": X_val, "y:0": mask_val, "training:0": False})
    print("Iteration {}".format(i))
    print("Loss -> train: {:.4f}, test: {:.4f}".format(cost, cost_test))
    print("Total time {} sec".format(time.time() - start_t))
    util.save_tf_model(sess)
    print("Devising testset results...")
    y_pred = np.empty((0, config.img_size, config.img_size, 1))
    for j in range(int(X_test.shape[0] / config.pred_step)):
        print("[{}] predicting...".format((j + 1)))
        y1 = sess.run(out, feed_dict={"x:0": X_test[j * config.pred_step:(j + 1) * config.pred_step, :, :, :], "training:0": False})
        y_pred = np.append(y_pred, y1, axis=0)
    if (j + 1) * config.pred_step < X_test.shape[0]:
        print("[{}] predicting...".format((j + 1)))
        y1 = sess.run(out, feed_dict={"x:0": X_test[(j + 1) * config.pred_step:, :, :, :], "training:0": False})
        y_pred = np.append(y_pred, y1, axis=0)
    return y_pred



if __name__ == "__main__":
    X_train, X_train_id, X_train_mask, X_test, X_test_id = preprocess()
    X_reduced_train, X_mask_red, X_reduced_train_id, X_validation, X_mask_validation, X_validation_id = train_validation(X_train, X_train_mask, X_train_id)
    sess = util.reset_tf(None)
    loss, optimizer, out = build_net()
    if os.path.isfile(os.path.join(config.CACHE_PATH, "{}.meta".format(config.MODEL_FILENAME))):
        print("Restoring model...")
        sess = tf.Session()
        util.reset_vars(sess)
        saver = tf.train.import_meta_graph(os.path.join(config.CACHE_PATH, "{}.meta".format(config.MODEL_FILENAME)))
        saver.restore(sess, tf.train.latest_checkpoint(config.CACHE_PATH))
        # Now, let's access and create placeholders variables and
        # create feed-dict to feed new data
        graph = tf.get_default_graph()
        print(graph)
        # Now, access the op that you want to run.
        op_to_restore = graph.get_tensor_by_name("Sigmoid:0")
        print("Devising testset results...")
        y_pred = np.empty((0, config.img_size, config.img_size, 1))
        for j in range(int(X_test.shape[0] / config.pred_step)):
            print("[{}] predicting...".format((j + 1)))
            y1 = sess.run(out, feed_dict={"x:0": X_test[j * config.pred_step:(j + 1) * config.pred_step, :, :, :], "training:0": False})
            y_pred = np.append(y_pred, y1, axis=0)
        if (j + 1) * config.pred_step < X_test.shape[0]:
            print("[{}] predicting...".format((j + 1)))
            y1 = sess.run(out, feed_dict={"x:0": X_test[(j + 1) * config.pred_step:, :, :, :], "training:0": False})
            y_pred = np.append(y_pred, y1, axis=0)
    else:
        y_pred = train_net(X_reduced_train, X_mask_red, X_reduced_train_id, X_validation, X_mask_validation, X_test, loss, optimizer, out, sess)
    no_test = y_pred.shape[0]
    y_pred = (y_pred > 0.5) * 1
    y_pred = y_pred.reshape((-1, config.img_size * config.img_size), order="F")
    to_submit = np.apply_along_axis(util.convert_for_submission, 1, y_pred)
    submit_df = pd.DataFrame({"id": X_test_id, "rle_mask": to_submit})
    submit_df.to_csv(os.path.join(config.CACHE_PATH, "submit.csv"))
    #z = y_pred[20].reshape(config.img_size, config.img_size)
    #img = ((y_pred[20] > 0.5) * 1.0).reshape(config.img_size, config.img_size)
    #f = plt.figure(figsize=(8, 6))
    #plt.imshow(img)
    #f.savefig("pred1.png")
    #f = plt.figure(figsize=(8, 6))
    #plt.imshow(X_mask_validation[20].reshape(config.img_size, config.img_size))
    #f.savefig("mask1.png")

    print("Done")


def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)