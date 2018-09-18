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
    y = tf.placeholder(tf.float32, shape=[None, 101, 101, config.n_out_layers], name="y")
    training = tf.placeholder(tf.bool, name="training")
    p = tf.layers.conv2d(x, 32, config.kernel_size, kernel_initializer=initializer, padding="same",
                         activation=tf.nn.relu, name="conv-2")
    p = tf.layers.conv2d(p, 32, config.kernel_size, kernel_initializer=initializer, padding="same",
                         activation=tf.nn.relu, name="conv-2a")
    p = tf.nn.max_pool(p, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    p = tf.layers.conv2d(p, 64, config.kernel_size, kernel_initializer=initializer, padding="same",
                         activation=tf.nn.relu, name="conv-3")
    p = tf.layers.conv2d(p, 64, config.kernel_size, kernel_initializer=initializer, padding="same",
                         activation=tf.nn.relu, name="conv-3a")
    p = tf.nn.max_pool(p, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    p = tf.layers.conv2d(p, 128, config.kernel_size, kernel_initializer=initializer, padding="same",
                         activation=tf.nn.relu, name="conv-5")
    p = tf.layers.conv2d(p, 128, config.kernel_size, kernel_initializer=initializer, padding="same",
                         activation=tf.nn.relu, name="conv-5a")
    p = tf.nn.max_pool(p, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    p = tf.layers.conv2d(p, 256, config.kernel_size, kernel_initializer=initializer, padding="same",
                         activation=tf.nn.relu, name="conv-6")
    p = tf.layers.conv2d(p, 256, config.kernel_size, kernel_initializer=initializer, padding="same",
                         activation=tf.nn.relu, name="conv-6a")
    p = tf.nn.max_pool(p, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    p = tf.layers.conv2d(p, 512, config.kernel_size, kernel_initializer=initializer, padding="same",
                         activation=tf.nn.relu, name="conv-7")
    p = tf.layers.conv2d(p, 512, config.kernel_size, kernel_initializer=initializer, padding="same",
                         activation=tf.nn.relu, name="conv-7a")
    p = tf.nn.max_pool(p, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    p = tf.layers.conv2d(p, 1024, config.kernel_size, kernel_initializer=initializer, padding="same", activation=tf.nn.relu, name="conv-8")

    bth_size = tf.placeholder(tf.int32, name="bth_size")
    p = tf.nn.conv2d_transpose(p, filter=tf.Variable(tf.random_normal([3, 3, 512, 1024], mean=0.0, stddev=0.02)), output_shape=[bth_size, 8, 8, 512], strides=[1, 2, 2, 1], padding="SAME")
    p = tf.nn.bias_add(p, tf.Variable(tf.random_normal([512], mean=0.0, stddev=0.02)))
    p = tf.nn.relu(p)

    p = tf.nn.conv2d_transpose(p, filter=tf.Variable(tf.random_normal([3, 3, 256, 512], mean=0.0, stddev=0.02)),
                               output_shape=[bth_size, 16, 16, 256], strides=[1, 2, 2, 1], padding="SAME")
    p = tf.nn.bias_add(p, tf.Variable(tf.random_normal([256], mean=0.0, stddev=0.02)))
    p = tf.nn.relu(p)

    p = tf.nn.conv2d_transpose(p, filter=tf.Variable(tf.random_normal([3, 3, 128, 256], mean=0.0, stddev=0.02)), output_shape=[bth_size, 32, 32, 128], strides=[1, 2, 2, 1], padding="SAME")
    p = tf.nn.bias_add(p, tf.Variable(tf.random_normal([128], mean=0.0, stddev=0.02)))
    p = tf.nn.relu(p)

    p = tf.nn.conv2d_transpose(p, filter=tf.Variable(tf.random_normal([3, 3, 64, 128], mean=0.0, stddev=0.02)), output_shape=[bth_size, 64, 64, 64],
                               strides=[1, 2, 2, 1], padding="SAME")
    p = tf.nn.bias_add(p, tf.Variable(tf.random_normal([64], mean=0.0, stddev=0.02)))
    p = tf.nn.relu(p)

    p = tf.nn.conv2d_transpose(p, filter=tf.Variable(tf.random_normal([3, 3, 32, 64], mean=0.0, stddev=0.02)), output_shape=[bth_size, 128, 128, 32],
                               strides=[1, 2, 2, 1], padding="SAME")
    p = tf.nn.bias_add(p, tf.Variable(tf.random_normal([32], mean=0.0, stddev=0.02)))
    p = tf.nn.relu(p)

    out_layer = tf.layers.conv2d(p, 1, 1, kernel_initializer=initializer, name="out", activation=tf.nn.relu)
    real_out = tf.layers.dense(out_layer, 101, activation=None)
    print("outlayer: {}".format(real_out))
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(real_out, shape=(-1, 1)),
                                                               labels= tf.reshape(y, shape=(-1, 1)))
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss)
    return loss, optimizer, tf.nn.sigmoid(real_out, name="predictlayer")

def train_validation(X_train, X_train_mask, X_train_id):
    assert len(X_train) == len(X_train_mask)
    assert len(X_train) == len (X_train_id)
    p = np.random.RandomState(seed=1977).permutation(len(X_train))
    X_train, X_train_mask, X_train_id =  np.copy(X_train[p]), np.copy(X_train_mask[p]), np.copy(X_train_id[p])
    sep = np.int16(len(X_train) * config.validation_perc)
    val_border = len(X_train) - sep
    return X_train[:val_border,:,:,:], X_train_mask[:val_border,:,:,:], X_train_id[:val_border], \
           X_train[val_border:,:,:,:], X_train_mask[val_border:,:,:,:], X_train_id[val_border:]

def choose_batch(X, mask, id, rnd):
    i = rnd.choice(len(X), config.batch_size)
    #print("batch {}".format(i))
    return np.copy(X[i, :, :, :]), np.copy(mask[i, :, :, :]), np.copy(id[i])

def train_net(X, mask, id_tr, X_val, mask_val, X_test, loss, optimizer, out, sess):
    print("Training...")
    rnd = np.random.RandomState(seed=1977)
    util.reset_vars(sess)
    start_t = time.time()
    for i in range(config.epochs):
        batch, mask_batch, id_batch = choose_batch(X, mask, id_tr, rnd)
        sess.run(optimizer, feed_dict={"x:0": batch, "y:0": mask_batch, "training:0": True, "bth_size:0": config.batch_size})
        if(i % config.display_steps == 0):
            cost = sess.run(loss, feed_dict={"x:0": batch, "y:0": mask_batch, "training:0": False, "bth_size:0": config.batch_size})
            cost_test = sess.run(loss, feed_dict={"x:0": X_val, "y:0": mask_val, "training:0": False, "bth_size:0": X_val.shape[0]})
            print("Iteration {}".format(i))
            print("Loss -> train: {:.4f}, test: {:.4f}".format(cost, cost_test))
    #cost = sess.run(loss, feed_dict={"x:0": X, "y:0": mask, "training:0": False, "bth_size:0": X.shape[0]})
    #cost_test = sess.run(loss, feed_dict={"x:0": X_val, "y:0": mask_val, "training:0": False, "bth_size:0": X_val.shape[0]})
    print("Iteration {}".format(i))
    #print("Loss -> train: {:.4f}, test: {:.4f}".format(cost, cost_test))
    print("Total time {} sec".format(time.time() - start_t))
    util.save_tf_model(sess)
    print("Devising testset results...")
    y_pred = np.empty((0, config.img_size, config.img_size, 1))
    for j in range(int(X_test.shape[0] / config.pred_step)):
        print("[{}] predicting...".format((j + 1)))
        y1 = sess.run(out, feed_dict={"x:0": X_test[j * config.pred_step:(j + 1) * config.pred_step, :, :, :], "training:0": False, "bth_size:0": config.pred_step})
        y_pred = np.append(y_pred, y1, axis=0)
    if (j + 1) * config.pred_step < X_test.shape[0]:
        print("[{}] predicting...".format((j + 1)))
        y1 = sess.run(out, feed_dict={"x:0": X_test[(j + 1) * config.pred_step:, :, :, :], "training:0": False, "bth_size:0": config.pred_step})
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
        op_to_restore = graph.get_tensor_by_name("predictlayer:0")
        print("Devising testset results...")
        y_pred = np.empty((0, config.img_size, config.img_size, 1))
        for j in range(int(X_test.shape[0] / config.pred_step)):
            print("[{}] predicting...".format((j + 1)))
            y1 = sess.run(op_to_restore, feed_dict={"x:0": X_test[j * config.pred_step:(j + 1) * config.pred_step, :, :, :], "training:0": False, "bth_size:0": config.pred_step})
            y_pred = np.append(y_pred, y1, axis=0)
        if (j + 1) * config.pred_step < X_test.shape[0]:
            print("[{}] predicting...".format((j + 1)))
            y1 = sess.run(op_to_restore, feed_dict={"x:0": X_test[(j + 1) * config.pred_step:, :, :, :], "training:0": False, "bth_size:0": config.pred_step})
            y_pred = np.append(y_pred, y1, axis=0)
    else:
        y_pred = train_net(X_reduced_train, X_mask_red, X_reduced_train_id, X_validation, X_mask_validation, X_test, loss, optimizer, out, sess)
    no_test = y_pred.shape[0]
    for th in config.thresholds:
        y_pred_def = (y_pred > th) * 1
        y_pred_def = y_pred_def.reshape((-1, config.img_size * config.img_size), order="F")
        to_submit = {idx: util.convert_for_submission(y_pred_def[i,:]) for i, idx in enumerate(X_test_id)}
        assert X_test_id.shape[0] == 18000
        sub = pd.DataFrame.from_dict(to_submit, orient='index')
        sub.index.names = ['id']
        sub.columns = ['rle_mask']
        sub.to_csv(os.path.join(config.CACHE_PATH, "submit-{}.csv".format(th)))
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
