import mc.utils as util
import config as config
import time
import os
import numpy as np
import mc.losses as losses
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import cv2

def preprocess():
    print("Preprocessing...")
    preprocessed_file = os.path.join(config.CACHE_PATH, config.config['train_persisted'])
    if os.path.isfile(preprocessed_file):
        X_train = util.load_cache(os.path.join(config.CACHE_PATH, config.config['train_persisted']))
        X_train_id = util.load_cache(os.path.join(config.CACHE_PATH, config.config['train_id_persisted']))
        X_train_mask = util.load_cache(os.path.join(config.CACHE_PATH, config.config['mask_persisted']))
    else:
        X_train, X_train_id, X_train_mask, _, _ = util.load_set(config.config["base_path"], True)
        if config.config["augment"]:
            X_aug, X_aug_mask, X_aug_id = util.augment(X_train, X_train_mask, X_train_id)
            X_train.extend(X_aug)
            X_train_id.extend(X_aug_id)
            X_train_mask.extend(X_aug_mask)
        X_train = util.normalize_set(X_train)
        X_train_mask = util.normalize_set(X_train_mask)
        if config.config["persist"]:
            util.persist(os.path.join(config.CACHE_PATH, config.config['train_persisted']), X_train)
            util.persist(os.path.join(config.CACHE_PATH, config.config['train_id_persisted']), X_train_id)
            util.persist(os.path.join(config.CACHE_PATH, config.config['mask_persisted']), X_train_mask)

    preprocessed_file = os.path.join(config.CACHE_PATH, config.config['test_persisted'])
    if os.path.isfile(preprocessed_file):
        X_test = util.load_cache(os.path.join(config.CACHE_PATH, config.config['test_persisted']))
        X_test_id = util.load_cache(os.path.join(config.CACHE_PATH, config.config['test_id_persisted']))
    else:
        X_test, X_test_id, _, _, _ = util.load_set(config.config["base_path"], False)
        X_test = util.normalize_set(X_test)
        util.persist(os.path.join(config.CACHE_PATH, config.config['test_persisted']), X_test)
        util.persist(os.path.join(config.CACHE_PATH, config.config['test_id_persisted']), X_test_id)
    return X_train, np.array(X_train_id), X_train_mask, X_test, np.array(X_test_id, dtype=np.object)

def build_net():
    initializer = tf.contrib.layers.xavier_initializer(uniform=False,dtype=tf.float32)
    x = tf.placeholder(tf.float32, shape=[None, config.img_size, config.img_size, config.n_channels], name="x")
    if config.conv_to_rgb:
        x = tf.image.grayscale_to_rgb(x)
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
    p = tf.nn.conv2d_transpose(p, filter=tf.Variable(tf.random_normal([3, 3, 512, 1024], mean=0.0, stddev=0.02)), output_shape=[bth_size, 7, 7, 512], strides=[1, 2, 2, 1], padding="SAME")
    p = tf.nn.bias_add(p, tf.Variable(tf.random_normal([512], mean=0.0, stddev=0.02)))
    p = tf.nn.relu(p)

    p = tf.nn.conv2d_transpose(p, filter=tf.Variable(tf.random_normal([3, 3, 256, 512], mean=0.0, stddev=0.02)),
                               output_shape=[bth_size, 13, 13, 256], strides=[1, 2, 2, 1], padding="SAME")
    p = tf.nn.bias_add(p, tf.Variable(tf.random_normal([256], mean=0.0, stddev=0.02)))
    p = tf.nn.relu(p)

    p = tf.nn.conv2d_transpose(p, filter=tf.Variable(tf.random_normal([3, 3, 128, 256], mean=0.0, stddev=0.02)), output_shape=[bth_size, 26, 26, 128], strides=[1, 2, 2, 1], padding="SAME")
    p = tf.nn.bias_add(p, tf.Variable(tf.random_normal([128], mean=0.0, stddev=0.02)))
    p = tf.nn.relu(p)

    p = tf.nn.conv2d_transpose(p, filter=tf.Variable(tf.random_normal([3, 3, 64, 128], mean=0.0, stddev=0.02)), output_shape=[bth_size, 51, 51, 64],
                               strides=[1, 2, 2, 1], padding="SAME")
    p = tf.nn.bias_add(p, tf.Variable(tf.random_normal([64], mean=0.0, stddev=0.02)))
    p = tf.nn.relu(p)

    p = tf.nn.conv2d_transpose(p, filter=tf.Variable(tf.random_normal([3, 3, 32, 64], mean=0.0, stddev=0.02)), output_shape=[bth_size, 101, 101, 32],
                               strides=[1, 2, 2, 1], padding="SAME")
    p = tf.nn.bias_add(p, tf.Variable(tf.random_normal([32], mean=0.0, stddev=0.02)))
    p = tf.nn.relu(p)

    out_layer = tf.layers.conv2d(p, 1, 1, kernel_initializer=initializer, name="out")
    print("outlayer: {}".format(out_layer))
    if config.use_lovasz_loss:
        lovasz = losses.lovasz_hinge_flat(logits=tf.reshape(out_layer, shape=(1, -1)),
                                                               labels= tf.reshape(y, shape=(1, -1)))
        loss= tf.reduce_mean(lovasz)
    else:
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(out_layer, shape=(-1, 1)),
                                                               labels= tf.reshape(y, shape=(-1, 1)))
        loss = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss)

    return loss, optimizer, tf.nn.sigmoid(out_layer, name="predictlayer")

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

def get_next_batch(X, mask, id):
    start_index = 0
    end_index = len(X)
    print("End index {}".format(end_index))
    while start_index <= end_index:
        yield np.copy(X[start_index:start_index+config.batch_size, :, :, :]), np.copy(mask[start_index:start_index+config.batch_size, :, :, :]), np.copy(id[start_index:start_index+config.batch_size])
        start_index += config.batch_size

def train_net(X, mask, id_tr, X_val, mask_val, X_test, loss, optimizer, out, sess):
    print("Training...")
    rnd = np.random.RandomState(seed=1977)
    util.reset_vars(sess)
    start_t = time.time()
    step = []
    cost_batch = []
    cost_val = []
    ious = None
    df_empty = pd.DataFrame({th: [] for th in config.thresholds})
    df_empty['epoch'] = []

    for i in range(config.epochs):
        print("Epoch {}".format(i))
        ii = 0
        for batch, mask_batch, id_batch in get_next_batch(X, mask, id_tr):
            sess.run(optimizer, feed_dict={"x:0": batch, "y:0": mask_batch, "training:0": True, "bth_size:0": len(batch)})
            if(ii % config.display_steps == 0):
                cost = sess.run(loss, feed_dict={"x:0": batch, "y:0": mask_batch, "training:0": False, "bth_size:0": len(batch)})
                cost_test = sess.run(loss, feed_dict={"x:0": X_val, "y:0": mask_val, "training:0": False, "bth_size:0": X_val.shape[0]})
                out_val = sess.run(out, feed_dict={"x:0": X_val, "training:0": False, "bth_size:0": X_val.shape[0]})
                out_val = out_val.reshape((-1, config.img_size * config.img_size), order="F")
                mask_val_tmp = mask_val.reshape((-1, config.img_size * config.img_size), order="F")
                def_res = util.devise_complete_iou_results(out_val, mask_val_tmp, config.thresholds, config.kaggle_thresholds)
                df_calc = pd.DataFrame(def_res)
                df_calc['epoch'] = (ii * (i + 1))
                df_empty = df_empty.append(df_calc)

                print("Epoch {} Iteration {}".format(i, ii))
                print("Loss -> train: {:.4f}, test: {:.4f}".format(cost, cost_test))
                print("IoUs {}".format(def_res))
                step.append(ii * (i + 1))
                cost_batch.append(cost)
                cost_val.append(cost_test)
            ii += 1
        print("Total batches {}".format(ii))
    cost_df = pd.DataFrame({"epoch": step, "cost_batch": cost_batch, "cost_val": cost_val})
    #cost = sess.run(loss, feed_dict={"x:0": X, "y:0": mask, "training:0": False, "bth_size:0": X.shape[0]})
    #cost_test = sess.run(loss, feed_dict={"x:0": X_val, "y:0": mask_val, "training:0": False, "bth_size:0": X_val.shape[0]})
    print("Iteration {}".format(i))
    #print("Loss -> train: {:.4f}, test: {:.4f}".format(cost, cost_test))
    print("Total time {} sec".format(time.time() - start_t))
    util.save_tf_model(sess)
    print("Devising testset results...")
    y_pred = np.empty((0, config.img_size *  config.img_size))
    X_test_aug = []
    if config.tta:
        X_test_aug = util.tta_augment(X_test)
    for j in range(int(X_test.shape[0] / config.pred_step)):
        print("[{}] predicting...".format((j + 1)))
        y1 = sess.run(out, feed_dict={"x:0": X_test[j * config.pred_step:(j + 1) * config.pred_step, :, :, :], "training:0": False, "bth_size:0": config.pred_step})
        y_aug = []
        y_aug.append(y1.reshape((-1, config.img_size * config.img_size), order="F"))
        for trans, aug in X_test_aug.items():
            y_tmp = sess.run(out, feed_dict={"x:0": aug[j * config.pred_step:(j + 1) * config.pred_step, :, :, :],
                                     "training:0": False, "bth_size:0": config.pred_step})
            y_tmp = util.deaugment(y_tmp, trans)
            y_aug.append(y_tmp.reshape((-1, config.img_size * config.img_size), order="F"))
        y_aug = np.array(y_aug)
        y_mean = y_aug.mean(axis=0)
        y_pred = np.append(y_pred, y_mean, axis=0)

    if (j + 1) * config.pred_step < X_test.shape[0]:
        print("[{}] predicting...".format((j + 1)))
        y1 = sess.run(out, feed_dict={"x:0": X_test[(j + 1) * config.pred_step:, :, :, :], "training:0": False, "bth_size:0": config.pred_step})
        y_aug = []
        y_aug.append(y1.reshape((-1, config.img_size * config.img_size), order="F"))
        for aug in X_test_aug:
            y_aug.append(sess.run(out, feed_dict={"x:0": aug[(j + 1) * config.pred_step:, :, :, :],
                                                  "training:0": False, "bth_size:0": config.pred_step}).reshape(
                (-1, config.img_size * config.img_size), order="F"))
        y_aug = np.array(y_aug)
        y_mean = y_aug.mean(axis=0)
        y_pred = np.append(y_pred, y_mean, axis=0)
    return y_pred, df_empty, cost_df



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
        y_pred, df_empty, cost_df = train_net(X_reduced_train, X_mask_red, X_reduced_train_id, X_validation, X_mask_validation, X_test, loss, optimizer, out, sess)
        df_empty.to_csv(os.path.join(config.CACHE_PATH, "ious_val.csv"))
        cost_df.to_csv(os.path.join(config.CACHE_PATH, "costs_train.csv"))
        th_max = df_empty.tail(1)[config.thresholds].idxmax(axis=1).values[0]
        print("Max threshold {}".format(th_max))
    no_test = y_pred.shape[0]
    y_pred_def = y_pred
    #y_pred_def.shape[0] == 18000

    y_pred_def = (y_pred_def > th_max) * 1

    to_submit = {idx: util.convert_for_submission(y_pred_def[i,:]) for i, idx in enumerate(X_test_id)}
    #assert X_test_id.shape[0] == 18000
    sub = pd.DataFrame.from_dict(to_submit, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(os.path.join(config.CACHE_PATH, "submit-{}.csv".format(th_max)))

    print("Done")


