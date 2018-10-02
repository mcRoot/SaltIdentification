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
from sklearn.model_selection import StratifiedShuffleSplit

current_global_step = None

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
        if config.config["persist"]:
            util.persist(os.path.join(config.CACHE_PATH, config.config['test_persisted']), X_test)
            util.persist(os.path.join(config.CACHE_PATH, config.config['test_id_persisted']), X_test_id)
    return X_train, np.array(X_train_id), X_train_mask, X_test, np.array(X_test_id, dtype=np.object)

def encode_layer(input=None, feature_maps=32, initializer=None, activation=tf.nn.relu, training=None, max_pooling=True):
    if config.user_resnet:
        return encode_layer_resnet(input=input, feature_maps=feature_maps, initializer=initializer, activation=activation, training=training, max_pooling=max_pooling)
    else:
        return encode_layer_norm(input=input, feature_maps=feature_maps, initializer=initializer, activation=activation, training=training, max_pooling=max_pooling)

def encode_layer_unet(input=None, feature_maps=32, initializer=None, activation=tf.nn.relu, training=None, max_pooling=True):
    n = None
    p = tf.layers.conv2d(input, feature_maps, config.kernel_size, kernel_initializer=initializer, padding="valid", activation=None)
    p = tf.layers.batch_normalization(p, training=training, momentum=config.momentum)
    p = activation(p)
    p = tf.layers.conv2d(p, feature_maps, config.kernel_size, kernel_initializer=initializer, padding="valid",
                         activation=None)
    p = tf.layers.batch_normalization(p, training=training, momentum=config.momentum)
    p = activation(p)

    if input.shape[3] != feature_maps:
        input_mod = tf.layers.conv2d(input, feature_maps, config.kernel_size, kernel_initializer=initializer, padding="valid", activation=None)
        input_mod = tf.layers.conv2d(input_mod, feature_maps, config.kernel_size, kernel_initializer=initializer, padding="valid", activation=None)
    else:
        input_mod = input
    p = p + input_mod
    p = tf.nn.relu(p)
    n = p
    if max_pooling:
        p = tf.nn.max_pool(p, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    return n, p

def encode_layer_norm(input=None, feature_maps=32, initializer=None, activation=tf.nn.relu, training=None, max_pooling=True):
    p = tf.layers.conv2d(input, feature_maps, config.kernel_size, kernel_initializer=initializer, padding="same",
                         activation=activation)
    p = tf.layers.conv2d(p, feature_maps, config.kernel_size, kernel_initializer=initializer, padding="same",
                         activation=activation)
    if max_pooling:
        p = tf.nn.max_pool(p, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    return p, None

def encode_layer_resnet(input=None, feature_maps=32, initializer=None, activation=tf.nn.relu, training=None, max_pooling=True):
    p = tf.layers.conv2d(input, feature_maps, 1, kernel_initializer=initializer, padding="same", activation=None)
    p = tf.layers.batch_normalization(p, training=training, momentum=config.momentum)
    p = activation(p)
    #p = tf.layers.conv2d(p, feature_maps // 4, config.kernel_size, kernel_initializer=initializer, padding="same", activation=None)
    #p = tf.layers.batch_normalization(p, training=training, momentum=config.momentum)
    #p = activation(p)
    p = tf.layers.conv2d(p, feature_maps, 1, kernel_initializer=initializer, padding="same", activation=None)
    p = tf.layers.batch_normalization(p, training=training, momentum=config.momentum)

    if input.shape[3] != feature_maps:
        input_mod = tf.layers.conv2d(input, feature_maps, 1, kernel_initializer=initializer, padding="same", activation=None)
    else:
        input_mod = input
    p = p + input_mod
    p = tf.nn.relu(p)
    to_copy = p
    if max_pooling:
        p = tf.nn.max_pool(p, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    return p, to_copy

def decode_layer(input=None, input_size=2048, output_size=1024, out_img_shape=4, batch_size=None, activation=tf.nn.relu):
    n = input.shape[1] * input.shape[2] * input.shape[3]
    std_dev = np.sqrt(2.0 / n.value)
    p = tf.nn.conv2d_transpose(input, filter=tf.Variable(tf.random_normal([3, 3, output_size, input_size], mean=0.0, stddev=0.02)),
                               output_shape=[batch_size, out_img_shape, out_img_shape, output_size], strides=[1, 2, 2, 1], padding="SAME")
    n = p.shape[1] * p.shape[2] * p.shape[3]
    std_dev = np.sqrt(2.0 / n.value)
    p = tf.nn.bias_add(p, tf.Variable(tf.random_normal([output_size], mean=0.0, stddev=0.02)))
    p = activation(p)
    return p


def build_net_unet_v2():
    initializer = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)
    x = tf.placeholder(tf.float32, shape=[None, 101, 101, config.n_channels], name="x")
    x = tf.image.resize_images(x, size=[196, 196],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False, preserve_aspect_ratio=True)
    if config.conv_to_rgb:
        x = tf.image.grayscale_to_rgb(x)
    y = tf.placeholder(tf.float32, shape=[None, 101, 101, config.n_out_layers], name="y")
    training = tf.placeholder(tf.bool, name="training")
    (c1, c1c) = encode_layer_unet(input=x, feature_maps=64, initializer=initializer, training=training) # 196 -> 192
    (c2, c2c) = encode_layer_unet(input=c1c, feature_maps=128, initializer=initializer, training=training)  # 96 -> 92
    (c3, c3c) = encode_layer_unet(input=c2c, feature_maps=256, initializer=initializer, training=training)  # 46 -> 42
    c4, _ = encode_layer_unet(input=c3c, feature_maps=512, initializer=initializer, training=training, max_pooling=False)  # 21 -> 17

    bth_size = tf.placeholder(tf.int32, name="bth_size")

    p = decode_layer(input=c4, input_size=512, output_size=256, out_img_shape=33, batch_size=bth_size) # 17 -> 33
    #cropping c3
    c3 = tf.image.resize_images(c3, size=[33, 33], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    p = tf.concat([p, c3], axis=3)
    p, _ = encode_layer_unet(input=p, feature_maps=256, initializer=initializer, training=training, max_pooling=False)

    p = decode_layer(input=p, input_size=256, output_size=128, out_img_shape=57, batch_size=bth_size)
    # cropping c2
    c2 = tf.image.resize_images(c2, size=[57, 57], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    p = tf.concat([p, c2], axis=3)
    p, _ = encode_layer_unet(input=p, feature_maps=128, initializer=initializer, training=training, max_pooling=False)

    p = decode_layer(input=p, input_size=128, output_size=64, out_img_shape=105, batch_size=bth_size)
    # cropping c1
    c1 = tf.image.resize_images(c1, size=[105, 105], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    p = tf.concat([p, c1], axis=3)
    p, _ = encode_layer_unet(input=p, feature_maps=64, initializer=initializer, training=training, max_pooling=False)


    out_layer = tf.layers.conv2d(p, 1, 1, kernel_initializer=initializer, name="out", activation=None)
    print("outlayer: {}".format(out_layer))

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(out_layer, shape=(-1, 1)),
                                                            labels=tf.reshape(y, shape=(-1, 1)))
    loss = tf.reduce_mean(cross_entropy)
    global current_global_step
    current_global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss)

    return loss, optimizer, tf.nn.sigmoid(out_layer, name="predictlayer"), None, None

def build_net():
    initializer = tf.contrib.layers.xavier_initializer(uniform=False,dtype=tf.float32)
    x = tf.placeholder(tf.float32, shape=[None, config.img_size, config.img_size, config.n_channels], name="x")
    if config.conv_to_rgb:
        x = tf.image.grayscale_to_rgb(x)
    y = tf.placeholder(tf.float32, shape=[None, 101, 101, config.n_out_layers], name="y")
    training = tf.placeholder(tf.bool, name="training")
    p, _ = encode_layer(input=x, feature_maps=64, initializer=initializer, training=training) #52
    p, _ = encode_layer(input=p, feature_maps=128, initializer=initializer, training=training) #26
    p, _ = encode_layer(input=p, feature_maps=256, initializer=initializer, training=training) #13
    p, _ = encode_layer(input=p, feature_maps=512, initializer=initializer, training=training) #7

    p = tf.layers.conv2d(p, 1024, config.kernel_size, kernel_initializer=initializer, padding="same", activation=tf.nn.relu, name="conv-9")
    print(p)
    bth_size = tf.placeholder(tf.int32, name="bth_size")

    p = decode_layer(input=p, input_size=1024, output_size=512, out_img_shape=13, batch_size=bth_size)
    #p = tf.concat([p, c3], axis=3)
    p = decode_layer(input=p, input_size=512, output_size=256, out_img_shape=26, batch_size=bth_size)
    #p = tf.concat([p, c2], axis=3)
    p = decode_layer(input=p, input_size=256, output_size=128, out_img_shape=51, batch_size=bth_size)
    #p = tf.concat([p, c1], axis=3)
    p = decode_layer(input=p, input_size=128, output_size=64, out_img_shape=101, batch_size=bth_size)

    out_layer = tf.layers.conv2d(p, 1, 1, kernel_initializer=initializer, name="out")
    print("outlayer: {}".format(out_layer))
    if config.use_lovasz_loss:
        lovasz = tf.reduce_mean(losses.lovasz_hinge_flat(logits=tf.reshape(out_layer, shape=(1, -1)),
                                                               labels= tf.reshape(y, shape=(1, -1))))
        lovasz_optimize = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(lovasz)
    else:
        lovasz = None
        lovasz_optimize = None
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(out_layer, shape=(-1, 1)),
                                                               labels= tf.reshape(y, shape=(-1, 1)))
    loss = tf.reduce_mean(cross_entropy)
    global current_global_step
    current_global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss, global_step=current_global_step)

    return loss, optimizer, tf.nn.sigmoid(out_layer, name="predictlayer"), lovasz, lovasz_optimize

def shuffle_set(X_train, X_train_mask, X_train_id, salt_coverage):
    assert len(X_train) == len(X_train_mask)
    assert len(X_train) == len (X_train_id)
    p = np.random.RandomState(seed=1977).permutation(len(X_train))
    X_train, X_train_mask, X_train_id =  np.copy(X_train[p]), np.copy(X_train_mask[p]), np.copy(X_train_id[p])
    salt_coverage = salt_coverage.loc[p]
    return X_train, X_train_mask, X_train_id, salt_coverage

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

def train_net(X, mask, id_tr, X_val, mask_val, X_test, loss, optimizer, lovasz_opt, lovasz, out, sess, final_prediction=False):
    print("Training...")
    util.reset_vars(sess)
    start_t = time.time()
    step = []
    cost_batch = []
    ious = None
    df_empty = pd.DataFrame({th: [] for th in config.thresholds})
    df_empty['epoch'] = []
    loss_fn = loss
    optimizer_fn = optimizer
    batch_norm_ops = []
    if config.user_resnet or config.use_original_unet:
        print("Use batch norm ops...")
        batch_norm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    g = tf.get_default_graph()
    for i in range(config.epochs):
        print("Epoch {}, elapsed min {:.2f}".format(i, ((time.time() - float(start_t)) / 60.0)))
        if (config.epochs - (i + 1)) <= config.lovasz_epochs:
            loss_fn = lovasz
            optimizer_fn = lovasz_opt
        ii = 0
        for batch, mask_batch, id_batch in get_next_batch(X, mask, id_tr):
            if config.user_resnet or config.use_original_unet:
                sess.run([optimizer_fn, batch_norm_ops], feed_dict={"x:0": batch, "y:0": mask_batch, "training:0": True, "bth_size:0": len(batch)})
            else:
                sess.run(optimizer_fn, feed_dict={"x:0": batch, "y:0": mask_batch, "training:0": True, "bth_size:0": len(batch)})
            if(not final_prediction and ii % config.display_steps == 0):
                print("Validation results...")
                cost = sess.run(loss_fn, feed_dict={"x:0": batch, "y:0": mask_batch, "training:0": False, "bth_size:0": len(batch)})
                cost_test = sess.run(loss_fn, feed_dict={"x:0": X_val, "y:0": mask_val, "training:0": False, "bth_size:0": X_val.shape[0]})
                out_val = sess.run(out, feed_dict={"x:0": X_val, "training:0": False, "bth_size:0": X_val.shape[0]})
                out_val = out_val.reshape((-1, config.img_size * config.img_size), order="F")
                mask_val_tmp = mask_val.reshape((-1, config.img_size * config.img_size), order="F")
                #def_res = util.devise_complete_iou_results(out_val, mask_val_tmp, config.thresholds,
                #                                           config.kaggle_thresholds)
                '''
                out_val = np.empty((0, config.img_size * config.img_size))
                for j in range(int(X_val.shape[0] / config.pred_step)):
                    out_val_pred = sess.run(out, feed_dict={"x:0": X_val[j * config.pred_step:(j + 1) * config.pred_step, :, :, :], "training:0": False, "bth_size:0": config.pred_step})
                    out_val = np.append(out_val, out_val_pred.reshape((-1, config.img_size * config.img_size), order="F"), axis=0)
                if (j + 1) * config.pred_step < X_val.shape[0]:
                    left_val_set = X_val[(j + 1) * config.pred_step:, :, :, :]
                    out_val_pred = sess.run(out, feed_dict={"x:0": left_val_set, "training:0": False, "bth_size:0": left_val_set.shape[0]})
                    out_val = np.append(out_val, out_val_pred.reshape((-1, config.img_size * config.img_size), order="F"))

                out_val = out_val.reshape((-1, config.img_size * config.img_size), order="F")
                mask_val_tmp = mask_val.reshape((-1, config.img_size * config.img_size), order="F")
                if config.save_model and i % config.save_model_step == 0:
                    util.persist(os.path.join(config.CACHE_PATH, "out_val-{}.pck".format(i)), out_val)
                    util.persist(os.path.join(config.CACHE_PATH, "mask_val-{}.pck".format(i)), mask_val_tmp)
                '''
                print("Tot val samples {}".format(out_val.shape[0]))
                def_res = util.devise_complete_iou_results(out_val, mask_val_tmp, config.thresholds, config.kaggle_thresholds)
                df_calc = pd.DataFrame(def_res)
                df_calc['epoch'] = (ii * (i + 1))
                df_empty = df_empty.append(df_calc)

                print("Epoch {} Iteration {}".format(i, ii))
                print("Loss -> train: {:.4f} test: {:.4f}".format(cost, cost_test))
                print("IoUs {}".format(def_res))
                step.append(ii * (i + 1))
                cost_batch.append(cost)
            ii += 1
        print("Total batches {}".format(ii))
    cost_df = pd.DataFrame({"epoch": step, "cost_batch": cost_batch})
    if config.save_model and i % config.save_model_step == 0:
        util.save_tf_model(sess, current_global_step)
    #cost = sess.run(loss, feed_dict={"x:0": X, "y:0": mask, "training:0": False, "bth_size:0": X.shape[0]})
    #cost_test = sess.run(loss, feed_dict={"x:0": X_val, "y:0": mask_val, "training:0": False, "bth_size:0": X_val.shape[0]})
    #print("Loss -> train: {:.4f}, test: {:.4f}".format(cost, cost_test))
    print("Total time {} sec".format(time.time() - start_t))
    if final_prediction:
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
            left_test_set = X_test[(j + 1) * config.pred_step:, :, :, :]
            y1 = sess.run(out, feed_dict={"x:0": left_test_set, "training:0": False, "bth_size:0": left_test_set.shape[0]})
            y_aug = []
            y_aug.append(y1.reshape((-1, config.img_size * config.img_size), order="F"))
            for aug in X_test_aug:
                y_aug.append(sess.run(out, feed_dict={"x:0": aug[(j + 1) * config.pred_step:, :, :, :],
                                                      "training:0": False, "bth_size:0": config.pred_step}).reshape(
                    (-1, config.img_size * config.img_size), order="F"))
            y_aug = np.array(y_aug)
            y_mean = y_aug.mean(axis=0)
            y_pred = np.append(y_pred, y_mean, axis=0)
    else:
        y_pred = np.empty((0, config.img_size * config.img_size))

    return y_pred, df_empty, cost_df



if __name__ == "__main__":
    X_train, X_train_id, X_train_mask, X_test, X_test_id = preprocess()
    df_coverage = pd.read_csv(os.path.join(config.config["base_path"], "coverage-cat.csv"))
    X_train, X_train_mask, X_train_id, df_coverage = shuffle_set(X_train, X_train_mask, X_train_id, df_coverage)

    #X_reduced_train, X_mask_red, X_reduced_train_id, X_validation, X_mask_validation, X_validation_id = train_validation(X_train, X_train_mask, X_train_id)
    sess = util.reset_tf(None)
    if config.use_original_unet:
        print("Using original UNET model")
        loss, optimizer, out, lovasz, lovasz_opt = build_net_unet_v2()
    else:
        loss, optimizer, out, lovasz, lovasz_opt = build_net()
    if os.path.isfile(os.path.join(config.CHECKPOINTS_PATH, "{}.meta".format(config.MODEL_FILENAME))):
        print("Restoring model...")
        saver = tf.train.import_meta_graph(os.path.join(config.CHECKPOINTS_PATH, "{}.meta".format(config.MODEL_FILENAME)))
        saver.restore(sess, tf.train.latest_checkpoint(config.CHECKPOINTS_PATH))
        # Now, let's access and create placeholders variables and
        # create feed-dict to feed new data
        graph = tf.get_default_graph()
        print(graph)
        # Now, access the op that you want to run.
        op_to_restore = graph.get_tensor_by_name("predictlayer:0")
        x_in = graph.get_tensor_by_name("x:0")
        bth_size = graph.get_tensor_by_name("bth_size:0")
        training = graph.get_tensor_by_name("training:0")
        print("Devising testset results...")
        y_pred = np.empty((0, config.img_size * config.img_size))
        X_test_aug = []
        if config.tta:
            X_test_aug = util.tta_augment(X_test)
        for j in range(int(X_test.shape[0] / config.pred_step)):
            print("[{}] predicting...".format((j + 1)))
            y1 = sess.run(op_to_restore, feed_dict={x_in: X_test[j * config.pred_step:(j + 1) * config.pred_step, :, :, :],
                                          training: False, bth_size: config.pred_step})
            y_aug = []
            y_aug.append(y1.reshape((-1, config.img_size * config.img_size), order="F"))
            for trans, aug in X_test_aug.items():
                y_tmp = sess.run(op_to_restore, feed_dict={x_in: aug[j * config.pred_step:(j + 1) * config.pred_step, :, :, :],
                                                 training: False, bth_size: config.pred_step})
                y_tmp = util.deaugment(y_tmp, trans)
                y_aug.append(y_tmp.reshape((-1, config.img_size * config.img_size), order="F"))
            y_aug = np.array(y_aug)
            y_mean = y_aug.mean(axis=0)
            y_pred = np.append(y_pred, y_mean, axis=0)

        if (j + 1) * config.pred_step < X_test.shape[0]:
            print("[{}] predicting...".format((j + 1)))
            left_test_set = X_test[(j + 1) * config.pred_step:, :, :, :]
            y1 = sess.run(out, feed_dict={"x:0": left_test_set, "training:0": False, "bth_size:0": left_test_set.shape[0]})
            y_aug = []
            y_aug.append(y1.reshape((-1, config.img_size * config.img_size), order="F"))
            for aug in X_test_aug:
                y_aug.append(sess.run(op_to_restore, feed_dict={x_in: aug[(j + 1) * config.pred_step:, :, :, :],
                                                      training: False, bth_size: config.pred_step}).reshape(
                    (-1, config.img_size * config.img_size), order="F"))
            y_aug = np.array(y_aug)
            y_mean = y_aug.mean(axis=0)
            y_pred = np.append(y_pred, y_mean, axis=0)
    else:
        ious = []
        costs = []
        ths = []
        sss = StratifiedShuffleSplit(n_splits=config.n_cv, test_size=config.validation_perc, random_state=42)
        #X_train, X_train_id, X_train_mask
        i = 1
        final_prediction = False
        if not config.skip_cv:
            for train_index, val_index in sss.split(X_train, df_coverage.salt):
                print("Cross validation fold {}".format(i))
                X_reduced_train = np.copy(X_train[train_index,:,:,:])
                X_reduced_train_id = np.copy(X_train_id[train_index])
                X_mask_red = np.copy(X_train_mask[train_index,:,:,:])
                X_validation = np.copy(X_train[val_index,:,:,:])
                X_mask_validation = np.copy(X_train_mask[val_index,:,:,:])

                y_pred, df_empty, cost_df = train_net(X_reduced_train, X_mask_red, X_reduced_train_id, X_validation, X_mask_validation, X_test, loss, optimizer, lovasz_opt, lovasz, out, sess)
                ious.append(("ious_val-{}.csv".format(i), df_empty))
                costs.append(("costs_train-{}.csv".format(i), cost_df))
                th_max = df_empty.tail(1)[config.thresholds].idxmax(axis=1).values[0]
                ths.append(th_max)

                #df_empty.to_csv(os.path.join(config.CACHE_PATH, "ious_val.csv"))
                #cost_df.to_csv(os.path.join(config.CACHE_PATH, "costs_train.csv"))

                i += 1
            print("Thresholds {}".format(ths))
            th_max = sum(ths) / float(len(ths))
            print("Max threshold {}".format(th_max))
            for c in costs:
                c[1].to_csv(os.path.join(config.CACHE_PATH, c[0]))
            for c in ious:
                c[1].to_csv(os.path.join(config.CACHE_PATH, c[0]))
        else:
            th_max = 0.48
        final_prediction = True
        y_pred, df_empty, cost_df = train_net(X_train, X_train_mask, X_train_id, [],
                                              [], X_test, loss, optimizer, lovasz_opt, lovasz, out, sess,
                                              final_prediction=final_prediction)

    print("Generating results: adopted threshold is: {}".format(th_max))
    no_test = y_pred.shape[0]
    y_pred_def = y_pred
    util.persist(os.path.join(config.CACHE_PATH, "pred_def.pck"), y_pred_def)
    #y_pred_def.shape[0] == 18000

    for curr_th in config.thresholds:
        y_pred_def = (y_pred_def > curr_th) * 1

        to_submit = {idx: util.convert_for_submission(y_pred_def[i,:]) for i, idx in enumerate(X_test_id)}
        #assert X_test_id.shape[0] == 18000
        sub = pd.DataFrame.from_dict(to_submit, orient='index')
        sub.index.names = ['id']
        sub.columns = ['rle_mask']
        sub.to_csv(os.path.join(config.CACHE_PATH, "submit-{}.csv".format(curr_th)))

    print("Done")


