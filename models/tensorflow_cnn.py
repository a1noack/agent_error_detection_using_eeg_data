from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split


tf.logging.set_verbosity(tf.logging.INFO)

# global variable denoting the length of each episode
eeg_length = -1

def expand_dataset(df, depth, crop_stride, orig_len, labels):
    # calculate new length of eeg readings
    new_len = orig_len - depth

    df_ = pd.DataFrame()
    for i in range(0, depth, crop_stride):
        # create a list of the readings to keep in this iteration
        eeg_keep = list(range(i, new_len + i))

        # add in the labels to that list
        columns_keep = eeg_keep + labels

        # keep the resulting dataframe
        df_i = df[columns_keep]

        # rename the columns in the resulting dataframe so that they all line up, index-wise
        df_i.columns = list(range(new_len)) + labels

        # append the iteration dataframe to the placeholder
        df_ = df_.append(df_i)
    return df_

#make hyperparameter string
def hyper_param_string(learn_rate, use_two_cv, use_two_fc):
    name = "lr_"+str(learn_rate)+"_use2cv_"+str(use_two_cv)+"_use2fc_"+str(use_two_fc)
    print(name)
    return name

# shallow convnet model
#params is a dict of the hyperparameters for the model
def cnn(features, labels, mode, params):
    global eeg_length

    # reshape input to account for minibatch size. Also, the number of channels (1 in this case) is specified.
    x = tf.reshape(features["x"], [-1, eeg_length, 1])

    # temporal convolution
    with tf.name_scope("conv1_pool1"):
        x = tf.layers.conv1d(
            inputs=x,
            filters=params["conv1_kernel_num"],
            kernel_size=params["conv1_filter_size"],
            strides=params["conv1_stride"],
            padding='valid',
            use_bias=True,
            activation=None,
            name="conv1")
        # with some luck, conv1 will have shape [-1,289,24] ie [batch_size, readings, num_kernels]
        # pooling layer
        x = tf.layers.max_pooling1d(
            inputs=x,
            pool_size=params["pool1_size"],
            strides=params["pool1_stride"],
            name="pool1")

    #only perform these ops if two layers of convolutions are specified
    if(params["use_two_cv"]):#60/100   .4(.4) + .6(.6) =
        with tf.name_scope("conv2_pool2"):
            x = tf.layers.conv1d(
                inputs=x,
                filters=params["conv2_kernel_num"],
                kernel_size=params["conv2_filter_size"],
                strides=params["conv2_stride"],
                padding='valid',
                use_bias=True,
                activation=None,
                name="conv2")
            # pooling layer
            x = tf.layers.max_pooling1d(
                inputs=x,
                pool_size=params["pool2_size"],
                strides=params["pool2_stride"],
                name="pool2")

    #flatten the data to pass through the dense layers
    x_shape = x.shape.as_list()
    x = tf.reshape(x, [-1, x_shape[1] * x_shape[2]])

    if(params["num_fc"] >= 1):
        with tf.name_scope("dense1"):
            x = tf.layers.dense(
                inputs=x,
                units=params["dense1_neurons"],
                activation=params["activations"],
                name="dense1")

    #only perform this if two fully connected layers are specified
    if(params["num_fc"] >= 2):
        with tf.name_scope("dense2"):
            x = tf.layers.dense(
                inputs=x,
                units=params["dense2_neurons"],
                activation=params["activations"],
                name="dense2")

    # rate = .4 => 40% of the output from the dense layers will be randomly dropped during training
    # training checks to see if the CNN is in training mode. if so, the dropout proceeds.
    with tf.name_scope("dropout"):
        x = tf.layers.dropout(
            inputs=x,
            rate=params['dropout_rate'],
            training=mode == tf.estimator.ModeKeys.TRAIN,
            name="dropout")

    # logits output has shape [batch_size, 2]
    with tf.name_scope("logits"):
        logits = tf.layers.dense(
            inputs=x,
            units=2,
            name="logits")

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    # if in TRAIN or EVAL mode, calculate loss and backpropagate
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # record metrics for display in tensorboard
    with tf.name_scope("metrics"):
        precision = tf.metrics.precision(
            labels=labels,
            predictions=predictions["classes"])

        recall = tf.metrics.recall(
            labels=labels,
            predictions=predictions["classes"])
        tf.summary.scalar('train_cross_entropy', loss)

        accuracy = tf.metrics.accuracy(
            labels=labels,
            predictions=predictions["classes"])
        tf.summary.scalar('train_accuracy', accuracy[1])

        tf.summary.scalar('train_precision', precision[1])

        tf.summary.scalar('train_recall', recall[1])

        predictions_float = tf.cast(predictions['classes'], dtype=tf.float32)
        win_prediction_freq = tf.reshape(tf.reduce_mean(predictions_float), shape=()) #convert to scalar
        tf.summary.scalar('win_prediction_freq', win_prediction_freq)

        tf.summary.histogram('train_prediction_distributions', predictions['probabilities'])

        # prediction_distribution = np.mean(tf.Session().run(predictions['probabilities']), axis=0)
        # print(predictions['probabilities'])
        # predict_win = prediction_distribution[1]
        # predict_loss = prediction_distribution[0]

        # tf.summary.scalar('predict_win', predict_win)
        # tf.summary.scalar('predict_loss', predict_loss)

    # if in training mode, calculate loss and step to take
    if mode == tf.estimator.ModeKeys.TRAIN:
        if(params["optimizer"] == "GD"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=params["learn_rate"])
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=params["learn_rate"])
        #i think tf.train.get_global_step() is self explanatory here
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # if in eval mode, eval!
    eval_metric_ops = {"eval_accuracy": accuracy,
                       "eval_precision": precision,
                       "eval_recall": recall}
                       # "predict_win": predict_win,
                       # "predict_loss": predict_loss}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # make eeg_length global
    global eeg_length
    # training only within a subject?
    # intra_sbj = True
    sbj_num = 6

    labels = ["episode_type", "trial_type", "trial_num",
              "episode_num", "is_corrupt", "action_index",
              "trimmed_data", "left", "subject_num", "win"]

    # load data
    data_full = pd.read_pickle("data/data_100hz_intrasubject_scaled.pkl")

    # #different hyperparameters I'd like to test
    learning_rates = [.0001, .00005, .00001]
    # two_conv_layers = [True]
    # two_dense_layers = [True]
    #
    conv1_filter_sizes = [30, 15]
    # conv2_filter_sizes = [30, 60]
    #
    conv1_kernel_nums = [30, 10]
    # conv2_kernel_nums = [75, 100]
    #
    conv1_strides = [3, 6]

    pool1__sizes = [10, 5]

    dropout_rates = [.4, .6]

    # conv2_strides = [5, 10]
    #
    # dense1_layer_sizes = [100, 200]
    # dense2_layer_sizes = [50, 100]

    batch_sizes = [50, 25]

    # learn_rate = .0001
    use_two_cv = False
    num_fcs = [0,1,2]
    conv1_stride = 3
    pool1_size = 20
    batch_size = 25

    conv1_filter_size = 25
    dropout_rate = .4
    learn_rate = .00001
    conv1_kernel_num = 25

    # for conv1_filter_size in conv1_filter_sizes:
    #     for conv1_kernel_num in conv1_kernel_nums:
    #         for conv1_stride in conv1_strides:
    #             for fc1_size in dense1_layer_sizes:
    # for learn_rate in learning_rates:
    #     for use_two_fc in two_dense_layers:
    #         for use_two_cv in two_conv_layers:
                #create the log name for this model
    for sbj_num in range(1,11):
        # remove data from other subjects if toggled
        data_sbj = data_full[data_full.subject_num == sbj_num].reset_index(drop=True)

        train, test = train_test_split(data_sbj, test_size=.25)

        expand = False

        if(expand):
            train = expand_dataset(train, 40, 5, 375, labels)
            test = expand_dataset(test, 40, 5, 375, labels)

        x_train = np.array(train.drop(labels, axis=1), dtype='float32')
        y_train = np.array(train['win'], dtype='float32')

        x_test = np.array(test.drop(labels, axis=1), dtype='float32')
        y_test = np.array(test['win'], dtype='float32')

        eeg_length = x_train[0].size
        # x = data_sbj.drop(labels, axis=1)
        # # fill in NaN values
        # x = x.fillna(x.mean())
        # # convert to numpy array rank 2
        # x = np.array(x, dtype="float32")
        # y = np.array(data_sbj['win'], dtype='float32')
        #
        # # get the number of readings per episode

        #
        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=27)

        # for conv1_filter_size in conv1_filter_sizes:
        #     for conv1_kernel_num in conv1_kernel_nums:
        #         for dropout_rate in dropout_rates:
        #             for learn_rate in learning_rates:
        model_name = "sbj_num_"+str(sbj_num)+"_cvf_"+str(conv1_filter_size)+\
                     "_dr_"+str(dropout_rate)+"_lr_"+str(learn_rate)+\
                     "_cvk_" +str(conv1_kernel_num)+"_cvs_"+str(conv1_stride)+"_pos_"+str(pool1_size)+\
                     "_bs_"+str(batch_size)

        hyp_params = {"optimizer":"Adam",
                      "use_f1_as_loss":False,
                      "learn_rate":learn_rate,
                      "batch_size":batch_size,
                      "steps":100,
                      "use_two_cv":use_two_cv,
                      "num_fc":num_fcs[1],
                      "conv1_filter_size":conv1_filter_size,
                      "conv1_kernel_num":conv1_kernel_num,
                      "conv1_stride":conv1_stride,
                      "pool1_size":pool1_size,
                      "pool1_stride":3,
                      "conv2_filter_size":15,
                      "conv2_kernel_num":80,
                      "conv2_stride":3,
                      "pool2_size":5,
                      "pool2_stride":2,
                      "dense1_neurons":50,
                      "dense2_neurons":200,
                      "activations":tf.nn.relu,
                      "dropout_rate":dropout_rate}

        # Create the Estimator and the model directory is where the model stats are saved
        #the log_name is dynamically decided based on the hyperparams above
        model_dir = "models/intrasubject_final_4/"+model_name
        classifier = tf.estimator.Estimator(
            params = hyp_params,
            model_fn = cnn,
            model_dir = model_dir)

        # Set up logging for predictions
        # these tensors are printed to the output every 50 iterations
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

        # def predict_input_fn():
        #     rand = np.random.randint(data_sbj.shape[0])
        #     rand_samp = data_sbj[data_sbj.index == rand]
        #     eeg = rand_samp.drop(labels, axis=1)
        #     eeg = np.array(eeg)
        #     print('TRUE:', rand_samp['win'].values[0])
        #     return eeg
        #
        # predict_results = classifier.predict(input_fn=predict_input_fn)
        # print("PREDICT:", list(predict_results))

        for i in range(15):
            # Train the model
            #this function returns a dictionary of features -> targets
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": x_train},
                y=y_train,
                batch_size=hyp_params["batch_size"],
                num_epochs=None,
                shuffle=True)
            classifier.train(
                input_fn=train_input_fn,
                steps=hyp_params["steps"],
                hooks=[logging_hook])

            # Evaluate the model and print results
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": x_test},
                y=y_test,
                num_epochs=1,
                shuffle=False)
            eval_results = classifier.evaluate(input_fn=eval_input_fn)
            print(eval_results)

        saver = tf.train.Saver()




if __name__ == "__main__":
    tf.app.run()