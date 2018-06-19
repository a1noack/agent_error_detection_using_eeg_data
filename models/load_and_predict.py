import numpy as np, tensorflow as tf, pandas as pd, os, sys
sys.path.append(os.getcwd())
from tensorflow_cnn import cnn

# this is where we will send it through episode through the model
classifier = tf.estimator.Estimator(
    model_fn=cnn,
    model_dir='intrasubject_final/sbj_num_6_cvf_25_dr_0.4_lr_1e-05_cvk_10_cvs_3_pos_20_bs_25',
)

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.random.randint(-20, 20, 375)},
    y= np.random.rand()<.5,
    shuffle=False)

classifier.predict(
    input_fn=predict_input_fn
)