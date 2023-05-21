from flask import Flask, request, jsonify
import time
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
import os
from os import path
from os.path import abspath
import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from env import mask_dir

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
path_value = str(os.environ.get('PATH'))
CUDA_bin = ';C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\bin;'
CUDA_libnvvp = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\libnvvp;'

os.environ['PATH'] = path_value + CUDA_bin + CUDA_libnvvp
# print(os.environ['CUDA_VISIBLE_DEVICES'])

import tensorflow as tf
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
import neuralgym as ng
from inpaint_model import InpaintCAModel

app = Flask(__name__)

out_dir = Path("temporary")
out_dir.mkdir(parents=True, exist_ok=True)

@app.route('/')
def index():
    return 'Hello, World!'

#try with gpu
@app.route('/choose-mask', methods=['GET'])
def choose_mask():
    start_time = time.perf_counter()
    FLAGS = ng.Config('inpaint.yml', False)

    mask_name = "dilated_" + request.args['mask_file']
    mask_path = mask_dir / mask_name
    image_path = mask_dir / "image.png"
    model = InpaintCAModel()
    image = cv2.imread(abspath(image_path))
    mask = cv2.imread(abspath(mask_path))

    image = cv2.resize(image, None, fx=0.3, fy=0.3)
    mask = cv2.resize(mask, None, fx=0.3, fy=0.3)
    assert image.shape == mask.shape

    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)
    
    input_image = tf.constant(input_image, dtype=tf.float32)
    output = model.build_server_graph(FLAGS, input_image)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    # load pretrained model
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable("model_logs", from_name)
        assign_ops.append(tf.assign(var, var_value))
    
    with tf.Session(config=sess_config) as sess:
        sess.run(assign_ops)
        result = sess.run(output)
        
        img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_path).name}"
        cv2.imwrite(abspath(img_inpainted_p), result[0][:, :, ::-1])

    with open(img_inpainted_p, "rb") as f:
        image_file = f.read()

    end_time = time.perf_counter()
    latency = (end_time - start_time) 
    print("Request latency: %.5f seconds" % latency)
    return image_file

if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=5001)