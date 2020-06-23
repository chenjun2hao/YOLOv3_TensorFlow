# coding: utf-8
# for more details about the yolo darknet weights file, refer to
# https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe

from __future__ import division, print_function

import os
import sys
import tensorflow as tf
import numpy as np

from model import yolov3
from utils.misc_utils import parse_anchors, load_weights, freeze_graph

num_class = 80      # 0. 类别和输入尺寸和训练时保持一直
img_size = 416
# ckpt_path = "./checkpoint/best_model_Epoch_20_step_21167_mAP_0.6431_loss_13.9490_lr_0.0001"          # 1. 训练得到的模型位置
# anchors = parse_anchors('./data/my_data/helmet_anchors.txt')      # 2. 使用训练时的anchors文件
# save_path = './checkpoint/yolov3_helmet.pd'
ckpt_path = "./yolov3_weights/yolov3.ckpt"
save_path = './checkpoint/yolov3_coco.pd'
anchors = parse_anchors("./data/yolo_anchors.txt")

model = yolov3(num_class, anchors)
with tf.Session() as sess:
    inputs = tf.placeholder(tf.float32, [1, img_size, img_size, 3])     # 3. 输入用placeholder占位

    with tf.variable_scope('yolov3'):
        feature_map = model.forward(inputs, False)

    saver = tf.train.Saver(var_list=tf.global_variables(scope='yolov3'))
    saver.restore(sess, ckpt_path)

    feature_map_1, feature_map_2, feature_map_3 = feature_map
    our_out_node_names = ["yolov3/yolov3_head/feature_map_1", "yolov3/yolov3_head/feature_map_2", "yolov3/yolov3_head/feature_map_3"]

    freeze_graph(sess, save_path, our_out_node_names)
    print('TensorFlow model checkpoint has been saved to {}'.format(save_path))



