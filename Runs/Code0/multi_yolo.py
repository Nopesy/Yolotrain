# yolo.py

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, LeakyReLU,
    UpSampling2D, Concatenate, Reshape
)
from tensorflow.keras.models import Model

def conv_block(x, filters, kernel_size, strides=1):
    x = Conv2D(filters, kernel_size, strides=strides,
               padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def decode_predictions(pred, anchors, input_shape):
    """
    pred: (batch, S, S, B, 5+num_classes)
    anchors: array shape (B,2) in pixel units
    returns: (batch, S, S, B, 4+1+num_classes) with (x1,y1,x2,y2,obj,classes…)
    """
    grid_size = tf.shape(pred)[1]
    B = tf.shape(pred)[3]
    num_classes = tf.shape(pred)[4] - 5

    # create grid
    grid_x = tf.range(grid_size, dtype=tf.float32)
    grid_y = tf.range(grid_size, dtype=tf.float32)
    cx, cy = tf.meshgrid(grid_x, grid_y)
    cx = tf.reshape(cx, (1, grid_size, grid_size, 1))
    cy = tf.reshape(cy, (1, grid_size, grid_size, 1))

    tx = pred[..., 0:1]
    ty = pred[..., 1:2]
    tw = pred[..., 2:3]
    th = pred[..., 3:4]
    to = pred[..., 4:5]
    tc = pred[..., 5:]

    bx = (tf.sigmoid(tx) + cx) / tf.cast(grid_size, tf.float32)
    by = (tf.sigmoid(ty) + cy) / tf.cast(grid_size, tf.float32)

    # turn anchors to tensor [1,1,1,B,2]
    anchors = tf.cast(tf.reshape(anchors, [1,1,1,B,2]), pred.dtype)
    bw = tf.exp(tw) * anchors[..., 0:1] / input_shape[1]
    bh = tf.exp(th) * anchors[..., 1:2] / input_shape[0]

    x1 = bx - bw/2
    y1 = by - bh/2
    x2 = bx + bw/2
    y2 = by + bh/2

    obj = tf.sigmoid(to)
    classes = tf.sigmoid(tc)

    return tf.concat([x1, y1, x2, y2, obj, classes], axis=-1)

def Darknet19(input_shape=(416,416,3), num_classes=20, anchors=[(10,13),(16,30),(33,23)]):
    """
    Anchor‑based single‑scale YOLO head.
    anchors: list of (w,h) in pixels.
    """
    B = len(anchors)
    inputs = Input(shape=input_shape)

    # ——— backbone ———
    x = conv_block(inputs,  32, 3);    x = tf.keras.layers.MaxPool2D(2)(x)
    x = conv_block(x,  64, 3);          x = tf.keras.layers.MaxPool2D(2)(x)
    x = conv_block(x, 128, 3); x = conv_block(x,  64, 1); x = conv_block(x, 128, 3)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = conv_block(x, 256, 3); x = conv_block(x, 128, 1); x = conv_block(x, 256, 3)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = conv_block(x, 512, 3); x = conv_block(x, 256, 1); x = conv_block(x, 512, 3)
    x = conv_block(x, 256, 1); x = conv_block(x, 512, 3)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = conv_block(x,1024, 3); x = conv_block(x, 512, 1); x = conv_block(x,1024, 3)
    x = conv_block(x, 512, 1); x = conv_block(x,1024, 3)

    # ——— detection head ———
    pred_filters = B * (5 + num_classes)
    y = Conv2D(pred_filters, 1, padding='same')(x)

    grid_h = input_shape[0] // (2**5)
    grid_w = input_shape[1] // (2**5)
    outputs = Reshape((grid_h, grid_w, B, 5 + num_classes))(y)

    model = Model(inputs, outputs, name="Darknet19_Anchors")
    model.anchors = np.array(anchors, dtype=np.float32)
    return model

def yolo_loss(y_true, y_pred,
              lambda_coord=5.0, lambda_noobj=1.0):
    """
    Assumes y_true and y_pred both (batch, S, S, B, 5+num_classes)
    – y_true[...,4] is 1 for the one matched anchor, 0 elsewhere
    """
    true_xy    = y_true[..., 0:2]
    true_wh    = y_true[..., 2:4]
    true_obj   = y_true[..., 4:5]
    true_class = y_true[..., 5:]

    pred_xy    = tf.sigmoid(y_pred[..., 0:2])
    pred_wh    = y_pred[..., 2:4]   # raw; matched in parse_example
    pred_obj   = tf.sigmoid(y_pred[..., 4:5])
    pred_class = y_pred[..., 5:]

    obj_mask   = true_obj
    noobj_mask = 1.0 - obj_mask

    loss_xy    = tf.reduce_sum(obj_mask * tf.square(pred_xy - true_xy))
    loss_wh    = tf.reduce_sum(obj_mask * tf.square(pred_wh - true_wh))
    loss_loc   = lambda_coord * (loss_xy + loss_wh)

    loss_obj   = tf.reduce_sum(obj_mask * tf.square(pred_obj - true_obj))
    loss_noobj = lambda_noobj * tf.reduce_sum(noobj_mask * tf.square(pred_obj))

    loss_class = tf.reduce_sum(obj_mask * tf.square(pred_class - true_class))

    return loss_loc + loss_obj + loss_noobj + loss_class