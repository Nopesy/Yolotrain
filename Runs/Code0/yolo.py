import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, LeakyReLU,
    MaxPool2D, Reshape
)
from tensorflow.keras.models import Model


def conv_block(x, filters, kernel_size, strides=1):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def Darknet19(input_shape=(416, 416, 3), num_classes=20, B=1):
    """
    Darknet-19 backbone with a simplified YOLO-style detection head
    using one bounding box per grid cell (no anchors).
    Outputs shape: (batch, grid_h, grid_w, 1, 5+num_classes).
    """
    inputs = Input(shape=input_shape)
    x = conv_block(inputs, 32, 3)
    x = MaxPool2D(2, 2, padding='same')(x)

    x = conv_block(x, 64, 3)
    x = MaxPool2D(2, 2, padding='same')(x)

    x = conv_block(x, 128, 3)
    x = conv_block(x, 64, 1)
    x = conv_block(x, 128, 3)
    x = MaxPool2D(2, 2, padding='same')(x)

    x = conv_block(x, 256, 3)
    x = conv_block(x, 128, 1)
    x = conv_block(x, 256, 3)
    x = MaxPool2D(2, 2, padding='same')(x)

    x = conv_block(x, 512, 3)
    x = conv_block(x, 256, 1)
    x = conv_block(x, 512, 3)
    x = conv_block(x, 256, 1)
    x = conv_block(x, 512, 3)
    x = MaxPool2D(2, 2, padding='same')(x)

    x = conv_block(x, 1024, 3)
    x = conv_block(x, 512, 1)
    x = conv_block(x, 1024, 3)
    x = conv_block(x, 512, 1)
    x = conv_block(x, 1024, 3)

    # Detection head: one box per cell
    pred_filters = B * (5 + num_classes)
    y = Conv2D(pred_filters, 1, padding='same')(x)

    grid_h = input_shape[0] // (2**5)
    grid_w = input_shape[1] // (2**5)
    outputs = Reshape((grid_h, grid_w, B, 5 + num_classes))(y)

    return Model(inputs, outputs, name='Darknet19_SingleBox')


def yolo_loss(y_true, y_pred, lambda_coord=5.0, lambda_noobj=1.0):
    """
    YOLO loss for one-box-per-cell model: localization + confidence + classification.
    """
    # Split targets and predictions
    true_xy = y_true[..., 0:2]
    true_wh = y_true[..., 2:4]
    true_obj = y_true[..., 4:5]  # keep dims
    true_class = y_true[..., 5:]

    pred_xy = y_pred[..., 0:2]
    pred_wh = y_pred[..., 2:4]
    pred_obj = y_pred[..., 4:5]
    pred_class = y_pred[..., 5:]

    # Masks
    obj_mask = true_obj
    noobj_mask = 1.0 - obj_mask

    # Localization (only where object exists)
    loss_xy = tf.reduce_sum(obj_mask * tf.square(pred_xy - true_xy))
    loss_wh = tf.reduce_sum(obj_mask * tf.square(pred_wh - true_wh))
    loss_loc = lambda_coord * (loss_xy + loss_wh)

    # Confidence
    loss_obj = tf.reduce_sum(obj_mask * tf.square(pred_obj - true_obj))
    loss_noobj = lambda_noobj * tf.reduce_sum(noobj_mask * tf.square(pred_obj))

    # Classification (only where object exists)
    loss_class = tf.reduce_sum(obj_mask * tf.square(pred_class - true_class))

    return loss_loc + loss_obj + loss_noobj + loss_class


if __name__ == '__main__':
    model = Darknet19(input_shape=(416, 416, 3), num_classes=20, B=1)
    model.compile(optimizer='adam', loss=yolo_loss)
    model.summary()
