import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, LeakyReLU,
    MaxPool2D, Reshape
)
import math

import tensorflow as tf
from tensorflow.keras.models import Model
from typing import List, Tuple, Dict

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
info = tf.sysconfig.get_build_info()
print("Built with CUDA:", info.get("cuda_version"))
print("Built with cuDNN:", info.get("cudnn_version"))

ANCHORS_PX: List[Tuple[float, float]] = [
    (55.4, 80.7), (139.2, 248.4), (327.9, 319.7)
]
anchors_px = tf.constant(ANCHORS_PX, tf.float32)  # shape (NUM_ANCHORS,2)
NUM_ANCHORS     = len(ANCHORS_PX)

def conv_block(x, filters, kernel_size, strides=1):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    return x


def Darknet19(input_shape=(416, 416, 3), num_classes=20, B=3):

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

    # 5 max pools each halving dims
    grid_h = input_shape[0] // (2**5)
    grid_w = input_shape[1] // (2**5)
    outputs = Reshape((grid_h, grid_w, B, 5 + num_classes))(y)

    return Model(inputs, outputs, name='Darknet19_SingleBox')

# def yolo_loss(y_true, y_pred, lambda_coord=5.0, lambda_noobj=0.05):
#     """
#     YOLO loss with:
#       0) anchors precomputed in “cells”
#       1) split ground-truth vs predictions
#       2) build the cell-grid
#       3) decode predicted tx,ty → cx,cy and tw,th → w,h
#       4) decode ground-truth the same way
#       5) compute IoU per anchor
#       6) localization loss = λ_coord * Σ (1 – IoU) over positives
#       7) objectness loss = sigmoid-CE, normalized over pos/neg
#       8) classification  loss = softmax-CE, normalized over positives
#     """

#     # ── 0) CONSTANTS & ANCHORS IN “CELL UNITS” ──
#     gsize = 416 // 32
#     cell_size = 416 / gsize
#     # convert ANCHORS_PX (pixel dims) → anchor sizes in grid‐cell units
#     anchors_cells = tf.reshape(
#         tf.constant([(aw/cell_size, ah/cell_size) for aw, ah in ANCHORS_PX],
#                     dtype=tf.float32),
#         [1, 1, NUM_ANCHORS, 2]
#     )

#     # ── 1) SPLIT GROUND‐TRUTH & PREDICTIONS ──
#     true_dxdy = y_true[..., 0:2]     # true offset within cell (0..1)
#     true_twth = y_true[..., 2:4]     # true log‐ratio to anchors
#     true_obj  = y_true[..., 4:5]     # 1 if object present in this anchor
#     true_cls  = y_true[..., 5:]      # one-hot class vector

#     raw_xy    = y_pred[..., 0:2]     # predicted tx, ty
#     raw_twh   = y_pred[..., 2:4]     # predicted tw, th
#     raw_conf  = y_pred[..., 4:5]     # predicted objectness logit
#     raw_class = y_pred[..., 5:]      # predicted class logits

#     obj_mask   = true_obj
#     noobj_mask = 1.0 - true_obj

#     # ── 2) BUILD CELL-GRID OFFSETS ──
#     cols, rows = tf.meshgrid(tf.range(gsize), tf.range(gsize), indexing="xy")
#     grid = tf.cast(tf.stack([cols, rows], axis=-1), tf.float32)
#     grid = grid[None, ..., None, :]   # shape (1, gsize, gsize, 1, 2)

#     # ── 3) DECODE PREDICTIONS ──
#     # 3.1) center x,y: sigmoid(tx,ty) + cell index → normalize by grid size
#     pred_xy = (tf.sigmoid(raw_xy) + grid) / tf.cast(gsize, tf.float32)

#     # 3.2) width, height: clip raw tw,th to avoid extreme exp(), then exp() → scale by anchors, normalize
#     raw_twh_clipped = tf.clip_by_value(raw_twh, -2.0, 2.0)
#     # pred_wh = (tf.exp(raw_twh_clipped) * anchors_cells) / tf.cast(gsize, tf.float32)
#     pred_wh = (raw_twh_clipped * anchors_cells) / tf.cast(gsize, tf.float32)

#     # ── 4) DECODE GROUND-TRUTH THE SAME WAY ──
#     true_xy = (true_dxdy + grid) / tf.cast(gsize, tf.float32)
#     # true_wh = (tf.exp(true_twth) * anchors_cells) / tf.cast(gsize, tf.float32)
#     true_wh = (true_twth * anchors_cells) / tf.cast(gsize, tf.float32)

#     # ── 5) COMPUTE IoU PER ANCHOR ──
#     def corners(xy, wh):
#         cx, cy = xy[..., 0], xy[..., 1]
#         w, h   = wh[..., 0], wh[..., 1]
#         return cx - w/2, cy - h/2, cx + w/2, cy + h/2

#     t_x1, t_y1, t_x2, t_y2 = corners(true_xy, true_wh)
#     p_x1, p_y1, p_x2, p_y2 = corners(pred_xy, pred_wh)

#     xi1 = tf.maximum(t_x1, p_x1)
#     yi1 = tf.maximum(t_y1, p_y1)
#     xi2 = tf.minimum(t_x2, p_x2)
#     yi2 = tf.minimum(t_y2, p_y2)
#     inter = tf.maximum(0.0, xi2 - xi1) * tf.maximum(0.0, yi2 - yi1)

#     area_t = (t_x2 - t_x1) * (t_y2 - t_y1)
#     area_p = (p_x2 - p_x1) * (p_y2 - p_y1)
#     union  = area_t + area_p - inter
#     iou    = inter / (union + 1e-10)

#     # ── 6) LOCALIZATION LOSS (IoU) ──
#     # only over positive anchors, scaled by λ_coord
#     loss_iou = tf.reduce_sum(obj_mask[..., 0] * (1.0 - iou))
#     loss_loc = lambda_coord * loss_iou

#     # ── 7) OBJECTNESS LOSS (sigmoid‐CE), NORMALISED ──
#     conf_ce = tf.nn.sigmoid_cross_entropy_with_logits(
#         labels=true_obj, logits=raw_conf
#     )
#     n_pos = tf.maximum(tf.reduce_sum(obj_mask), 1.0)
#     n_neg = tf.maximum(tf.reduce_sum(noobj_mask), 1.0)

#     loss_obj   = tf.reduce_sum(obj_mask   * conf_ce) / n_pos
#     loss_noobj = lambda_noobj * tf.reduce_sum(noobj_mask * conf_ce) / n_neg

#     # ── 8) CLASSIFICATION LOSS (softmax‐CE), NORMALISED ──
#     cls_ce = tf.nn.softmax_cross_entropy_with_logits(
#         labels=true_cls, logits=raw_class
#     )
#     loss_class = tf.reduce_sum(obj_mask[..., 0] * cls_ce) / n_pos

#     # ---- OPTIONAL DEBUG PRINTS ----
#     # (uncomment to inspect per-anchor values at training time)
#     # tf.print("raw_twh (model prediction):", raw_twh)
#     # tf.print("decoded pred_wh:", pred_wh)
#     # tf.print("decoded true_wh:", true_wh)
#     tf.print("IoU for positives:", tf.boolean_mask(iou, tf.cast(obj_mask[...,0],tf.bool)))
#     tf.print("loss_loc, loss_obj, loss_noobj, loss_class:",
#              loss_loc, loss_obj, loss_noobj, loss_class)

#     return loss_loc + loss_obj + loss_noobj + loss_class

# def yolo_loss(
#         y_true, y_pred,
#         lambda_coord=5.,
#         lambda_noobj=0.5,
#         beta_ratio=0.10):

#     """
#     --------  YOLO loss with direct (Δ / anchor) regression  --------

#     Fixed & instrumented version – changes are marked with  ### FIX  🔎
#     """

#     # ── 0) CONSTANTS & ANCHORS ────────────────────────────────────
#     gsize      = 416 // 32
#     cell_size  = 416 / gsize
#     anchors_px = tf.constant(ANCHORS_PX, tf.float32)
#     anchors_cells = tf.reshape(anchors_px / cell_size, [1, 1, NUM_ANCHORS, 2])

#     # ── 1) SPLIT  y_true / y_pred ────────────────────────────────
#     true_xy, true_ratio   = y_true[..., :2],  y_true[..., 2:4]
#     true_obj, true_cls    = y_true[..., 4:5], y_true[..., 5:]

#     raw_xy,  raw_ratio    = y_pred[..., :2],  y_pred[..., 2:4]
#     raw_conf, raw_class   = y_pred[..., 4:5], y_pred[..., 5:]

#     obj_mask   = true_obj                       # 1 for positives
#     noobj_mask = 1. - true_obj                  # 1 for negatives

#     # ── 2) CELL GRID ------------------------------------------------
#     cols, rows   = tf.meshgrid(tf.range(gsize), tf.range(gsize), indexing='xy')
#     grid         = tf.stack([cols, rows], -1)[None, ..., None, :]   # (1,g,g,1,2)
#     grid         = tf.cast(grid, tf.float32)

#     # ── 3) DECODE CENTER ------------------------------------------------
#     pred_xy = (tf.sigmoid(raw_xy) + grid) / gsize
#     true_xy = (true_xy          + grid) / gsize

#     # ── 4) DECODE  WH  --------------------------------------------------
#     raw_ratio_clipped = tf.clip_by_value(raw_ratio, -20.0, 20.0)
#     raw_ratio_soft = tf.nn.softplus(raw_ratio_clipped)                 # ≥0 with grad
#     pred_wh        = (raw_ratio_soft * anchors_cells) / gsize
#     true_wh        = (true_ratio      * anchors_cells) / gsize

#     # ── 5) IoU  ----------------------------------------------------------
#     def _corners(xy, wh):
#         cx, cy = xy[..., 0], xy[..., 1]
#         w,  h  = wh[..., 0], wh[..., 1]
#         return cx-w/2, cy-h/2, cx+w/2, cy+h/2

#     tx1, ty1, tx2, ty2 = _corners(true_xy, true_wh)
#     px1, py1, px2, py2 = _corners(pred_xy, pred_wh)

#     xi1 = tf.maximum(tx1, px1);  yi1 = tf.maximum(ty1, py1)
#     xi2 = tf.minimum(tx2, px2);  yi2 = tf.minimum(ty2, py2)

#     inter   = tf.maximum(0., xi2-xi1) * tf.maximum(0., yi2-yi1)
#     area_t  = (tx2-tx1) * (ty2-ty1)
#     area_p  = (px2-px1) * (py2-py1)
#     union   = area_t + area_p - inter
#     iou     = inter / (union + 1e-9)

#     # number of positive anchors in the **current** batch
#     n_pos = tf.maximum(tf.reduce_sum(obj_mask), 1.)            ### FIX

#     # ── 6) LOCALISATION LOSS  -------------------------------------------
#     loss_iou   = tf.reduce_sum(obj_mask[..., 0] * (1. - iou))   / n_pos
#     #ratio_mse  = tf.square(true_ratio - raw_ratio_soft)
#     ratio_mse  = tf.square(true_ratio - raw_ratio)
#     #loss_ratio = tf.reduce_sum(obj_mask[..., 0] * ratio_mse)    / n_pos
#     mask5 = obj_mask[..., 0][..., None]
#     loss_ratio = tf.reduce_sum(mask5 * ratio_mse) / n_pos
#     loss_loc   = lambda_coord * (loss_iou + beta_ratio*loss_ratio)

#     # ── 7) OBJECTNESS & CLASS  ------------------------------------------
#     conf_ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_obj,
#                                                       logits=raw_conf)
#     loss_obj   = tf.reduce_sum(obj_mask   * conf_ce) / n_pos
#     loss_noobj = lambda_noobj * tf.reduce_sum(noobj_mask * conf_ce) / (
#                    tf.maximum(tf.reduce_sum(noobj_mask), 1.))

#     cls_ce     = tf.nn.softmax_cross_entropy_with_logits(labels=true_cls,
#                                                          logits=raw_class)
#     loss_class = tf.reduce_sum(obj_mask[..., 0] * cls_ce) / n_pos

#     # ── 8) EXTRA DEBUG OUTPUT  ------------------------------------------
#     tf.print("\n=== YOLO-LOSS DEBUG ===  (batch size:", tf.shape(y_true)[0], ")")
#     tf.print("#pos anchors:", n_pos,
#              "  IoU (pos):", tf.boolean_mask(iou, tf.cast(obj_mask[...,0],tf.bool)))
#     tf.print("true_ratio (pos):", tf.boolean_mask(true_ratio, obj_mask[...,0]>0))
#     tf.print("pred_ratio (pos):", tf.boolean_mask(raw_ratio_soft, obj_mask[...,0]>0))
#     tf.print("raw_ratio   min/max:", tf.reduce_min(raw_ratio),
#                                    tf.reduce_max(raw_ratio))
#     tf.print("loss_iou   :", loss_iou,
#              "  loss_ratio:", loss_ratio,
#              "  loss_loc:", loss_loc)
#     tf.print("loss_obj :", loss_obj,
#              "  loss_noobj:", loss_noobj,
#              "  loss_class:", loss_class)
#     tf.print("total loss →", loss_loc + loss_obj + loss_noobj + loss_class, "\n")

#     # runtime numeric sanity – abort on NaNs / infs                ### FIX
#     tf.debugging.check_numerics(loss_loc,   "loss_loc")
#     tf.debugging.check_numerics(loss_class, "loss_class")

#     return loss_loc + loss_obj + loss_noobj + loss_class

# def yolo_loss(y_true, y_pred,
#               lambda_coord=5.0,
#               lambda_noobj=0.01):
#     # anchors_px: a (NUM_ANCHORS,2) float32 Tensor of your pixel‐anchors

#     # ── 0) constants ───────────────────────────────────────────────
#     gsize       = 416 // 32
#     cell_size   = 416 / gsize   # =32
#     # anchors in “cells” (so we can compare to true_wh * gsize)
#     anchors_cells = tf.reshape(anchors_px / cell_size,
#                                [1, 1, 1, -1, 2])  # (1,1,1,A,2)

#     # ── 1) split y_true / y_pred ─────────────────────────────────
#     # y_true[...,0:4] = [x1,y1,x2,y2], normalized to [0,1]
#     true_box, true_obj, true_cls = tf.split(
#         y_true,
#         (4, 1, tf.shape(y_true)[-1] - 5),
#         axis=-1
#     )
#     raw_xy    = y_pred[...,  0:2]    # network’s tx,ty logits
#     raw_wh    = y_pred[...,  2:4]    # network’s tw,th logits
#     raw_conf  = y_pred[...,  4:5]
#     raw_class = y_pred[...,  5:]

#     obj_mask  = tf.squeeze(true_obj, -1)           # (b,g,g,A)
#     n_pos     = tf.maximum(tf.reduce_sum(obj_mask), 1.0)

#     # ── 2) compute true center+size from corners ────────────────
#     # true centers & sizes in normalized [0,1]
#     true_xy_norm = (true_box[...,0:2] + true_box[...,2:4]) * 0.5
#     true_wh_norm =  true_box[...,2:4] - true_box[...,0:2]

#     # small-box scaling
#     box_loss_scale = 2.0 - true_wh_norm[...,0] * true_wh_norm[...,1]

#     # ── 3) build the cell‐grid & invert tx,ty decode ────────────
#     cols, rows = tf.meshgrid(tf.range(gsize), tf.range(gsize), indexing='xy')
#     grid = tf.cast(tf.stack([cols, rows], axis=-1), tf.float32)  # (g,g,2)
#     grid = tf.reshape(grid, (1, gsize, gsize, 1, 2))

#     # target offsets within each cell:
#     true_xy = true_xy_norm * gsize - grid    # (b,g,g,A,2)

#     # ── 4) invert the tw,th decode → log‐ratio targets ──────────
#     # true_wh_norm * gsize = size in “cells”
#     true_wh_cells = true_wh_norm * gsize
#     true_wh = tf.math.log(true_wh_cells / anchors_cells + 1e-9)   # (b,g,g,A,2)
#     true_wh = tf.where(tf.math.is_inf(true_wh),
#                        tf.zeros_like(true_wh),
#                        true_wh)

#     # ── 5) xy & wh MSE losses ────────────────────────────────────
#     # xy loss: compare raw_xy logits vs true_xy
#     xy_err = tf.square(true_xy - raw_xy)                           # (b,g,g,A,2)
#     xy_loss_cell = obj_mask[...,None] * box_loss_scale[...,None] * xy_err
#     xy_loss = tf.reduce_sum(xy_loss_cell) / n_pos

#     # wh loss: compare raw_wh logits vs true_wh
#     wh_err = tf.square(true_wh - raw_wh)                           # (b,g,g,A,2)
#     wh_loss_cell = obj_mask[...,None] * box_loss_scale[...,None] * wh_err
#     wh_loss = tf.reduce_sum(wh_loss_cell) / n_pos

#     loss_loc = lambda_coord * (xy_loss + wh_loss)

#     # ── 6) objectness loss ────────────────────────────────────────
#     conf_ce = tf.nn.sigmoid_cross_entropy_with_logits(
#                  labels=true_obj, logits=raw_conf
#              )       # (b,g,g,A,1)
#     n_neg = tf.maximum(tf.reduce_sum(1.0-obj_mask), 1.0)
#     loss_obj   = tf.reduce_sum(obj_mask[...,None]   * conf_ce) / n_pos
#     loss_noobj = tf.reduce_sum((1-obj_mask)[...,None]* conf_ce) / n_neg
#     loss_conf  = loss_obj + lambda_noobj * loss_noobj

#     # ── 7) classification loss ────────────────────────────────────
#     # true_cls is one‐hot over your NUM_CLASSES
#     cls_ce    = tf.nn.softmax_cross_entropy_with_logits(
#                     labels=true_cls, logits=raw_class
#                 )       # (b,g,g,A)
#     loss_cls  = tf.reduce_sum(obj_mask * cls_ce) / n_pos

#     return loss_loc # + loss_conf + loss_cls


# def yolo_loss(y_true,
#               y_pred,
#               step,
#               lambda_coord=2.0,
#               lambda_noobj=0.1,
#               ignore_thresh=0.5,
#               burnin_steps=800):      # how many steps to use pure MSE
#     # ── 0) dynamic grid & anchors ───────────────────────────
#     batch_size = tf.shape(y_pred)[0]
#     gsize      = tf.shape(y_pred)[1]
#     anchors_px = tf.constant(ANCHORS_PX, tf.float32)
#     anchors_cells = tf.reshape(anchors_px / 416.0, [1,1,1,-1,2])

#     # ── 1) split GT & pred ───────────────────────────────────
#     true_box, true_obj, true_cls = tf.split(
#         y_true, (4,1,tf.shape(y_true)[-1]-5), axis=-1
#     )
#     raw_xy   = y_pred[...,0:2]
#     raw_wh   = y_pred[...,2:4]
#     raw_conf = y_pred[...,4:5]
#     raw_cls  = y_pred[...,5:]

#     max_wh = 2.0
#     clamp_raw_wh = (2.0/math.pi) * max_wh * tf.atan(raw_wh)
#     reg_loss = 0.01 * tf.reduce_mean(tf.square(raw_wh)) # prevent large raw_wh
#     #clamp_raw_wh   = tf.clip_by_value(raw_wh, -2.0, 2.0)   # clamp to ±2

#     obj_mask = tf.squeeze(true_obj, -1)
#     n_pos    = tf.maximum(tf.reduce_sum(obj_mask), 1.0)
#     n_neg    = tf.maximum(tf.reduce_sum(1.0 - obj_mask), 1.0)

#     # ── 2) decode GT & preds ─────────────────────────────────
#     true_xy  = (true_box[...,0:2] + true_box[...,2:4]) * 0.5
#     true_wh  =  true_box[...,2:4] - true_box[...,0:2]

#     cols, rows = tf.meshgrid(tf.range(gsize), tf.range(gsize), indexing='xy')
#     grid = tf.cast(tf.stack([cols,rows],-1), tf.float32)
#     grid = tf.reshape(grid, (1,gsize,gsize,1,2))

#     pred_xy = (tf.sigmoid(raw_xy) + grid) / tf.cast(gsize, tf.float32)
#     pred_wh = (tf.exp(clamp_raw_wh) * anchors_cells) / tf.cast(gsize, tf.float32)

#     # ── 3) CIoU components ────────────────────────────────────
#     tx, ty = true_xy[...,0], true_xy[...,1]
#     tw, th = true_wh[...,0], true_wh[...,1]
#     px, py = pred_xy[...,0], pred_xy[...,1]
#     pw, ph = pred_wh[...,0], pred_wh[...,1]

#     t_x1, t_y1 = tx - tw/2, ty - th/2
#     t_x2, t_y2 = tx + tw/2, ty + th/2
#     p_x1, p_y1 = px - pw/2, py - ph/2
#     p_x2, p_y2 = px + pw/2, py + ph/2

#     inter_w = tf.maximum(0.0, tf.minimum(t_x2,p_x2)-tf.maximum(t_x1,p_x1))
#     inter_h = tf.maximum(0.0, tf.minimum(t_y2,p_y2)-tf.maximum(t_y1,p_y1))
#     inter_area = inter_w * inter_h

#     true_area = (t_x2-t_x1)*(t_y2-t_y1)
#     pred_area = (p_x2-p_x1)*(p_y2-p_y1)
#     union = true_area + pred_area - inter_area + 1e-9
#     iou   = inter_area / union

#     center_dist = tf.square(tx-px) + tf.square(ty-py)
#     c_x1 = tf.minimum(t_x1,p_x1); c_y1 = tf.minimum(t_y1,p_y1)
#     c_x2 = tf.maximum(t_x2,p_x2); c_y2 = tf.maximum(t_y2,p_y2)
#     c2 = tf.square(c_x2-c_x1) + tf.square(c_y2-c_y1) + 1e-9

#     diou = iou - center_dist / c2
#     v    = (4.0/(math.pi**2))*tf.square(
#                tf.atan(tw/(th+1e-9)) - tf.atan(pw/(ph+1e-9))
#            )
#     alpha = v / (1.0 - iou + v + 1e-9)
#     ciou  = diou - alpha * v

#     # ── 4) CIoU loss ──────────────────────────────────────────
#     ciou_loss = tf.reduce_sum(obj_mask * (1.0 - ciou)) / n_pos

#     # ── 5) MSE on box params ─────────────────────────────────
#     mse_xy = tf.reduce_sum(obj_mask[...,None] *
#                tf.square(pred_xy - true_xy)) / n_pos
#     # convert true_wh → "clamp_raw_wh target" for tw/th MSE
#     tw_th_target = tf.math.log(true_wh * tf.cast(gsize,tf.float32) / (anchors_cells + 1e-9) + 1e-9)
#     mse_wh = tf.reduce_sum(obj_mask[...,None] *
#                tf.square(clamp_raw_wh - tw_th_target)) / n_pos
#     mse_loss = mse_xy + mse_wh

#     # ── 6) mix per step ───────────────────────────────────────
#     α = tf.maximum(1.0 - tf.cast(step,tf.float32) / tf.cast(burnin_steps,tf.float32), 0.0)
#     loss_loc = lambda_coord * (α * mse_loss + (1.0 - α) * ciou_loss)

#     # ── 7) obj + noobj loss ───────────────────────────────────
#     conf_ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_obj, logits=raw_conf)
#     ignore_mask = tf.cast(iou < ignore_thresh, tf.float32)
#     loss_obj   = tf.reduce_sum(obj_mask[...,None]    * conf_ce) / n_pos
#     loss_noobj = tf.reduce_sum((1-obj_mask)[...,None] * ignore_mask[...,None] * conf_ce) / n_neg
#     loss_conf  = loss_obj + lambda_noobj * loss_noobj

#     # ── 8) cls loss ──────────────────────────────────────────
#     cls_ce = tf.nn.softmax_cross_entropy_with_logits(labels=true_cls, logits=raw_cls)
#     loss_cls = tf.reduce_sum(obj_mask * cls_ce) / n_pos

#        # ---- Debug Prints ----
#     tf.print("\n=== YOLO LOSS DEBUG ===")
#     # batch & anchor counts
#     tf.print(" batch size:", batch_size)
#     tf.print(" positives:", tf.reduce_sum(obj_mask), 
#              " negatives:", tf.reduce_sum(1.0 - obj_mask))

#     # get a boolean mask of positive anchors
#     pos_mask = obj_mask > 0                 # shape (b, g, g, A)
#     pos_idx  = tf.where(pos_mask)           # (b, y, x, anchor)
#     tf.print(" sample pos indices (b,y,x,anchor):", pos_idx, summarize=10)

#     # raw logits vs. decoded values at positives
#     raw_xy_pos   = tf.boolean_mask(raw_xy,   pos_mask)  # -> [num_pos, 2]
#     raw_wh_pos   = tf.boolean_mask(raw_wh,   pos_mask)  # -> [num_pos, 2]
#     pred_xy_pos  = tf.boolean_mask(pred_xy,  pos_mask)  # -> [num_pos, 2]
#     pred_wh_pos  = tf.boolean_mask(pred_wh,  pos_mask)  # -> [num_pos, 2]
#     tf.print(" raw_xy @ pos:", raw_xy_pos,   summarize=10)
#     tf.print(" raw_wh @ pos:", raw_wh_pos,   summarize=10)
#     tf.print(" pred_xy_norm @ pos:", pred_xy_pos, summarize=10)
#     tf.print(" pred_wh_norm @ pos:", pred_wh_pos, summarize=10)

#     # IoU / CIoU at positives
#     iou_pos  = tf.boolean_mask(iou,  pos_mask)
#     ciou_pos = tf.boolean_mask(ciou, pos_mask)
#     tf.print(" IoU @ pos:",  iou_pos,  summarize=10)
#     tf.print(" CIoU @ pos:", ciou_pos, summarize=10)

#     # loss components
#     tf.print(" loss_loc   (λcoord*CIoU):", loss_loc)
#     tf.print(" loss_obj   (obj conf):",   loss_obj)
#     tf.print(" loss_noobj (neg conf):",   loss_noobj)
#     tf.print(" loss_conf  (total conf):", loss_conf)
#     tf.print(" loss_cls   (class CE):",   loss_cls)

#     # hyperparameters
#     tf.print(" λcoord:", lambda_coord,
#              " λnoobj:", lambda_noobj,
#              " ignore_thresh:", ignore_thresh)

#     # total
#     total_loss = loss_loc + loss_conf + loss_cls
#     tf.print(" TOTAL_LOSS:", total_loss)
#     tf.print("=== END YOLO LOSS DEBUG ===\n")

#     return loss_loc + loss_conf + loss_cls + reg_loss

def yolo_loss(y_true, y_pred,
              lambda_coord=5.0, lambda_noobj=0.5):
    # 0) shapes & constants --------------------------------------------------
    gsize      = tf.shape(y_pred)[1]                   # e.g. 13
    anchors_px = tf.constant(ANCHORS_PX, tf.float32)
    anchors_c  = tf.reshape(anchors_px / 416.0, [1,1,1,-1,2])

    # 1) split tensors -------------------------------------------------------
    true_off, true_twth, true_obj, true_cls = tf.split(
        y_true,
        [2, 2, 1, tf.shape(y_true)[-1] - 5],
        axis=-1
    )
    raw_xy, raw_twth, raw_conf, raw_cls = tf.split(
        y_pred,
        [2, 2, 1, tf.shape(y_pred)[-1] - 5],
        axis=-1
    )

    obj_mask = tf.squeeze(true_obj, -1)    # 1 where there is an object
    n_pos    = tf.maximum(tf.reduce_sum(obj_mask), 1.0)

    # 2) build grid ----------------------------------------------------------
    cols, rows = tf.meshgrid(tf.range(gsize), tf.range(gsize), indexing='xy')
    grid = tf.cast(tf.stack([cols, rows], -1), tf.float32)  # (g,g,2)
    grid = grid[None, ..., None, :]                         # (1,g,g,1,2)

    # 3) decode --------------------------------------------------------------
    pred_xy = (tf.sigmoid(raw_xy) + grid) / tf.cast(gsize, tf.float32)
    pred_wh = tf.exp(raw_twth) * anchors_c / tf.cast(gsize, tf.float32)

    true_xy = (true_off + grid) / tf.cast(gsize, tf.float32)
    true_wh = tf.exp(true_twth) * anchors_c / tf.cast(gsize, tf.float32)

    # 4) IoU‐based localization loss ----------------------------------------
    def _corners(xy, wh):
        cx, cy = xy[...,0], xy[...,1]
        w,  h  = wh[...,0], wh[...,1]
        return cx - w/2, cy - h/2, cx + w/2, cy + h/2

    tx1, ty1, tx2, ty2 = _corners(true_xy,  true_wh)
    px1, py1, px2, py2 = _corners(pred_xy,  pred_wh)

    inter = tf.maximum(0., tf.minimum(tx2, px2) - tf.maximum(tx1, px1)) * \
            tf.maximum(0., tf.minimum(ty2, py2) - tf.maximum(ty1, py1))
    union = (tx2-tx1)*(ty2-ty1) + (px2-px1)*(py2-py1) - inter + 1e-9
    iou   = inter / union

    loc_loss = tf.reduce_sum(obj_mask * (1. - iou)) / n_pos

    # 5) xy/wh‐MSE for faster convergence -------------------------------
    mse_off = tf.reduce_sum(
        obj_mask[...,None] * tf.square(tf.sigmoid(raw_xy) - true_off)
    ) / n_pos
    mse_wh  = tf.reduce_sum(
        obj_mask[...,None] * tf.square(raw_twth - true_twth)
    ) / n_pos

    loc_loss = lambda_coord * (loc_loss + mse_off + mse_wh)

    # 6) confidence & classification ----------------------------------------
    conf_ce = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=true_obj, logits=raw_conf
    )

    # ---- positive (object) loss
    loss_obj = tf.reduce_sum(obj_mask[...,None] * conf_ce) / n_pos

    # ---- negative (no‐object) loss: *also* divide by n_pos!
    #     this makes each false‐positive cost as much as a false‐negative
    loss_noobj = tf.reduce_sum((1 - obj_mask)[...,None] * conf_ce) / n_pos

    cls_ce   = tf.nn.softmax_cross_entropy_with_logits(
        labels=true_cls, logits=raw_cls
    )
    loss_cls = tf.reduce_sum(obj_mask * cls_ce) / n_pos

    return loc_loss + loss_obj + lambda_noobj * loss_noobj + loss_cls


def yolo_loss_debug(y_true, y_pred):
    """
    Debugging version of YOLOv2 loss.
    Expects y_true and y_pred with shape [batch, grid_h, grid_w, anchors, ..] where 
    y_true contains encoded coordinates and object mask, and y_pred contains raw outputs.
    """
    # Assuming y_pred has structure: [..., 0:2] = raw tx, ty; [..., 2:4] = raw tw, th; 
    # [..., 4] = raw objectness score (logit); [..., 5:] = class predictions.
    raw_xy = y_pred[..., 0:2]        # raw x, y predictions (before sigmoid)
    raw_tw_th = y_pred[..., 2:4]     # raw w, h (log-space predictions)
    raw_conf = y_pred[..., 4:5]      # raw objectness score
    raw_class = y_pred[..., 5:]      # class scores (if any)

    # Separate ground truth components (assuming y_true is formatted similarly to y_pred in encoding)
    true_xy = y_true[..., 0:2]       # ground truth x, y (relative offsets in cell)
    true_tw_th = y_true[..., 2:4]    # ground truth t_w, t_h (log-space targets)
    object_mask = y_true[..., 4:5]   # object mask (1 if object present for this anchor, else 0)
    true_class = y_true[..., 5:]     # ground truth class one-hot vector (if applicable)

    # Constants (you should replace these with actual values from your model configuration)
    lambda_coord = 5.0       # example coordinate weight (adjust to your YOLO config)
    lambda_noobj = 0.5       # example no-object weight for confidence
    grid_size = tf.shape(y_pred)[1]  # grid dimension (assuming square grid)
    grid_size = tf.cast(grid_size, tf.float32)
    # Anchors (replace with your actual anchor tensor of shape [N_anchors, 2])
    anchors = tf.constant(ANCHORS_PX, dtype=tf.float32)  
    # Expand anchors to [1,1,1,N,2] for broadcasting
    anchors = tf.reshape(anchors, [1, 1, 1, -1, 2])
    
    # Decode predicted x,y to [0,1] coordinates (relative to image)
    # Create a mesh grid of cell indices (shape [grid_h, grid_w, 1, 2]):
    grid_y = tf.range(tf.shape(y_pred)[1], dtype=tf.float32)
    grid_x = tf.range(tf.shape(y_pred)[2], dtype=tf.float32)
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)                    # shape [grid_h, grid_w]
    grid_stack = tf.stack([grid_x, grid_y], axis=-1)                # shape [grid_h, grid_w, 2]
    grid_stack = tf.expand_dims(grid_stack, axis=2)                 # shape [grid_h, grid_w, 1, 2]
    grid_stack = tf.tile(grid_stack, [1, 1, tf.shape(y_pred)[3], 1])# tile to [grid_h, grid_w, anchors, 2]

    # Apply sigmoid to raw x,y and add cell index, then normalize by grid size
    pred_xy = (tf.sigmoid(raw_xy) + grid_stack) / grid_size  # shape [batch, gh, gw, anchors, 2]
    # Clamp raw w,h for safety (to avoid inf exp) – only clamp to upper bound to keep gradient for negative values
    clamped_raw_tw_th = tf.clip_by_value(raw_tw_th, -2.0, 2.0)
    # Decode w,h: apply exp to raw prediction and multiply by anchor, then normalize by image size
    pred_wh = tf.exp(clamped_raw_tw_th) * anchors / grid_size      # shape [batch, gh, gw, anchors, 2]

    # Compute corner coordinates for predicted and true boxes (normalized [0,1] coordinates)
    pred_x1y1 = pred_xy - pred_wh / 2.0
    pred_x2y2 = pred_xy + pred_wh / 2.0
    # Decode ground-truth w,h from t_w,t_h (assuming anchors and grid scaling same as pred decode)
    true_wh = tf.exp(true_tw_th) * anchors / grid_size
    # Compute ground truth box corners
    true_xy_abs = (true_xy + grid_stack) / grid_size  # absolute center (add cell offset and normalize)
    true_x1y1 = true_xy_abs - true_wh / 2.0
    true_x2y2 = true_xy_abs + true_wh / 2.0

    # IoU calculation between pred box and true box for each anchor
    intersect_x1 = tf.maximum(pred_x1y1[..., 0:1], true_x1y1[..., 0:1])
    intersect_y1 = tf.maximum(pred_x1y1[..., 1:2], true_x1y1[..., 1:2])
    intersect_x2 = tf.minimum(pred_x2y2[..., 0:1], true_x2y2[..., 0:1])
    intersect_y2 = tf.minimum(pred_x2y2[..., 1:2], true_x2y2[..., 1:2])
    intersect_w = tf.maximum(intersect_x2 - intersect_x1, 0.0)
    intersect_h = tf.maximum(intersect_y2 - intersect_y1, 0.0)
    intersect_area = intersect_w * intersect_h
    pred_area = (pred_x2y2[..., 0:1] - pred_x1y1[..., 0:1]) * (pred_x2y2[..., 1:2] - pred_x1y1[..., 1:2])
    true_area = (true_x2y2[..., 0:1] - true_x1y1[..., 0:1]) * (true_x2y2[..., 1:2] - true_x1y1[..., 1:2])
    iou = intersect_area / (pred_area + true_area - intersect_area + 1e-10)  # IoU per anchor

    # give higher weights to small boxes
    box_loss_scale = 2.0 - true_wh[..., 0] * true_wh[..., 1]

    # instantiate a Huber loss that returns per‐element values
    # (no reduction, so shape stays [..., 2])
    huber_fn = tf.keras.losses.Huber(
        delta=1.0,
        reduction=tf.keras.losses.Reduction.NONE
    )

    # compute Huber loss between the two decoded wh tensors
    # this yields shape [batch, gh, gw, anchors, 2]
    raw_wh_diff = huber_fn(true_wh, pred_wh)

    # sum over the width/height dimensions (last axis)
    wh_loss = tf.reduce_sum(raw_wh_diff, axis=-1)

    # now mask out non‐object cells and scale by box size
    wh_loss = object_mask[..., 0] * box_loss_scale * wh_loss

    # Objectness (confidence) loss:
    conf_target = object_mask * 1.0  # using 1 for object presence (could also use iou if that’s the intended target)
    conf_noobj_mask = 1.0 - object_mask
    conf_loss = object_mask * tf.square(1.0 - tf.sigmoid(raw_conf)) \
              + conf_noobj_mask * lambda_noobj * tf.square(0.0 - tf.sigmoid(raw_conf))
    # (If using IoU as target, replace the above two lines with:
    # conf_target = object_mask * iou
    # conf_loss = tf.square(conf_target - tf.sigmoid(raw_conf)) * (object_mask + lambda_noobj * (1-object_mask))
    # )
    # Class loss (if applicable): only apply when object is present
    class_loss = object_mask * tf.square(true_class - tf.sigmoid(raw_class))  # (or use softmax and cross-entropy in real implementation)

    # Sum up losses (for reporting purposes)
    xy_loss_sum = tf.reduce_sum(xy_loss)
    wh_loss_sum = tf.reduce_sum(wh_loss)
    conf_loss_sum = tf.reduce_sum(conf_loss)
    class_loss_sum = tf.reduce_sum(class_loss)
    total_loss = lambda_coord * (xy_loss_sum + wh_loss_sum) + conf_loss_sum + class_loss_sum

    # ---- Debug Prints ----
    object_mask_squeezed = tf.squeeze(object_mask, axis=-1)
    tf.print("\n=== Debug Info ===")
    tf.print("Object mask shape:", tf.shape(object_mask), " Num. of objects in batch:", tf.reduce_sum(object_mask, keepdims=False))
    tf.print("Indices of object anchors (y, x, anchor):", tf.where(object_mask_squeezed), summarize=1000)
    tf.print("raw t_w/t_h (predicted) for object anchors:", tf.boolean_mask(raw_tw_th, object_mask_squeezed), summarize=1000)
    tf.print("t_w/t_h targets (ground truth) for objects:", tf.boolean_mask(true_tw_th, object_mask_squeezed), summarize=1000)
    tf.print("Decoded pred w/h (normalized) for objects:", tf.boolean_mask(pred_wh, object_mask_squeezed), summarize=1000)
    tf.print("Ground-truth w/h (normalized) for objects:", tf.boolean_mask(true_wh, object_mask_squeezed), summarize=1000)
    tf.print("IoU of pred vs true for object anchors:", tf.boolean_mask(iou, object_mask_squeezed), summarize=1000)
    tf.print("xy_loss:", xy_loss_sum, " wh_loss:", wh_loss_sum, " conf_loss:", conf_loss_sum, " class_loss:", class_loss_sum)
    tf.print("Total loss:", total_loss)
    tf.print("=== End Debug Info ===\n")

    return total_loss


if __name__ == '__main__':
    model = Darknet19(input_shape=(416, 416, 3), num_classes=20, B=1)
    model.compile(optimizer='adam', loss=yolo_loss)
    model.summary()
