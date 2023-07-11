from typing import Dict

import tensorflow as tf

from octa_infer.unet import UNet


def model_fn(features, labels, mode: tf.estimator.ModeKeys, params: Dict[str, object]):
    unet = UNet()
    output = unet(features)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'output': output,
            'image': features['image']}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    if mode == tf.estimator.ModeKeys.EVAL:
        raise NotImplemented

    if mode == tf.estimator.ModeKeys.TRAIN:
        raise NotImplemented
