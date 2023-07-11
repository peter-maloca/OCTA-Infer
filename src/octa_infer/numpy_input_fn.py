import numpy as np

import tensorflow as tf


def numpy_input_fn(image: np.ndarray):
    image = image / 256.
    dataset = tf.data.Dataset.from_tensor_slices({'image': image.reshape([-1, 512, 512, 1]).astype(np.float32)})
    dataset = dataset.batch(1)
    return dataset.make_one_shot_iterator().get_next()