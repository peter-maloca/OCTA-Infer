import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf

from octa_infer.model_fn import model_fn
from octa_infer.numpy_input_fn import numpy_input_fn

_logger = logging.getLogger(__name__)


class Predictor:
    _dir_images: Path
    _dir_predictions: Path
    _dir_model: Path

    def __init__(self, dir_images_: Path, dir_predictions_: Path, dir_model_: Path):
        if not os.path.exists(dir_predictions_):
            os.makedirs(dir_predictions_)
        self._dir_images = dir_images_
        self._dir_predictions = dir_predictions_
        self._dir_model = dir_model_

    def run(self) -> None:
        image_paths = _find_image_paths(self._dir_images)
        for image_path in image_paths:
            self._run_image(image_path)

    def _run_image(self, image_path: Path):
        image = Image.open(image_path).convert('L')
        image_size = image.size
        image.resize((512, 512), Image.ANTIALIAS)
        prediction = _predict(np.array(image), checkpoints_dir=self._dir_model)
        self._save_prediction(labels_map=prediction, image_size=image_size, name_img=image_path.name)

    def _save_prediction(self, labels_map: np.array, image_size: Tuple[int, int], name_img: str):
        img_clipped = np.clip(labels_map, a_min=0, a_max=1)
        img_normalized = 255 * img_clipped
        labels_map_rgba = Image.fromarray(img_normalized).convert('RGB').resize(size=image_size)
        labels_map_rgba.save(self._dir_predictions / name_img)


def _find_image_paths(dir_: Path) -> List[Path]:
    images = sorted([f for f in os.listdir(dir_) if f.upper().endswith(".PNG") or f.upper().endswith(".BMP")])
    return [Path(os.path.join(dir_, i)) for i in images]


def _predict(image: np.ndarray, checkpoints_dir: Path) -> np.ndarray:
    _logger.info('Executing inference on image ...')
    image.resize((1, 512, 512, 1))
    labels_predicted = _predict_numpy(image, checkpoints_dir)
    labels_predicted.resize((512, 512))
    return labels_predicted


def _predict_numpy(image: np.ndarray, checkpoints_dir: Path) -> np.array:
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=str(checkpoints_dir))

    predictions = estimator.predict(
        input_fn=lambda: numpy_input_fn(image=image))

    predicted_outputs = []
    for prediction in predictions:
        outputs = np.array(prediction['output'])
        predicted_outputs.append(outputs)

    return np.reshape(predicted_outputs, newshape=(image.shape[0], 512, 512))
