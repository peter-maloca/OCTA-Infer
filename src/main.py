import logging
import os.path
from pathlib import Path

from octa_infer.predictor import Predictor

logging.basicConfig(format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s', level=logging.INFO)
PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent


# Run an example OCT scan through the OCTA-Infer net
if __name__ == '__main__':
    dir_images = PROJECT_ROOT / "data/images/oct"
    dir_predictions = PROJECT_ROOT / "data/images/octa_predicted"
    dir_model = PROJECT_ROOT / "data/checkpoints"
    Predictor(dir_images_=dir_images, dir_predictions_=dir_predictions, dir_model_=dir_model).run()
