from PIL import Image
from train import HyphenDetection


def create_segmentation(model: HyphenDetection, image: Image, granularity: int):
    # TODO(frederik): for each point in the image create a mask
    # TODO(frederik): classify image with each mask
    # TODO(frederik): overlay masks weighted with predictions
    # TODO(frederik): perform argmax and return segmentation mask
    pass