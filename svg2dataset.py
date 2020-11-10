import os
import csv
import shutil
import random
from pathlib import Path, PosixPath
from xml.dom import minidom

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyvips
from common.utils import imap_progress
from loguru import logger
from tap import Tap
from tqdm import tqdm


def find_circles(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

    lower_bound = np.array([10])
    upper_bound = np.array([255])

    mask = cv2.inRange(image, lower_bound, upper_bound)

    contours = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]

    centers = []
    for c in contours:
        (x, y), r = cv2.minEnclosingCircle(c)
        center = (int(y), int(x))
        centers.append(center)
    return sorted(
        sorted(centers, key=lambda center: center[0]), key=lambda center: center[1]
    )


class Arguments(Tap):
    input_dir: str = "data/counts"
    output_dir: str = "nobackup/dataset"
    debug: bool = False
    num_samples: int = 2
    val_split: float = 0.2


def extract_points_and_images(file: PosixPath):
    logger.trace("Processing file {}", str(file))
    doc_image = minidom.parse(str(file))
    doc_points = minidom.parse(str(file))

    images_el = doc_points.getElementsByTagName("image")
    for image_el in images_el:
        image_el.parentNode.removeChild(image_el)
    points_svg = doc_points.toxml()

    points_el = doc_image.getElementsByTagName(
        "circle"
    ) + doc_image.getElementsByTagName("ellipse")
    for point_el in points_el:
        point_el.parentNode.removeChild(point_el)

    image_svg = doc_image.toxml()

    image = pyvips.Image.new_from_buffer(image_svg.encode("utf-8"), "")
    points = pyvips.Image.new_from_buffer(points_svg.encode("utf-8"), "")
    bbox = image.find_trim()
    if len(bbox) != 4 or bbox[-1] == 0 or bbox[-2] == 0:
        logger.warning("File {} did not contain correct data", str(file))
        return None
    image = image.crop(*bbox)
    points = points.crop(*bbox)
    image = np.ndarray(
        buffer=image.write_to_memory(),
        shape=[image.height, image.width, image.bands],
        dtype=np.uint8,
    )
    points = np.ndarray(
        buffer=points.write_to_memory(),
        shape=[points.height, points.width, points.bands],
        dtype=np.uint8,
    )
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    circles = find_circles(points)
    raster_gap_y = int(np.max(np.diff(np.sort([c[0] for c in circles]))))
    raster_gap_x = int(np.max(np.diff(np.sort([c[1] for c in circles]))))
    logger.trace("Found raster with gap {}/{}", raster_gap_y, raster_gap_x)

    bboxes = []
    labels = []
    if circles:
        logger.trace("Found {} circles", len(circles))
        for y, x in circles:
            is_positive = points[y, x, -1] != 0
            labels.append(1 if is_positive else 0)
            min_x = round(max(0, x - raster_gap_x / 2))
            max_x = round(min(image.shape[1], x + raster_gap_x / 2))
            min_y = round(max(0, y - raster_gap_y / 2))
            max_y = round(min(image.shape[0], y + raster_gap_y / 2))
            bboxes.append([min_x, max_x, min_y, max_y])
    else:
        logger.warning("Did not find circle grid")
    return (
        image,
        bboxes,
        labels,
        points,
    )


def main():
    args = Arguments().parse_args()
    svg_files = list(Path(args.input_dir).glob("**/Bilder_mitRaster/*.svg"))
    logger.info("Found {} svg files", len(svg_files))
    if args.debug:
        logger.warning("Debug mode, using only {} files", args.num_samples)
        svg_files = svg_files[: args.num_samples]
    images, bbox_lists, label_lists, points = zip(
        *imap_progress(extract_points_and_images, svg_files)
    )
    logger.info("Extracted {} images", len(images))
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    train_output_dir = os.path.join(args.output_dir, "train")
    os.makedirs(train_output_dir)
    val_output_dir = os.path.join(args.output_dir, "val")
    os.makedirs(val_output_dir, exist_ok=True)
    logger.info("Writing cropped images, bboxes and labels...")
    fieldnames = [
        "image_path",
        "label",
        "bbox_min_x",
        "bbox_max_x",
        "bbox_min_y",
        "bbox_max_y",
    ]
    with open(
        os.path.join(train_output_dir, "annotations.csv"), "w", newline=""
    ) as train_csv:
        train_writer = csv.DictWriter(train_csv, fieldnames=fieldnames)
        train_writer.writeheader()
        with open(
            os.path.join(val_output_dir, "annotations.csv"), "w", newline=""
        ) as val_csv:
            val_writer = csv.DictWriter(val_csv, fieldnames=fieldnames)
            val_writer.writeheader()
            for input_path, img, bboxes, labels, p in tqdm(
                zip(svg_files, images, bbox_lists, label_lists, points),
                total=len(images),
            ):
                is_val = random.random() < args.val_split
                output_dir = val_output_dir if is_val else train_output_dir
                writer = val_writer if is_val else train_writer

                output_path_image = str(input_path).replace(args.input_dir, output_dir)
                output_path_image = output_path_image.replace(".svg", ".png")
                os.makedirs(os.path.dirname(output_path_image), exist_ok=True)
                logger.trace("Writing image to {}", output_path_image)
                plt.imsave(output_path_image, img)

                for bbox, label in zip(bboxes, labels):
                    writer.writerow(
                        {
                            "image_path": output_path_image,
                            "label": label,
                            "bbox_min_x": bbox[0],
                            "bbox_max_x": bbox[1],
                            "bbox_min_y": bbox[2],
                            "bbox_max_y": bbox[3],
                        }
                    )
                if args.debug:
                    output_path_points = output_path_image.replace(
                        ".png", "_points.png"
                    )
                    logger.trace("Writing points to {}", output_path_points)
                    plt.imsave(output_path_points, p)
                    combination_mask = np.expand_dims(p[:, :, -1] != 0, -1)
                    output_path_combination = output_path_points.replace(
                        "_points.png", "combination.png"
                    )
                    plt.imsave(
                        output_path_combination, np.where(combination_mask, p, img)
                    )


if __name__ == "__main__":
    main()
