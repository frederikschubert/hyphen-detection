import base64
import csv
import io
import json
import logging
import multiprocessing as mp
import operator
import os
import random
import shutil
from copy import deepcopy
from itertools import product
from pathlib import Path, PosixPath
from typing import List, Tuple
from xml.dom import minidom

import cv2
import numpy as np
import pandas as pd
import pyvips
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

log = logging.getLogger(__name__)


def imap_progress(f, iterable: List, flatten=False):
    with mp.Pool(processes=os.cpu_count()) as pool:
        results = []
        for result in tqdm(pool.imap(f, iterable), total=len(iterable)):
            if result:
                if flatten:
                    results.extend(result)
                else:
                    results.append(result)
    return results


def get_centers(grid_path: str):
    coordinates = pd.read_csv(grid_path)
    x, y = coordinates["x"], coordinates["y"]
    centers = list(product(x, y))
    centers = [
        (int(circle[0]), int(circle[1]))
        for circle in centers
        if not np.isnan(circle[0]) and not np.isnan(circle[1])
    ]
    return centers


def validate(points, centers):
    return (
        len(points) == len(centers)
        and len(np.unique(points[:, 0])) == len(np.unique(centers[:, 0]))
        and len(np.unique(points[:, 1])) == len(np.unique(centers[:, 1]))
    )


def try_rounding(points):
    points = deepcopy(points)
    return [[round(point[0]), round(point[1]), point[2]] for point in points]


def try_matching(points):
    points = deepcopy(points)
    for point in points:
        for coord in [0, 1]:
            closest_match = np.inf
            for p in points:
                if p == point:
                    continue
                if (
                    np.abs(point[coord] - closest_match)
                    > np.abs(point[coord] - p[coord])
                    and p[coord] % 1 == 0
                ):
                    closest_match = p[coord]
            if closest_match == np.inf:
                closest_match = round(point[coord])
            point[coord] = closest_match
    return points


def process_points(points):
    points = sorted(points, key=operator.itemgetter(0, 1))
    points = points[1:]  # first point is reference point
    points = np.array(points)
    return points


def get_labels(centers, points_el):
    points = [
        [
            float(point_el.getAttribute("cx")),
            float(point_el.getAttribute("cy")),
            0 if "fill:none" in point_el.getAttribute("style") else 1,
        ]
        for point_el in points_el
    ]

    centers = np.array(centers)

    points_rounded = try_rounding(points)
    points_rounded = process_points(points_rounded)
    if validate(points_rounded, centers):
        labels = [int(point[2]) for point in points_rounded]
        return labels
    points_matched = try_matching(points)
    points_matched = process_points(points_matched)
    if validate(points_matched, centers):
        labels = [int(point[2]) for point in points_matched]
        return labels
    points_rounded_matched = try_matching(try_rounding(points))
    points_rounded_matched = process_points(points_rounded_matched)
    if validate(points_rounded_matched, centers):
        labels = [int(point[2]) for point in points_rounded_matched]
        return labels
    return None


def get_image(image_el, size):
    href = image_el.getAttribute("xlink:href")
    msg = base64.b64decode(href.split("base64,")[1])
    buf = io.BytesIO(msg)
    img = Image.open(buf).convert("RGB").resize(size)
    return img


def extract_points_and_images(data: Tuple[PosixPath, str, Tuple[int, int]]):
    file, grid, size = data
    # logger.info("Processing file {}", str(file))
    if "Asendorf" in str(file):
        doc = minidom.parse(str(file))
        points_el = doc.getElementsByTagName("circle") + doc.getElementsByTagName(
            "ellipse"
        )
        image_el = doc.getElementsByTagName("image")[0]
        centers = get_centers(grid)
        labels = get_labels(centers, points_el)
        if labels is None:
            log.warning("Could not extract labels for file {}", file)
            return None
        image = get_image(image_el, size)
        image = np.array(image)
    else:
        doc_image = minidom.parse(str(file))
        doc_points = minidom.parse(str(file))

        images_el = doc_points.getElementsByTagName("image")
        for image_el in images_el:
            image_el.parentNode.removeChild(image_el)
            image_x, image_y = float(image_el.getAttribute("x")), float(
                image_el.getAttribute("y")
            )
        points_el = doc_points.getElementsByTagName(
            "circle"
        ) + doc_image.getElementsByTagName("ellipse")

        x, y = [], []
        for point_el in points_el:
            point_el.setAttribute(
                "cx", str(float(point_el.getAttribute("cx")) - image_x)
            )
            point_el.setAttribute(
                "cy", str(float(point_el.getAttribute("cy")) - image_y)
            )
            x.append(float(point_el.getAttribute("cx")))
            y.append(float(point_el.getAttribute("cy")))
        x, y = np.unique(x), np.unique(y)
        points_svg = doc_points.toxml()

        points_el = doc_image.getElementsByTagName(
            "circle"
        ) + doc_image.getElementsByTagName("ellipse")
        for point_el in points_el:
            point_el.parentNode.removeChild(point_el)
        images_el = doc_image.getElementsByTagName("image")
        for image_el in images_el:
            image_el.setAttribute("x", str(0))
            image_el.setAttribute("y", str(0))

        image_svg = doc_image.toxml()

        with open("./data/image.svg", "w") as f:
            f.write(image_svg)

        with open("./data/points.svg", "w") as f:
            f.write(points_svg)

        image = pyvips.Image.new_from_buffer(image_svg.encode("utf-8"), "", dpi=96)
        points = pyvips.Image.new_from_buffer(points_svg.encode("utf-8"), "", dpi=96)
        bbox = image.find_trim()
        if len(bbox) != 4 or bbox[-1] == 0 or bbox[-2] == 0:
            log.warning("File {} did not contain correct data", str(file))
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
        points = cv2.cvtColor(points, cv2.COLOR_RGBA2RGB)

        coordinates = pd.read_csv(grid)
        x, y = coordinates["x"], coordinates["y"]
        circles = list(product(y, x))
        circles = [
            (int(circle[0]), int(circle[1]))
            for circle in circles
            if not np.isnan(circle[0]) and not np.isnan(circle[1])
        ]
        centers = []
        labels = []
        if circles:
            for y, x in circles:
                is_positive = (
                    (points[y : y + 9, x : x + 9] == np.array([255, 0, 0]))
                    .all(axis=-1)
                    .any()
                )
                labels.append(1 if is_positive else 0)
                centers.append((x, y))
        else:
            log.warning("Did not find circle grid")

    return (
        image,
        centers,
        labels,
    )


def prepare_dataset(cfg: DictConfig):
    svg_files = []
    grid_files = []
    sizes = []
    for directory, grid in zip(cfg.dataset.input_dirs, cfg.dataset.grid_files):
        files = list(Path(directory).glob("**/*aster/*.svg"))
        svg_files.extend(files)
        grid_files.extend([grid] * len(files))
        if "Asendorf" in directory:
            size = (1024, 768)
        else:
            size = (780, 660)
        sizes.extend([size] * len(files))
    log.info("Found {} svg files", len(svg_files))
    images, center_lists, label_lists = zip(
        *imap_progress(
            extract_points_and_images, list(zip(svg_files, grid_files, sizes))
        )
    )
    log.info("Extracted {} images", len(images))
    if os.path.exists(cfg.dataset.output_dir):
        shutil.rmtree(cfg.dataset.output_dir)
    train_output_dir = os.path.join(cfg.dataset.output_dir, "train")
    os.makedirs(train_output_dir)
    val_output_dir = os.path.join(cfg.dataset.output_dir, "val")
    os.makedirs(val_output_dir, exist_ok=True)
    log.info("Writing images, centers and labels...")
    fieldnames = [
        "image_path",
        "label",
        "x",
        "y",
    ]
    removed_from_val_percentage = len(
        [input_path for input_path in svg_files if "Asendorf" in str(input_path)]
    ) / len(svg_files)
    mean = np.array([0.0, 0.0, 0.0])
    std = np.array([0.0, 0.0, 0.0])
    count = 0
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
            for input_path, img, centers, labels in tqdm(
                zip(svg_files, images, center_lists, label_lists),
                total=len(images),
            ):
                is_val = (
                    "Asendorf" not in str(input_path)
                    and random.random()
                    < cfg.dataset.val_split
                    + removed_from_val_percentage * cfg.dataset.val_split
                )
                if not is_val:
                    img_np = np.array(img).reshape([-1, img.shape[-1]])
                    mean += np.mean(img_np, axis=0)
                    std += np.std(img_np, axis=0)
                    count += 1
                output_dir = val_output_dir if is_val else train_output_dir
                writer = val_writer if is_val else train_writer
                output_path_image = os.path.join(
                    output_dir, os.path.relpath(input_path, cfg.dataset.input_dir_base)
                )
                output_path_image = output_path_image.replace(".svg", ".png")
                os.makedirs(os.path.dirname(output_path_image), exist_ok=True)
                log.debug("Writing image to {}", output_path_image)
                plt.imsave(output_path_image, img)
                for center, label in zip(centers, labels):
                    writer.writerow(
                        {
                            "image_path": output_path_image,
                            "label": label,
                            "x": center[0],
                            "y": center[1],
                        }
                    )
    mean = mean / count
    std = std / count
    with open(os.path.join(cfg.dataset.output_dir, "metadata.json"), "w") as f:
        json.dump({"statistics": {"mean": mean.tolist(), "std": std.tolist()}}, f)
