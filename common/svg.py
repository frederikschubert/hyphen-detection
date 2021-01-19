from typing import List, Tuple
import base64

from wand.image import Image
from svgwrite import Drawing
from svgwrite.image import Image as SVGImage
from svgwrite.shapes import Circle
import numpy as np


def create_svg(
    filename: str, image_path: str, centers: List[Tuple[int, int]], labels: List[int]
) -> str:
    img = Image(filename=image_path)
    image_data = img.make_blob(format="png")
    encoded = base64.b64encode(image_data).decode()
    pngdata = f"data:image/png;base64,{encoded}"
    dwg = Drawing(filename, profile="tiny")
    dwg.add(SVGImage(href=pngdata))
    for center, label in zip(centers, labels):
        dwg.add(
            Circle(
                center=center,
                r=4,
                style=f"fill:{'#ff0000' if label == 1 else 'none'};stroke: #fbfbfb;stroke-width: 1;stroke-opacity: 1;",
            )
        )
    dwg.save(True)
    return dwg.tostring()
