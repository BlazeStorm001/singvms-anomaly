import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

def get_poly_area(points):
    """Calculate the area of a polygon defined by a list of points."""
    area = 0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) / 2

def calculate_occupation_area_shapely(predictions, roi_area):
    """
    Calculate the occupation area of bounding boxes on an image using Shapely.
    
    Args:
        predictions (dict): Dictionary containing predictions with bounding boxes.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        
    Returns:
        float: Occupation ratio (area of bounding boxes / total image area).
    """
    # List to store Shapely box objects
    polygons = []

    # Iterate over each bounding box
    for pred in predictions["predictions"]:
        x_center, y_center = pred["x"], pred["y"]
        width, height = pred["width"], pred["height"]

        # Calculate the top-left and bottom-right coordinates
        x1, y1 = x_center - width / 2, y_center - height / 2
        x2, y2 = x_center + width / 2, y_center + height / 2

        # Create a Shapely box (polygon) for the bounding box
        polygons.append(box(x1, y1, x2, y2))
    
    # Compute the union of all bounding boxes
    union_polygon = unary_union(polygons)

    # Calculate the union area
    union_area = union_polygon.area


    # Calculate the occupation ratio
    occupation_ratio = union_area / roi_area

    return occupation_ratio
