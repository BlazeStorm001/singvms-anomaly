import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

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

if __name__ == "__main__":
    # Given prediction output as JSON
    preds = { 
        "predictions": [{
            "x": 500.0,
            "y": 500.0,
            "width": 1000,
            "height": 1000,
            "class": "hand",
            "confidence": 0.943
        }, {
            "x": 504.5,
            "y": 363.0,
            "width": 215,
            "height": 172,
            "class": "hand",
            "confidence": 0.917
        }, {
            "x": 400,
            "y": 400,
            "width": 50,
            "height": 52,
            "class": "hand",
            "confidence": 0.87
        }, {
            "x": 78.5,
            "y": 700.0,
            "width": 139,
            "height": 34,
            "class": "hand",
            "confidence": 0.404
        }]
    }

    # Assume the image is 2000x1000
    image_width = 1000
    image_height = 1000

    occupation_ratio = calculate_occupation_area_shapely(preds, image_width, image_height)
    print(f"Occupation Ratio: {occupation_ratio:.4f}")
