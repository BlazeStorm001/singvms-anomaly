import cv2
import numpy as np
# import matplotlib.pyplot as plt
from pdb import set_trace
import os

def get_poly_area(points):
    area = 0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) / 2

def segment_img(cam_id, image_path, points):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

    polygon_points = np.array(points, dtype=np.int32)

    # Create a mask with the same dimensions as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Fill the polygon on the mask
    cv2.fillPoly(mask, [polygon_points], 255)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

    # Convert back to BGR for saving
    masked_image_bgr = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
    filename = image_path.split("/")[-1]
    SAVE_DIR = f"data/{cam_id}/masked/"
    os.makedirs(SAVE_DIR, exist_ok=True)
    # Save the masked image
    cv2.imwrite(f'{SAVE_DIR}/{filename}', masked_image_bgr)



if __name__ == "__main__":
    points = [(6, 537), (625, 287), (701, 220), (805, 222), (983, 346), (1082, 501), (1364, 701), (1775, 1024), (1845, 1078), (8, 1076)]
    print(get_poly_area(points))
    for f in os.listdir("data/3796/raw"):
        if f.endswith(".jpg"):
            segment_img(3796, f"data/3796/raw/{f}", points)
    segment_img(3796, "data/3796/raw/3796_20241210_184605.jpg", points)
