import cv2
import numpy as np
import os
import json
import argparse


def segment_img(output_path, image_path, points):
    """Segment the image by applying a polygon mask."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return

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
    os.makedirs(output_path, exist_ok=True)

    # Save the masked image
    save_path = f'{output_path}/{filename}'
    cv2.imwrite(save_path, masked_image_bgr)
    print(f"Saved masked image to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment images by masking a polygonal area.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing input images.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder containing output images.")
    parser.add_argument("--points_json", type=str, required=True, help="Path to the JSON file containing polygon points.")

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    points_json = args.points_json

    # Load polygon points
    with open(points_json, 'r') as file:
        points = json.load(file)

    print(f"Polygon Points: {points}")

    # Process all images in the input folder
    for f in os.listdir(input_folder):
        if f.endswith(".jpg"):
            segment_img(output_folder, f"{input_folder}/{f}", points)
