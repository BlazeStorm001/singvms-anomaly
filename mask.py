import cv2
import json
import numpy as np
import argparse

points = []

def get_points(event, x, y, flags, param):
    """Callback function to capture points on mouse click."""
    global points
    if event == cv2.EVENT_LBUTTONDOWN:  # Left-click to add a point
        points.append((x, y))
        print(f"Point added: ({x}, {y})")

def load_image(image_path):
    """Load an image from the given path."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image

def display_image(image):
    """Display the image and capture points from user input."""
    window_title = "Press ESCAPE when done"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)  # Make the window resizable
    cv2.setMouseCallback(window_title, get_points)

    try:
        while True:
            temp_image = image.copy()  # Create a copy of the image to draw on

            for point in points:
                cv2.circle(temp_image, point, 5, (0, 255, 0), -1)  # Draw the points on the image

            if len(points) > 1:
                cv2.polylines(temp_image, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)  # Draw the polygon

            cv2.imshow(window_title, temp_image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:  # Press 'ESC' or close window to exit
                break
    finally:
        cv2.destroyAllWindows()
        print("Final Points:", points)
        save_points_as_json(points)

def save_points_as_json(points):
    """Save the points as a JSON file."""
    with open("points.json", "w") as f:
        json.dump(points, f)
    print("Points saved to points.json")

def main():
    parser = argparse.ArgumentParser(description="Select points on an image and save them as a JSON file.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()

    image_path = args.image_path

    try:
        image = load_image(image_path)
        display_image(image)
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()
