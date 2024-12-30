import cv2

points = []

def get_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:  # Left-click to add a point
        points.append((x, y))
        print(f"({x}, {y})")

# Load image
image_path = "data\\3796\\raw\\3796_20241216_223631.jpg"
image = cv2.imread(image_path)

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Make the window resizable
cv2.setMouseCallback("Image", get_points)

while True:
    for point in points:
        cv2.circle(image, point, 5, (0, 255, 0), -1)  # Draw the points on the image

    cv2.imshow("Image", image)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'ESC' to exit
        break

cv2.destroyAllWindows()
print("Final Points:", points)
