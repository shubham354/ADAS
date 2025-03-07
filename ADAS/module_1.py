import cv2
import numpy as np

def detect_lane_lines(image_path):
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    height, width = edges.shape
    mask = np.zeros_like(edges)

    polygon = np.array([[(0, height), (width, height), (width//2, height//2)]])
    cv2.fillPoly(mask, [polygon], 255)

    masked_edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow("Lane Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_lane_lines('test_road.jpg')
