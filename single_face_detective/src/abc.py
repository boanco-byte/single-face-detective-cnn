import cv2
import os

img_path = r"C:/Python/single_face_detective/dataset/train/images/Image-from-iOS_MOV-11_jpg.rf.Nz0Lqatmsw8dZnQ0EY6g.jpg"
label_path = r"C:/Python/single_face_detective/dataset/train/labels/Image-from-iOS_MOV-11_jpg.rf.Nz0Lqatmsw8dZnQ0EY6g.txt"

image = cv2.imread(img_path)
Height, Width, Channels = image.shape

with open(label_path, 'r') as f:
    string_data = f.readline().strip()

    elements = [float(x) for x in string_data.split()]

cx = elements[1]
cy = elements[2]
w = elements[3]
h = elements[4]

x1 = int((cx - w/2) * Width)
y1 = int((cy - h/2) * Height)
x2 = int((cx + w/2) * Width)
y2 = int((cy + h/2) * Height)

image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Rectangles", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



