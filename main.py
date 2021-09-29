import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import imutils
import os

def get_pts(file_pts):
    pts = file_pts.readlines()
    pts = pts[3:len(pts) - 1]
    points = np.zeros((len(pts), 2))
    for i in range(len(pts)):
        pts[i] = pts[i].replace('\n', '')
        x = pts[i].split(' ')
        points[i][0] = float(x[0])
        points[i][1] = float(x[1])

    angle = math.atan2(points[31, 1] - points[36, 1], points[31, 0] - points[36, 0]) * 180 / math.pi
    if angle > 90:
        angle = angle - 180
    if angle < -90:
        angle = angle + 180
    return  points,angle

def main():
    image_path = "FGNET/images"
    pts_path = "FGNET/points"

    for filename in os.listdir(image_path):
        image = cv2.imread(os.path.join(image_path, filename))
        name_pts = filename.split('.')[0].lower()+".pts"

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        file_pts = open(os.path.join(pts_path,name_pts))

        points,angle = get_pts(file_pts)

        rotated = imutils.rotate(image, angle)
        m_i = min(points[16][1], points[22][1])
        if filename!="033A30.JPG":
            croped = rotated[int(m_i - 10):int(points[7][1]), int(points[0][0] - 10):int(points[14][0] + 10)]
        else:
            croped = rotated[int(m_i - 10):int(points[7][1]), int(points[0][0] ):int(points[14][0] + 5)]
        rot_crop_his = cv2.equalizeHist(croped)
        try:
            cv2.imwrite(os.path.join("C:/Users/Hosein/PycharmProjects/untitled/venv/Scripts/images_new", filename),rot_crop_his)
        except:
            print(filename)


if "__main__":
    main()

