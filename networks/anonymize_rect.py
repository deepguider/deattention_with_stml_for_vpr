import cv2
import numpy as np
import os
import argparse

from ipdb import set_trace as bp


def anonymize_rect_gaussian(patch, factor=3.0):
    # Refer to : https://www.pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/
    # Blur and anonymize faces with OpenCV and Python
    # automatically determine the size of the blurring kernel based
    # on the spatial dimensions of the input image
    (h, w) = patch.shape[:2]
    kW = int(w / factor)
    kH = int(h / factor)
    # ensure the width of the kernel is odd
    if kW % 2 == 0:
        kW -= 1
    # ensure the height of the kernel is odd
    if kH % 2 == 0:
        kH -= 1
    # apply a Gaussian blur to the input image using our computed
    # kernel size
    try:
        if (kW > 0) and (kH > 0):
            blurred = cv2.GaussianBlur(patch, (kW, kH), 0)
        else:
            blurred = patch
    except:
        bp()
    return cv2.resize(blurred, (w, h), interpolation=cv2.INTER_NEAREST)


if __name__ == '__main__':
    src = cv2.imread('/home/ccsmm/workdir/cv2_examples/img/img1.jpg')
    out = src
    bboxes=[[223,140, 100, 100], [473,130,100,100]]
    for x, y, w, h in bboxes:
        patch = src[y: y + h, x: x + w]
        patch_blurred = anonymize_rect_gaussian(patch)
        #patch_blurred = anonymize_face_simple(patch)
        out[y: y + h, x: x + w] = patch_blurred
        # out = cv2.rectangle(out, (x,y), (x+w,y+h), (255,0,0), 1)
    cv2.imshow('img', out)
    cv2.waitKey(0)
