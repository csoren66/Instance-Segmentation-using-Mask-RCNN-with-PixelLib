import pixellib
from pixellib.instance import instance_segmentation
import cv2

segment_image = instance_segmentation()
segment_image.load_model("mask_rcnn_coco.h5")

camera = cv2.VideoCaputure(0)
while camera.isOpened():
    res, frame = camera.read()

    #Appy segmentation
    result = segment_image.segmentFrame(frame, show_bboxes=True)
    image = result[1]
    cv2.imshow('Image Segmentation', Image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()