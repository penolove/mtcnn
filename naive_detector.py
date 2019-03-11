import argparse
import arrow

import caffe
from eyewitness.image_id import ImageId
from eyewitness.image_utils import ImageHandler, Image
from bistiming import SimpleTimer

from demo import MtcnnFaceDetector


def get_face_detector():
    caffe_model_path = "./model"
    return MtcnnFaceDetector(caffe_model_path)


if __name__ == '__main__':
    model_name = 'Mtcnn'
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    if args.gpu_id >= 0:
        caffe.set_device(args.gpu_id)
        caffe.set_mode_gpu()

    with SimpleTimer("Loading model %s" % model_name):
        object_detector = get_face_detector()
    raw_image_path = 'demo/test_image.jpg'
    image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
    image_obj = Image(image_id, raw_image_path=raw_image_path)
    with SimpleTimer("Predicting image with classifier"):
        detection_result = object_detector.detect(image_obj)
    print("detected %s objects" % len(detection_result.detected_objects))
    ImageHandler.draw_bbox(image_obj.pil_image_obj, detection_result.detected_objects)
    ImageHandler.save(image_obj.pil_image_obj, "detected_image/drawn_image.jpg")
