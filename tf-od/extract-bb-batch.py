import os
import pathlib
import itertools

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
import tensorflow_hub as hub

tf.get_logger().setLevel('ERROR')

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

pjoin = os.path.join
fexists = os.path.exists

outfile = 'efficient-bb.npz'
image_dir = '/root/data/images'
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
cameras = ['54138969', '55011271', '58860488', '60457274']


def load_image_into_numpy_array(path):
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (1, im_height, im_width, 3)).astype(np.uint8)

def visualize_bboxes(image_np, detection_result):
    scores = detection_result['detection_scores'][0]
    boxes = detection_result['detection_boxes'][0]
    labels = detection_result['detection_classes'][0]

    person_scores = scores * [label == 1 for label in labels]
    boxes = [box for i, box in enumerate(boxes) if scores[i] > 0.5]

    # For demo only:
    fig,ax = plt.subplots(1)
    ax.imshow(image_np[0])
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 *= image_np.shape[1]
        x2 *= image_np.shape[1]
        y1 *= image_np.shape[2]
        y2 *= image_np.shape[2]

        rect = patches.Rectangle((y1,x1),y2-y1,x2-x1,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.show()

def get_labels():
    label_map_path = 'object_detection/data/mscoco_label_map.pbtxt'
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
            label_map,
            max_num_classes=label_map_util.get_max_label_map_index(label_map),
            use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)
    return label_map_dict

def inferBoundingBox(image_path: str):
    image_np = load_image_into_numpy_array(image_path)

    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)
    
    return detections

def main():
    print('loading model...')
    model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d4/1")
    print('model loaded')

    PERSON_LABEL = get_labels()['person']

    for subject in subjects:
        actions = os.listdir(pjoin(image_dir, subject))
        for action in actions:
            if fexists(pjoin(image_dir, subject, action, outfile)):
                continue
            bbs = {camera: [] for camera in cameras}
            for camera in cameras:
                print('Processing {} / {} camera {}...'.format(subject, action, camera))
                images_filenames = os.listdir(pjoin(image_dir, subject, action, 'imageSequence', camera))
                images_paths = [pjoin(image_dir, subject, action, 'imageSequence', camera, f) for f in images_filenames]
                
                for _, batch in itertools.groupby(range(len(images_paths)), lambda k: k//10):
                    images = []

                    for i in batch:
                        image_path = images_paths[i]
                        image_filename = images_filenames[i]
                        
                        image_np = load_image_into_numpy_array(image_path)
                        images.append(image_np)
                        pass
                    
                    images_np = np.concatenate(images)

                    print('Running predictions on {} images ({})'.format(len(images_np), images_np.shape))
                    results = model(images_np)
                    result = {key:value.numpy() for key,value in results.items()}
                
                    print('Processing boxes...')
                    for i in batch:
                        image_filename = images_filenames[i]

                        scores = result['detection_scores'][i]
                        boxes = result['detection_boxes'][i]
                        labels = result['detection_classes'][i]
                        
                        person_scores = scores * [label == PERSON_LABEL for label in labels]
                        person_boxes = [box for i, box in enumerate(boxes) if scores[i] > 0.5]
                        
                        # For demo only:
                        # visualize_bboxes(image_np, result)
                        
                        bbs[camera].append({
                            'image': image_filename,
                            'scores': [score for score in person_scores if score > 0.5],
                            'boxes': person_boxes
                        })
                        pass
            
            np.savez(pjoin(image_dir, subject, action, outfile), data=bbs)
            print('Saving: {}'.format(pjoin(image_dir, subject, action, outfile)))
            pass
        pass
    print('Finished extracting bounding boxes')

if __name__ == '__main__':
    main()