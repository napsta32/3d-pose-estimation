"""
@author: Wenbo Li
@contact: fenglinglwb@gmail.com
"""

import cv2
import json
import numpy as np
import os
import h5py

from dataset.JointsDataset import JointsDataset


class H36MDataset(JointsDataset):

    def __init__(self, DATASET, stage, transform=None):
        super().__init__(DATASET, stage, transform)
        if self.stage == 'train':
            self.subjects = DATASET.TRAIN.SUBJECTS
        else:
            self.subjects = DATASET.TEST.SUBJECTS

        self.bb = np.load('/data/H36M/bb/bboxes-Human36M-GT.npy', allow_pickle=True).item()

        self.data = list()
        for subject in self.subjects:
            subject_dir = '/data/H36M/images/{}'.format(subject)
            actions = sorted(os.listdir(subject_dir)) # Adding sorted for consistency
            print('Loading metadata for subject {} (0/{})'.format(subject, len(actions)), end='\r')

            subject_data = list()
            for i, action in enumerate(actions):
                if action == 'MySegmentsMat':
                    continue
                action_dir = os.path.join(subject_dir, action)
                annot_file = os.path.join(action_dir, 'annot.h5')
                annot = h5py.File(annot_file, 'r')
                subject_data += self._get_action_data(subject, action, action_dir, annot, len(self.data) + len(subject_data))
                print('Loading metadata for subject {} ({}/{})'.format(subject, i+1, len(actions)), end='\r')
            print()
            np.savez('/app/dataset/H36M/gt/{}.npz'.format(subject), data=subject_data)
            self.data += subject_data

        self.data_num = len(self.data)

    def _get_action_data(self, subject, action, action_dir, annot, min_id):
        data = list()

        for i, frame in enumerate(annot['frame']):
            camera = str(annot['camera'][i])
            pose2d = annot['pose']['2d'][i]
            if len(self.bb[subject][action][camera]) < frame:
                # No more data to show - It looks like an error, hope it doesn't happen
                print('no more data to show ({}/{})'.format(len(self.bb[subject][action][camera]), frame))
                continue
            bbox = self.bb[subject][action][camera][frame-1]
            x1, y1, x2, y2 = (bbox[0], bbox[1], bbox[2], bbox[3])
            markers =  [0, 1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 17, 18, 19, 25, 26, 27]

            img_id = '{}'.format(min_id + i)
            img_path = os.path.join(action_dir, 'imageSequence-undistorted', camera, 'img_%06d.jpg' % (frame,))
            joints = pose2d[markers]
            center, scale = self._bbox_to_center_and_scale((x1, y1, x2-x1, y2-y1))

            data.append(dict(img_id=img_id,
                img_path=img_path,
                joints=joints,
                center=center,
                scale=scale,
            ))
        
        return data

    def _get_data(self):
        data = list()

        if self.stage == 'train':
            coco = COCO(self.train_gt_path)
        elif self.stage == 'val':
            coco = COCO(self.val_gt_path)
            self.val_gt = coco
        else:
            pass

        if self.stage == 'train':
            for aid, ann in coco.anns.items():
                img_id = ann['image_id']
                if img_id not in coco.imgs \
                        or img_id in self._exception_ids:
                    continue
                
                if ann['iscrowd']:
                    continue

                img_name = coco.imgs[img_id]['file_name']
                prefix = 'val2014' if 'val' in img_name else 'train2014'
                img_path = os.path.join(self.cur_dir, 'images', prefix,
                        img_name)

                bbox = np.array(ann['bbox'])
                area = ann['area']
                joints = np.array(ann['keypoints']).reshape((-1, 3))
                headRect = np.array([0, 0, 1, 1], np.int32)

                center, scale = self._bbox_to_center_and_scale(bbox)

                if np.sum(joints[:, -1] > 0) < self.kp_load_min_num or \
                        ann['num_keypoints'] == 0:
                    continue

                d = dict(aid=aid,
                         area=area,
                         bbox=bbox,
                         center=center,
                         headRect=headRect,
                         img_id=img_id,
                         img_name=img_name,
                         img_path=img_path,
                         joints=joints,
                         scale=scale)
                
                data.append(d)

        else:
            if self.stage == 'val':
                det_path = self.val_det_path
            else:
                det_path = self.test_det_path
            dets = json.load(open(det_path))

            for det in dets:
                if det['image_id'] not in coco.imgs or det['category_id'] != 1:
                    continue

                img_id = det['image_id']
                img_name = 'COCO_val2014_000000%06d.jpg' % img_id 
                img_path = os.path.join(self.cur_dir, 'images', 'val2014',
                        img_name)

                bbox = np.array(det['bbox'])
                center, scale = self._bbox_to_center_and_scale(bbox)
                joints = np.zeros((self.keypoint_num, 3))
                score = det['score']
                headRect = np.array([0, 0, 1, 1], np.int32)

                d = dict(bbox=bbox,
                         center=center,
                         headRect=headRect,
                         img_id=img_id,
                         img_name=img_name,
                         img_path=img_path,
                         joints=joints,
                         scale=scale,
                         score=score)

                data.append(d)

        return data

    def _bbox_to_center_and_scale(self, bbox):
        x, y, w, h = bbox

        center = np.zeros(2, dtype=np.float32)
        center[0] = x + w / 2.0
        center[1] = y + h / 2.0

        scale = np.array([w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
                dtype=np.float32)

        return center, scale

    def evaluate(self, pred_path):
        pred = self.val_gt.loadRes(pred_path)
        coco_eval = COCOeval(self.val_gt, pred, iouType='keypoints')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    def visualize(self, img, joints, score=None):
        pairs = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        color = np.random.randint(0, 256, (self.keypoint_num, 3)).tolist()

        for i in range(self.keypoint_num):
            if joints[i, 0] > 0 and joints[i, 1] > 0:
                cv2.circle(img, tuple(joints[i, :2]), 2, tuple(color[i]), 2)
        if score:
            cv2.putText(img, score, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (128, 255, 0), 2)

        def draw_line(img, p1, p2):
            c = (0, 0, 255)
            if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                cv2.line(img, tuple(p1), tuple(p2), c, 2)

        for pair in pairs:
            draw_line(img, joints[pair[0] - 1], joints[pair[1] - 1])

        return img

# Manual testing util
def __save_img(imgfile: str, joints, outfile: str):
    img = mpimg.imread(imgfile)
    
    fig, ax = plt.subplots()
    fig.set_figheight(30)
    fig.set_figwidth(20)
    
    ax.imshow(img)
    ax.scatter(joints[:,0], joints[:,1])
    for i in range(joints.shape[0]):
        ax.text(joints[i, 0], joints[i, 1], str(i), size=16, color='green')
        pass

    plt.savefig(outfile)

# Manual testing
if __name__ == '__main__':
    from dataset.attribute import load_dataset
    ###
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.lines import Line2D
    ###

    dataset = load_dataset('H36M')
    h36m = H36MDataset(dataset, 'train')
    print(h36m.data_num)
    samples = [0, 70000, 433000, 190000, 600000]
    # samples = [0]
    for i, x in enumerate(samples):
        imgfile = 'train{}-{}.png'.format(x, i)
        data = h36m[x]
        print(data[0].shape)
        print(data[1].shape)
        print(data[3].shape)
        cv2.imwrite(imgfile, data[0])
        __save_img(imgfile, data[3], 'markers-{}.png'.format(x))
