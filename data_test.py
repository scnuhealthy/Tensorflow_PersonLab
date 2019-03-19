import cv2
import numpy as np
from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt

import os
from transformer import Transformer, AugmentSelection
from config import config, TransformationParams
from data_pred import *
ANNO_FILE = 'E:/dataset/coco2017/annotations/person_keypoints_val2017.json'
IMG_DIR = 'E:/dataset/coco2017/val2017'

coco = COCO(ANNO_FILE)
img_ids = list(coco.imgs.keys())

img_id = img_ids[0]
filepath = os.path.join(IMG_DIR,coco.imgs[img_id]['file_name'])
img = cv2.imread(filepath)
io.imsave('1.jpg',img)

h, w, c = img.shape
crowd_mask = np.zeros((h, w), dtype='bool')
unannotated_mask = np.zeros((h,w), dtype='bool')
instance_masks = []
keypoints = []
img_anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
#print(img_anns)

for anno in img_anns:
    mask = coco.annToMask(anno)
    if anno['iscrowd'] ==1:
        crowd_mask = np.logical_or(crowd_mask,mask)
    elif anno['num_keypoints'] == 0:
        unannotated_mask = np.logical_or(unannotated_mask, mask)
        instance_masks.append(mask)
        keypoints.append(anno['keypoints'])
    else:
        instance_masks.append(mask)
        keypoints.append(anno['keypoints'])
    #plt.imshow(mask)
    #plt.show()
if len(instance_masks)<=0:
    pass
kp = np.reshape(keypoints, (-1, config.NUM_KP, 3))
instance_masks = np.stack(instance_masks).transpose((1,2,0))
overlap_mask = instance_masks.sum(axis=-1) > 1
seg_mask = np.logical_or(crowd_mask,np.sum(instance_masks,axis=-1))
print(kp.shape)

# Data Augmentation
single_masks = [seg_mask, unannotated_mask, crowd_mask, overlap_mask]
all_masks = np.concatenate([np.stack(single_masks, axis=-1), instance_masks], axis=-1)
aug = AugmentSelection.unrandom()
img, all_masks, kp = Transformer.transform(img, all_masks, kp, aug=aug)

num_instances = instance_masks.shape[-1]
instance_masks = all_masks[:,:, -num_instances:]
seg_mask, unannotated_mask, crowd_mask, overlap_mask = all_masks[:,:, :4].transpose((2,0,1))
seg_mask, unannotated_mask, crowd_mask, overlap_mask = [np.expand_dims(m, axis=-1) for m in [seg_mask, unannotated_mask, crowd_mask, overlap_mask]]

kp = [np.squeeze(k) for k in np.split(kp, kp.shape[0], axis=0)]
kp_maps, short_offsets, mid_offsets, long_offsets = get_ground_truth(instance_masks, kp)

'''
# encode
encoding = np.argmax(np.stack([np.zeros((h,w))]+instance_masks, axis=-1), axis=-1).astype('uint8')
encoding = np.unpackbits(np.expand_dims(encoding, axis=-1), axis=-1)
# No image has more than 63 instance annotations, so the first 2 channels are zeros
encoding[:,:,0] = unannotated_mask.astype('uint8')
encoding[:,:,1] = crowd_mask.astype('uint8')
encoding = np.packbits(encoding, axis=-1)

# Decode
seg_mask = encoding > 0
encoding = np.unpackbits(np.expand_dims(encoding[:,:,0], axis=-1), axis=-1)
unannotated_mask = encoding[:,:,0].astype('bool')
crowd_mask = encoding[:,:,1].astype('bool')
encoding[:,:,:2] = 0
encoding = np.squeeze(np.packbits(encoding, axis=-1))

num_instances = int(encoding.max())
instance_masks = np.zeros((encoding.shape+(num_instances,)))
for i in range(num_instances):
    instance_masks[:,:,i] = encoding==i+1
'''