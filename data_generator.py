#import tensorflow as tf
import cv2
import numpy as np
from pycocotools.coco import COCO
from skimage import io
import os
from transformer import Transformer, AugmentSelection
from config import config, TransformationParams
from data_pred import *
from matplotlib import pyplot as plt

ANNO_FILE = config.ANNO_FILE
IMG_DIR = config.IMG_DIR


class DataGeneraotr(object):
    def __init__(self):  
        self.coco = COCO(ANNO_FILE)
        self.img_ids = list(self.coco.imgs.keys())
        self.datasetlen = len(self.img_ids)
        self.id = 0
    
    def get_multi_scale_img(self,give_id,scale):
        img_id = give_id
        filepath = os.path.join(IMG_DIR,self.coco.imgs[img_id]['file_name'])
        img = cv2.imread(filepath)
        cv_shape = (config.IMAGE_SHAPE[1], config.IMAGE_SHAPE[0])
        cv_shape2 = (int(cv_shape[0]*scale),int(cv_shape[1]*scale))
        max_shape = max(img.shape[0],img.shape[1])
        scale2 = cv_shape2[0]/max_shape
        img = cv2.resize(img,None,fx=scale2,fy=scale2)
        img = cv2.copyMakeBorder(img,0,cv_shape2[0]-img.shape[0],0,cv_shape2[1]-img.shape[1],cv2.BORDER_CONSTANT,value=[127,127,127])
        return img

    
    def get_one_sample(self,give_id=None,is_aug=True):
        if self.id == self.datasetlen:
            self.id = 0
        if give_id==None:
            img_id = self.img_ids[self.id]
        else:
            img_id = give_id
        filepath = os.path.join(IMG_DIR,self.coco.imgs[img_id]['file_name'])
        img = cv2.imread(filepath)
        h, w, c = img.shape
        # read the annotation, and get the keypoints and masks
        crowd_mask = np.zeros((h, w), dtype='bool')
        unannotated_mask = np.zeros((h,w), dtype='bool')
        instance_masks = []
        keypoints = []
        img_anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
         
        for anno in img_anns:
            # if crowd, don't compute loss
            mask = self.coco.annToMask(anno)
            if anno['iscrowd'] ==1:
                crowd_mask = np.logical_or(crowd_mask,mask)
            # if tiny instance, don't compute loss
            elif anno['num_keypoints'] == 0:
                unannotated_mask = np.logical_or(unannotated_mask, mask)
                instance_masks.append(mask)
                keypoints.append(anno['keypoints'])
            else:
                instance_masks.append(mask)
                keypoints.append(anno['keypoints'])
        if len(instance_masks)<=0:
            self.id +=1
            return None

        kp = np.reshape(keypoints, (-1, config.NUM_KP, 3))
        instance_masks = np.stack(instance_masks).transpose((1,2,0))
        overlap_mask = instance_masks.sum(axis=-1) > 1
        seg_mask = np.logical_or(crowd_mask,np.sum(instance_masks,axis=-1))
        
        # Data Augmentation
        single_masks = [seg_mask, unannotated_mask, crowd_mask, overlap_mask]
        all_masks = np.concatenate([np.stack(single_masks, axis=-1), instance_masks], axis=-1)
        if is_aug:
            aug = AugmentSelection.random()
        else:
            aug = AugmentSelection.unrandom()
        img, all_masks, kp = Transformer.transform(img, all_masks, kp, aug=aug) 

        num_instances = instance_masks.shape[-1]
        instance_masks = all_masks[:,:, -num_instances:]
        seg_mask, unannotated_mask, crowd_mask, overlap_mask = all_masks[:,:, :4].transpose((2,0,1))
        seg_mask, unannotated_mask, crowd_mask, overlap_mask = [np.expand_dims(m, axis=-1) for m in [seg_mask, unannotated_mask, crowd_mask, overlap_mask]]

        # the area not to compute loss is set 0
        unannotated_mask = np.logical_not(unannotated_mask)
        crowd_mask = np.logical_not(crowd_mask)
        overlap_mask = np.logical_not(overlap_mask)
        
        # get ground truth from keypoints
        kp = [np.squeeze(k) for k in np.split(kp, kp.shape[0], axis=0)]
        print(kp)
        kp_maps, short_offsets, mid_offsets, long_offsets = get_ground_truth(instance_masks, kp)
        # print(img.shape,kp_maps.shape,short_offsets.shape,mid_offsets.shape,long_offsets.shape)
        # shape: img(401,401,3) kp_maps(401,401,17) short(401,401,34) medium(401,401,64) long(401,401,34) 
        self.id +=1
        return [img.astype('float32')/255.0,kp_maps.astype('float32'),short_offsets.astype('float32'),
               mid_offsets.astype('float32'),long_offsets.astype('float32'),seg_mask.astype('float32'),
               crowd_mask.astype('float32'),unannotated_mask.astype('float32'),overlap_mask.astype('float32')]
                
    def gen_batch(self,batch_size=4):
        h,w,c = config.IMAGE_SHAPE
        while True:
            imgs_batch = np.zeros((batch_size,h,w,c))
            kp_maps_batch = np.zeros((batch_size,h,w,config.NUM_KP))
            short_offsets_batch = np.zeros((batch_size,h,w,2*config.NUM_KP))
            mid_offsets_batch = np.zeros((batch_size,h,w,4*(config.NUM_EDGES)))
            long_offsets_batch = np.zeros((batch_size,h,w,2*config.NUM_KP))
            seg_mask_batch = np.zeros((batch_size,h,w,1))
            crowd_mask_batch = np.zeros((batch_size,h,w,1))
            unannotated_mask_batch = np.zeros((batch_size,h,w,1))
            overlap_mask_batch = np.zeros((batch_size,h,w,1))
            
            for i in range(batch_size):
                sample = self.get_one_sample()
                while sample ==None: # not to train the images with no instance
                    sample = self.get_one_sample()           
                imgs_batch[i] = sample[0]
                kp_maps_batch[i] = sample[1]
                short_offsets_batch[i] = sample[2]
                mid_offsets_batch[i] = sample[3]
                long_offsets_batch[i] = sample[4]
                seg_mask_batch[i] = sample[5]
                crowd_mask_batch[i] = sample[6]
                unannotated_mask_batch[i] = sample[7]
                overlap_mask_batch[i] = sample[8]
             
            yield [imgs_batch,kp_maps_batch,short_offsets_batch,mid_offsets_batch,long_offsets_batch,
                  seg_mask_batch,crowd_mask_batch,unannotated_mask_batch,overlap_mask_batch]
                    
        
#plt.imshow(batch[5][2][:,:,0])        
dataset = DataGeneraotr()
#dataset.get_one_sample()
#batch = next(dataset.gen_batch())
img = dataset.get_multi_scale_img(13291,0.5)
plt.imshow(img)  