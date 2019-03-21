## PersonLab

An Tensorflow implementation of PersonLab for Multi-Person Pose Estimation and Instance Segmentation. Identify every person instance, localize its facial and body keypoints, and estimate its instance segmentation mask.


### Introduction

Code repo for reproducing 2018 ECCV paper [PersonLab: Person Pose Estimation and Instance Segmentation with a Bottom-Up, Part-Based, Geometric Embedding Model](https://link.springer.com/chapter/10.1007/978-3-030-01264-9_17) 

### Result

**Pose**

![pose](https://github.com/scnuhealthy/Tensorflow_PersonLab/blob/master/demo_result/pose.jpg)

**Segmentation**

![segmentation](https://github.com/scnuhealthy/Tensorflow_PersonLab/blob/master/demo_result/instances_masks.jpg)

### Require

* Python3

* Tensorflow 1.80
* pycocotools  2.0
* skimage  0.13.0
* python-opencv 3.4.1



### Demo

* Download the [model]()
* python demo.py to run the demo and visualize the model result



### Training

* Download the COCO 2017 dataset

  http://images.cocodataset.org/zips/train2017.zip

  http://images.cocodataset.org/zips/val2017.zip

  http://images.cocodataset.org/annotations/annotations_trainval2017.zip

  training images in `coco2017/train2017/` , val images in `coco2017/val2017/`, training annotations in `coco2017/annotations/`

* Download the [Resnet101](http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz) pretrained model, put the model in `./model/101/resnet_v2_101.ckpt`

* Edit the [config.py](https://github.com/scnuhealthy/Tensorflow_PersonLab/blob/master/config.py) to set options for training, e.g. dataset position, input tensor shape, learning rate. 
* Run the train.py script

#### Evaluation

* coming soon

### Technical Debts

The augmentation code (which is different from the procedure in the PersonLab paper) and data iterator code is heavily borrowed from [this fork](https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation) of the Keras implementation of CMU's "Realtime Multi-Person Pose Estimation". (The pose plotting function is also influenced by the one in that repo.)

The mask generation code and visualization code are from [this fork](https://github.com/octiapp/KerasPersonLab) of the Keras implementation of PersonLab.

### Citation

```
@inproceedings{papandreou2018personlab,
  title={PersonLab: Person pose estimation and instance segmentation with a bottom-up, part-based, geometric embedding model},
  author={Papandreou, George and Zhu, Tyler and Chen, Liang-Chieh and Gidaris, Spyros and Tompson, Jonathan and Murphy, Kevin},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={269--286},
  year={2018}
}
```
