import tensorflow as tf 
from config import config
import model
import numpy as np
from data_generator import DataGeneraotr
import os 
slim = tf.contrib.slim

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def count1():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)

def tf_repeat(tensor, repeats):
    """
    From  https://github.com/tensorflow/tensorflow/issues/8246
    
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:
    
    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
        repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tensor


def kp_map_loss(kp_maps_true,kp_maps_pred,unannotated_mask,crowd_mask):
    loss = tf.keras.backend.binary_crossentropy(kp_maps_true,kp_maps_pred)
    loss = loss*crowd_mask*unannotated_mask
    loss = tf.reduce_mean(loss)*config.LOSS_WEIGHTS['heatmap']
    return loss

def short_offset_loss(short_offset_true,short_offsets_pred,kp_maps_true):
    loss = tf.abs(short_offset_true-short_offsets_pred)/config.KP_RADIUS
    loss = loss*tf_repeat(kp_maps_true,[1,1,1,2])
    loss = tf.reduce_sum(loss) / (tf.reduce_sum(kp_maps_true)+1)
    return loss*config.LOSS_WEIGHTS['short']

def mid_offset_loss(mid_offset_true,mid_offset_pred,kp_maps_true):
    loss = tf.abs(mid_offset_pred-mid_offset_true)/config.KP_RADIUS
    recorded_maps = []
    for mid_idx, edge in enumerate(config.EDGES + [edge[::-1] for edge in config.EDGES]):
        from_kp = edge[0]
        recorded_maps.extend([kp_maps_true[:,:,:,from_kp], kp_maps_true[:,:,:,from_kp]])
    recorded_maps = tf.stack(recorded_maps,axis=-1)
    # print(recorded_maps)
    loss = loss*recorded_maps
    loss = tf.reduce_sum(loss)/(tf.reduce_sum(recorded_maps)+1)
    return loss*config.LOSS_WEIGHTS['mid']

def long_offset_loss(long_offset_true,long_offsets_pred,seg_true,crowd_mask,unannotated_mask,overlap_mask):
    loss = tf.abs(long_offsets_pred-long_offset_true)/config.KP_RADIUS
    instances = seg_true*crowd_mask*unannotated_mask*overlap_mask
    loss = loss*instances
    loss = tf.reduce_sum(loss)/(tf.reduce_sum(instances)+1)
    return loss*config.LOSS_WEIGHTS['long']

def segmentation_loss(seg_true,seg_pred,crowd_mask):
    loss = tf.keras.backend.binary_crossentropy(seg_true,seg_pred)
    loss = loss*crowd_mask
    return tf.reduce_mean(loss)*config.LOSS_WEIGHTS['seg']

def get_losses(ground_truth,outputs):
    kp_maps_true, short_offset_true, mid_offset_true, long_offset_true, seg_true, crowd_mask, unannotated_mask, overlap_mask = ground_truth
    kp_maps, short_offsets, mid_offsets, long_offsets, seg_mask = outputs
    losses = []
    losses.append(kp_map_loss(kp_maps_true,kp_maps,unannotated_mask,crowd_mask))
    losses.append(short_offset_loss(short_offset_true,short_offsets,kp_maps_true))
    losses.append(mid_offset_loss(mid_offset_true,mid_offsets,kp_maps_true))
    losses.append(long_offset_loss(long_offset_true, long_offsets,seg_true,crowd_mask,unannotated_mask,overlap_mask))
    losses.append(segmentation_loss(seg_true,seg_mask,crowd_mask))
    return losses

def train(load_pretrained_model=True,checkpoint_path=None):
    batch_size,height,width=config.BATCH_SIZE,config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1]
    img = tf.placeholder(tf.float32,shape=[batch_size,height,width,3])
    kp_maps_true = tf.placeholder(tf.float32,shape=[batch_size,height,width,config.NUM_KP])
    short_offsets_true = tf.placeholder(tf.float32,shape=[batch_size,height,width,2*config.NUM_KP])
    mid_offsets_true = tf.placeholder(tf.float32,shape=[batch_size,height,width,4*config.NUM_EDGES])
    long_offsets_true = tf.placeholder(tf.float32,shape=[batch_size,height,width,2*config.NUM_KP])
    seg_mask_true = tf.placeholder(tf.float32,shape=[batch_size,height,width,1])
    crowd_mask = tf.placeholder(tf.float32,shape=[batch_size,height,width,1])
    unannotated_mask = tf.placeholder(tf.float32,shape=[batch_size,height,width,1])
    overlap_mask = tf.placeholder(tf.float32,shape=[batch_size,height,width,1])
    
    # forward propagation
    outputs = model.model(img) 
    ground_truth = [kp_maps_true, short_offsets_true, mid_offsets_true, long_offsets_true, seg_mask_true, crowd_mask, unannotated_mask, overlap_mask]
    loss = get_losses(ground_truth,outputs)
    exclusions = ['resnet_v2_101/logits']
    param_except_logits = slim.get_variables_to_restore(include=['resnet_v2_101'],exclude=exclusions)
    print("[*]\tModel Build Finished!")

    # back propagation
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(config.LEARNING_RATE)
    train_step = optimizer.minimize(sum(loss)/batch_size)
    dataset = DataGeneraotr()
    
    # initializer
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    # saver and load checkpoint
    global_vars = tf.global_variables()
    saver = tf.train.Saver(var_list = global_vars)
    if checkpoint_path!=None:
        saver.restore(sess,checkpoint_path)
        print("[*]\tSESS Restored!")
    
    # load pretrained model
    if load_pretrained_model:
        init_fn = slim.assign_from_checkpoint_fn(config.PRETRAINED_MODEL_PATH,param_except_logits)
        init_fn(sess)    
        print("[*]\tPretrained Model Restored!")
        

    print("[*]\tDataset Build Finished!")

    # start training
    print("[*]\tTraining Started!")
    for n in range(config.NUM_EPOCHS):
        for m in range(config.NUM_EPOCHS_SIZE):
            batch = next(dataset.gen_batch(batch_size=batch_size))
            print("[*]\tOne Batch Generated!")
            feed_dict = {img:batch[0],kp_maps_true:batch[1],short_offsets_true:batch[2],mid_offsets_true:batch[3],long_offsets_true:batch[4],
                         seg_mask_true:batch[5],crowd_mask:batch[6],unannotated_mask:batch[7],overlap_mask:batch[8]}
            _,train_loss = sess.run([train_step,loss],feed_dict=feed_dict)
            iters = n*config.NUM_EPOCHS_SIZE+m
            if iters%1==0:
                print('[*]\titers:'+str(iters)+',loss:',train_loss)
                print('[*]\titers:'+str(iters)+',total loss:',sum(train_loss))
        
        saver.save(sess,os.path.join(config.SAVE_MODEL_PATH,'model.ckpt'),n)

train()


