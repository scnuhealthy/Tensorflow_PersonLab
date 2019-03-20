import tensorflow as tf   
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from config import config
from bilinear import bilinear_sampler
slim = tf.contrib.slim

def refine(base,offsets,num_steps=2):
    for i in range(num_steps):
        base = base + bilinear_sampler(offsets,base)
    return base

def split_and_refine_mid_offsets(mid_offsets, short_offsets):
    output_mid_offsets = []
    for mid_idx, edge in enumerate(config.EDGES+[edge[::-1] for edge in config.EDGES]):
        to_keypoint = edge[1]
        kp_short_offsets = short_offsets[:,:,:,2*to_keypoint:2*to_keypoint+2]
        kp_mid_offsets = mid_offsets[:,:,:,2*mid_idx:2*mid_idx+2]
        kp_mid_offsets = refine(kp_mid_offsets,kp_short_offsets,2)
        output_mid_offsets.append(kp_mid_offsets)
    return tf.concat(output_mid_offsets,axis=-1)

def split_and_refine_long_offsets(long_offsets, short_offsets):
    output_long_offsets = []
    for i in range(config.NUM_KP):
        kp_long_offsets = long_offsets[:,:,:,2*i:2*i+2]
        kp_short_offsets = short_offsets[:,:,:,2*i:2*i+2]
        refine_1 = refine(kp_long_offsets,kp_long_offsets)
        refine_2 = refine(refine_1,kp_short_offsets)
        output_long_offsets.append(refine_2)
    return tf.concat(output_long_offsets,axis=-1)


#inputs=tf.random_uniform((batch_size,height,width,3),dtype=tf.float32)

def model(inputs):

    batch_size,height,width=config.BATCH_SIZE,config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1]
    
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
          #net, end_points = resnet_v2.resnet_v2_101(inputs, 1001, is_training=False)
          net, end_points = resnet_v2.resnet_v2_101(inputs,
                                                    2048,
                                                    is_training=True,
                                                    global_pool=False,
                                                    reuse=tf.AUTO_REUSE,
                                                    output_stride=config.OUTPUT_STRIDE)
    # print(net)
    kp_maps = tf.contrib.layers.conv2d(net,num_outputs = config.NUM_KP,
                                             kernel_size=(1,1),activation_fn=tf.nn.sigmoid,stride=1,scope='kp_maps',reuse=tf.AUTO_REUSE)
    short_offsets = tf.contrib.layers.conv2d(net,num_outputs = 2*config.NUM_KP,
                                             kernel_size=(1,1),activation_fn=None,stride=1,scope='short_offsets',reuse=tf.AUTO_REUSE)
    mid_offsets = tf.contrib.layers.conv2d(net,num_outputs = 4*config.NUM_EDGES,
                                             kernel_size=(1,1),activation_fn=None,stride=1,scope='mid_offsets',reuse=tf.AUTO_REUSE)
    long_offsets = tf.contrib.layers.conv2d(net,num_outputs = 2*config.NUM_KP,
                                             kernel_size=(1,1),activation_fn=None,stride=1,scope='long_offsets',reuse=tf.AUTO_REUSE)
    seg_mask = tf.contrib.layers.conv2d(net,num_outputs = 1,
                                             kernel_size=(1,1),activation_fn=tf.nn.sigmoid,stride=1,scope='seg_mask',reuse=tf.AUTO_REUSE)
 
    kp_maps = tf.image.resize_bilinear(kp_maps, (height,width), align_corners=True)
    short_offsets = tf.image.resize_bilinear(short_offsets, (height,width), align_corners=True)
    mid_offsets = tf.image.resize_bilinear(mid_offsets, (height,width), align_corners=True)
    long_offsets = tf.image.resize_bilinear(long_offsets, (height,width), align_corners=True)
    seg_mask = tf.image.resize_bilinear(seg_mask, (height,width), align_corners=True)
    
    '''
    with tf.name_scope('kp_maps_deconv') as scope:
        wt = tf.Variable(tf.truncated_normal([9, 9, config.NUM_KP, config.NUM_KP]))
        kp_maps = tf.nn.conv2d_transpose(kp_maps, wt, [batch_size, height, width, config.NUM_KP], [1, 8, 8, 1], 'SAME')
    
    with tf.name_scope('short_offsets_deconv') as scope:
        wt = tf.Variable(tf.truncated_normal([9, 9, 2*config.NUM_KP, 2*config.NUM_KP]))
        short_offsets = tf.nn.conv2d_transpose(short_offsets, wt, [batch_size, height, width, 2*config.NUM_KP], [1, 8, 8, 1], 'SAME')
    
    with tf.name_scope('mid_offsets_deconv') as scope:
        wt = tf.Variable(tf.truncated_normal([9, 9, 4*config.NUM_EDGES, 4*config.NUM_EDGES]))
        mid_offsets = tf.nn.conv2d_transpose(mid_offsets, wt, [batch_size, height, width, 4*config.NUM_EDGES], [1, 8, 8, 1], 'SAME')
    
    with tf.name_scope('long_offsets_deconv') as scope:
        wt = tf.Variable(tf.truncated_normal([9, 9, 2*config.NUM_KP, 2*config.NUM_KP]))
        long_offsets = tf.nn.conv2d_transpose(long_offsets, wt, [batch_size, height, width, 2*config.NUM_KP], [1, 8, 8, 1], 'SAME')
    
    with tf.name_scope('seg_mask_deconv') as scope:
        wt = tf.Variable(tf.truncated_normal([9, 9, 1, 1]))
        seg_mask = tf.nn.conv2d_transpose(seg_mask, wt, [batch_size, height, width, 1], [1, 8, 8, 1], 'SAME')
    '''
    mid_offsets = split_and_refine_mid_offsets(mid_offsets, short_offsets)
    long_offsets = split_and_refine_long_offsets(long_offsets, short_offsets)
    outputs = [kp_maps,short_offsets,mid_offsets,long_offsets,seg_mask]
    return outputs

