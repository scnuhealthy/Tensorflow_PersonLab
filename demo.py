from matplotlib import pyplot as plt
import tensorflow as tf 
from config import config
import model
from data_generator import DataGeneraotr
import numpy as np
from skimage import io

batch_size,height,width=1,config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1]
tf_img = tf.placeholder(tf.float32,shape=[batch_size,height,width,3])

outputs = model.model(tf_img) 
sess = tf.Session()

global_vars = tf.global_variables()
saver = tf.train.Saver(var_list = global_vars)
checkpoint_path = './model/personlab/'+'model.ckpt-7'
saver.restore(sess,checkpoint_path)
print("[*]\tSESS Restored!")
    
dataset = DataGeneraotr()
batch = dataset.get_one_sample(give_id=13291,is_aug=False) 
img = batch[0][:,:,[2,1,0]]
plt.figure()
plt.imshow(img)
imgs_batch = np.zeros((batch_size,config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1],3))
imgs_batch[0] = img
sample_output = sess.run(outputs,feed_dict={tf_img:imgs_batch})
sample_output = [o[0] for o in sample_output]

from plot import *
def overlay(img, over, alpha=0.5):
    out = img.copy()
    if img.max() > 1.:
        out = out / 255.
    out *= 1-alpha
    if len(over.shape)==2:
        out += alpha*over[:,:,np.newaxis]
    else:
        out += alpha*over    
    return out

Rshoulder_map = sample_output[0][:,:,config.KEYPOINTS.index('Rshoulder')]
plt.figure()
plt.imshow(overlay(img, Rshoulder_map, alpha=0.7))


from post_proc import *
# Gaussian filtering helps when there are multiple local maxima for the same keypoint.
H = compute_heatmaps(kp_maps=sample_output[0], short_offsets=sample_output[1])
for i in range(17):
    H[:,:,i] = gaussian_filter(H[:,:,i], sigma=2)
plt.figure()
plt.imshow(H[:,:,config.KEYPOINTS.index('Rshoulder')])
# The heatmaps are computed using the short offsets predicted by the network
# Here are the right shoulder offsets
visualize_short_offsets(offsets=sample_output[1], heatmaps=H, keypoint_id='Rshoulder', img=img, every=8)

# The connections between keypoints are computed via the mid-range offsets.
# We can visuzalize them as well; for example right shoulder -> right hip
visualize_mid_offsets(offsets= sample_output[2], heatmaps=H, from_kp='Rshoulder', to_kp='Rhip', img=img, every=8)

# And we can see the reverse connection (Rhip -> Rshjoulder) as well
visualize_mid_offsets(offsets= sample_output[2], heatmaps=H, to_kp='Rshoulder', from_kp='Rhip', img=img, every=8)

# We can use the heatmaps to compute the skeletons
pred_kp = get_keypoints(H)
pred_skels = group_skeletons(keypoints=pred_kp, mid_offsets=sample_output[2])
pred_skels = [skel for skel in pred_skels if (skel[:,2]>0).sum() > 4]
print ('Number of detected skeletons: {}'.format(len(pred_skels)))

plot_poses(img, pred_skels,save_path='./skel.jpg')

plt.figure()
plt.imshow(apply_mask(img, sample_output[4][:,:,0]>0.5, color=[255,0,0]))

visualize_long_offsets(offsets=sample_output[3], keypoint_id='Rshoulder', seg_mask=sample_output[4], img=img, every=8)
plt.show()