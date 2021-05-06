from densenet import *
import matplotlib.pyplot as plt

from Config import Config
from data import CloudDataSet
import numpy as np


from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#设置gpu内存动态增长
gpu_options = tf.GPUOptions(allow_growth=True)
set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
#Directory to save logs and trained model
MODEL_DIR='./tmp'


#Configuration for training on the Cloud dataset.
config=Config()

#Validation dataset
data_val = CloudDataSet()
data_val.load_Mete('val')
data_val.prepare()

model=DeFCN(mode="inference",config=config,model_dir=MODEL_DIR)
model.load_weights(model.find_last(),by_name=True)
#model.load_weights('./tmp/cloud20200819T1046/DeFCN_cloud_0060.h5',by_name=True)
aTP=0
aFP=0
aTN=0
aFN=0

def scale_img(img):
    w, h, d = img.shape
    img = np.reshape(img, [w * h, d]).astype(np.float64)
    #mins = np.percentile(img, 2, axis=0)
    #maxs = np.percentile(img, 98, axis=0) - mins
    a = np.min(img, axis = 0)
    a = np.min(a, axis = 0)
    b = np.max(img, axis = 0)
    b = np.max(b, axis = 0)
    img = (img - a) / b
    img = np.reshape(img, [w, h, d])
    return img.clip(0, 1)


for id in data_val.image_ids:
    image = data_val.load_image(id)
    gt_map = data_val.load_mask(id)

    pre=model.detect(image)
    pre.astype(np.int)
    pre=np.reshape(pre,(224,224))

    FP=pre-gt_map

    FP[np.where(FP==-1)]=0
    TP=pre-FP
    FN=gt_map-TP
    TN=1-FN-FP-TP

    
    aTP=np.sum(TP)+aTP
    aFP=np.sum(FP)+aFP
    aTN=np.sum(TN)+aTN
    aFN=np.sum(FN)+aFN
    # P=np.sum(TP)/(np.sum(TP)+np.sum(FP))
    # R=np.sum(TP)/(np.sum(TP)+np.sum(FN))
    # F1=2*P*R/(P+R)
    image =np.apply_along_axis(lambda x:x/x.max(),2,image)
    image = scale_img(image)
    #plt.subplot(131)
    #plt.imshow(image[:,:,0:3])
    #plt.imshow((image* 255).astype(np.uint8))
    #plt.subplot(132)
    #plt.imshow((gt_map* 255).astype(np.uint8))
    #plt.subplot(133)
    #plt.imshow((pre* 255).astype(np.uint8))
    #plt.show()

    print(id)
P=np.sum(aTP)/(np.sum(aTP)+np.sum(aFP))
R=np.sum(aTP)/(np.sum(aTP)+np.sum(aFN))
F1=2*P*R/(P+R)
MRate=(aFP+aFN)/(aTP+aTN+aFP+aFN)
mIoU=np.sum(aTP)/(np.sum(aTP)+np.sum(aFP)+np.sum(aFN))
ACC=(np.sum(aTP)+np.sum(aTN))/(np.sum(aTP)+np.sum(aFP)+np.sum(aFN)+np.sum(aTN))

#plt.subplot(131)
#plt.imshow(image[:,:,0:3])
    #plt.imshow((image* 255).astype(np.uint8))
#plt.subplot(132)
#plt.imshow((gt_map* 255).astype(np.uint8))
#plt.subplot(133)
#plt.imshow((pre* 255).astype(np.uint8))
#plt.show()

print(TP,FN,TN)
print(P,R,F1,MRate,mIoU,ACC)
#loading image for  test.
image=data_val.load_image(54)
gt_map=data_val.load_mask(54)
#feed the image to model
pre=model.detect(image)
pre.astype(np.int)
pre=np.reshape(pre,(224,224))
#

