from densenet import *
import matplotlib.pyplot as plt
from skimage import transform,io,data
from Config import Config
from data import CloudDataSet
import numpy as np
import os
from keras.backend.tensorflow_backend import set_session

from keras.preprocessing import image


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#设置gpu内存动态增长
gpu_options = tf.GPUOptions(allow_growth=True)
set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
#Directory to save logs and trained model
MODEL_DIR='./tmp'
#Configuration for training on the Cloud dataset.
config=Config()

#Validation dataset
# sourceDir = "/home/wumengchen/Seg_Decloud/cloudData/c1"
# for file in os.listdir(sourceDir):
#         # 图片路径
#         imgPath = os.path.join(sourceDir, file)
#         # 读取图片
#         x = io.imread(os.path.expanduser(imgPath))
#dirlist = []
# def scale_img(img):
#     w, h, d = img.shape
#     img = np.reshape(img, [w * h, d]).astype(np.float64)
#     mins = np.percentile(img, 2, axis=0)

#     maxs = np.percentile(img, 98, axis=0) - mins
#     img = (img - mins[None, :]) / maxs[None, :]
#     img = np.reshape(img, [w, h, d])
#     return img.clip(0, 1)


model=DeFCN(mode="inference",config=config,model_dir=MODEL_DIR)
model.load_weights(model.find_last(),by_name=True)
#model.load_weights('./tmp/cloud20200819T1046/DeFCN_cloud_0060.h5',by_name=True)

path = "/share/home/dongzhen/DeSegCloud/cloudData/cd/180514"
for dirpath, dirname, filename in os.walk(path):
    for i in filename:
        #dirlist(os.path.join(dirpath,i))
        image = io.imread(os.path.join(dirpath,i))
        image = np.reshape(image,(-1,224,224,1))

        pre=model.detect(image)
        pre.astype(np.int)
        pre=np.reshape(pre,(224,224))


        image =np.apply_along_axis(lambda x:x/x.max(),2,image)
        
        #plt.subplot(131)
        #plt.imshow(image[-1,:,:,0:3])

        #plt.subplot(132)   
        #plt.imshow((pre* 255).astype(np.uint8))
        #plt.show()

        io.imsave(os.path.join('/share/home/dongzhen/DeSegCloud/cloudData/cd/1805141',i),pre)



#image = io.imread('/home/wumengchen/Seg_Decloud/cloudData/images224/00001.tif')
##-------------------------------------------------------------------------------
##-------------------------------------------------------------------------------
# model=DeFCN(mode="inference",config=config,model_dir=MODEL_DIR)
# model.load_weights(model.find_last(),by_name=True)

# pre=model.detect(x)
# pre.astype(np.int)
# pre=np.reshape(pre,(224,224))




# image =np.apply_along_axis(lambda x:x/x.max(),2,x)
# image = scale_img(image)
# plt.subplot(131)
# plt.imshow(image[:,:,0:3])

# plt.subplot(132)   
# plt.imshow((pre* 255).astype(np.uint8))
# plt.show()

# io.imsave('/home/wumengchen/Seg_Decloud/cloudData/test/x.tif',pre*255)


#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

#print(pre)
# for id in data_val.image_ids:
#     image = data_val.load_image(id)
#     gt_map = data_val.load_mask(id)

#     pre=model.detect(x)
#     pre.astype(np.int)
#     pre=np.reshape(pre,(224,224))

#     FP=pre-gt_map

#     FP[np.where(FP==-1)]=0
#     TP=pre-FP
#     FN=gt_map-TP
#     TN=1-FN-FP-TP




    # aTP=np.sum(TP)+aTP
    # aFP=np.sum(FP)+aFP
    # aTN=np.sum(TN)+aTN
    # aFN=np.sum(FN)+aFN
    # P=np.sum(TP)/(np.sum(TP)+np.sum(FP))
    # R=np.sum(TP)/(np.sum(TP)+np.sum(FN))
    # F1=2*P*R/(P+R)
#     image =np.apply_along_axis(lambda x:x/x.max(),2,image)
#     image = scale_img(image)
#     plt.subplot(131)
#     plt.imshow(image[:,:,0:3])
#     #plt.imshow((image* 255).astype(np.uint8))
#     plt.subplot(132)
#     plt.imshow((gt_map* 255).astype(np.uint8))
#     plt.subplot(133)
#     plt.imshow((pre* 255).astype(np.uint8))
#     plt.show()

#     print(image_ids)
#     print(id)
# P=np.sum(aTP)/(np.sum(aTP)+np.sum(aFP))
# R=np.sum(aTP)/(np.sum(aTP)+np.sum(aFN))
# F1=2*P*R/(P+R)
# MRate=(aFP+aFN)/(aTP+aTN+aFP+aFN)
# print(P,R,F1,MRate)
# #loading image for  test.
# image=data_val.load_image(54)
# gt_map=data_val.load_mask(54)
# #feed the image to model
# pre=model.detect(image)
# pre.astype(np.int)
# pre=np.reshape(pre,(224,224))
# #

