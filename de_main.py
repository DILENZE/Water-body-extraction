from densenet import *

import keras.models as KM
from Config import Config
from data import CloudDataSet
from keras.layers import Input


from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#设置gpu内存动态增长
gpu_options = tf.GPUOptions(allow_growth=True)
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
#Directory to save logs and trained model
MODEL_DIR='./tmp'


#Configuration for training on the Cloud dataset.
config = Config()

#Training dataset
data_train = CloudDataSet()
data_train.load_Mete('training')
data_train.prepare()

#Validation dataset
data_val = CloudDataSet()
data_val.load_Mete('val')
data_val.prepare()

model=DeFCN(mode="training",config=config,model_dir=MODEL_DIR)
#print(model.find_last())
#model.load_weights('./tmp/cloud20200624T0937/DeFCN_cloud_0013.h5',by_name=True)
model.train(data_train,data_val,learning_rate=config.LEARNING_RATE, epochs=30)

model.train(data_train,data_val,learning_rate=config.LEARNING_RATE/10, epochs=50)

model.train(data_train,data_val,learning_rate=config.LEARNING_RATE/100, epochs=60)
