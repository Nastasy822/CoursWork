from absl import logging
import os
import tensorflow as tf  
import yaml

from modules.models import RetinaFaceModel
from modules.utils import (set_memory_growth, load_yaml)
import glob
from tensorflow.python import keras 
from tensorflow.python.keras import backend as K

def create_model_detection(value_iou_th=0.4,value_score_th= 0.5,down_scale_factor=1.0):
    cfg_path='configs/retinaface_mbv2.yaml' #путь до модели
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #????

    logger = tf.get_logger() 
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(cfg_path)

    # define network
    model = RetinaFaceModel(cfg, training=False, iou_th=value_iou_th,score_th=value_score_th)

    # load checkpoint
    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(model=model)
  
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()
    return cfg,model
 
  
#подготовка базы данных
def prepare_database(FRmodel):
    database = {}
    for file in glob.glob("/content/drive/MyDrive/myrsach/face_detection/images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_path_to_encoding(file, FRmodel)
    return database


