import cv2
import numpy as np
import os
import tensorflow as tf 
from modules.models import RetinaFaceModel
from modules.utils import (pad_input_image, recover_pad_output)
import json
import dlib


def feature_extraction(frame):
    #перенести с сборку модели!
    sp = dlib.shape_predictor('/content/drive/MyDrive/myrsach/face_detection/shape_predictor_68_face_landmarks.dat')
    facerec = dlib.face_recognition_model_v1('/content/drive/MyDrive/myrsach/face_detection/dlib_face_recognition_resnet_model_v1.dat')
    faces=dlib.rectangle(0,0,200,200)
    shape = sp(frame, faces)
    face_descriptor = facerec.compute_face_descriptor(frame, shape)
    descriptor  = np.asarray(face_descriptor)
    return descriptor


def face_recognition(descriptor):
    with open('/content/drive/MyDrive/myrsach/face_detection/students.json') as f:
        known_faces = json.load(f)
    min_dist = 100
    identity = None
    for item in known_faces:
        db_enc=item['descriptor']
        db_enc=np.asarray(db_enc)
        dist = np.linalg.norm(db_enc - descriptor)
        if dist < min_dist:
            min_dist = dist
            name = item['name']
            id_stud=item['id_stud']

    if min_dist > 0.7:
        name="Anon"
        id_stud=0
    else:
        None
    return name,id_stud
        

def save_result(img_path,img,name,id_stud,x1, y1, x2, y2):


    save_img_path = os.path.join('log/' + os.path.basename(img_path))
    name_file=os.path.join('attendance_'+ os.path.splitext(os.path.basename(img_path))[0]+'.json')
    to_json = {'id_stud': id_stud}
    if name!="Anon":
        with open(name_file, 'a') as f:
            f.write(json.dumps(to_json))
            f.write('\n')


    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, name, (x1, y1),cv2.FONT_HERSHEY_DUPLEX, 2, (0,0, 0),3)
    cv2.imwrite(save_img_path, img)


def quality_checking(img):
  tmp_img=img.flatten()  
  brightness=0
  for i in range(1,len(tmp_img)-1,2):
      brightness_pixel=(int(tmp_img[i-1])-int(tmp_img[i+1]))**2
      brightness += int(brightness_pixel)
  brightness=brightness/len(tmp_img)
  if ((brightness<=10) or (brightness>=200)):
    return 0
  elif (((brightness>10) and (brightness<90)) or ((brightness>160) and (brightness<200))):
    return 0.5
  else:
    return 1

def preprocessing_foto(img_raw):
  image=img_raw.copy()
  gaussian_3 = cv2.GaussianBlur(image, (0, 0), 2.0)
  unsharp_image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)
  unsharp_image = cv2.equalizeHist(unsharp_image)
  image = np.float32(unsharp_image)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #перевод в серый
  return image



def preprocessing_face(frame):
    frame = cv2.resize(frame,(200, 200))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #перевод в серый
    #frame = np.asarray(frame)
    #frame =  frame/np.sqrt(np.sum(frame**2))
    #frame=np.array(frame)
    return frame



def face_detection(img,cfg,model):
    # pad input image to avoid unmatched shape problem
    img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

    # run model
    coordinates = model(img[np.newaxis, ...]).numpy()

    # recover padding effect
    coordinates = recover_pad_output(coordinates, pad_params)
    return coordinates

def get_coordinates(img,ann):
    img_height, img_width, _ = img.shape
    x1, y1, x2, y2 = int(ann[0] * img_width), int(ann[1] * img_height), \
                     int(ann[2] * img_width), int(ann[3] * img_height)
    return x1, y1, x2, y2
    


def controller(img_path,cfg,model):
    
    print("[*] Processing on single image {}".format(img_path))
    img_raw = cv2.imread(img_path,0)
    #проверка качества
    if (quality_checking(img_raw))==0:
      return False
    elif (quality_checking(img_raw))==0.5:
      img=preprocessing_foto(img_raw)  # предобработка 
    else:
      None
    outputs=face_detection(img,cfg,model)  #поиск координат лица
  
    #проходим по каждому найденному лицу на фото
    for prior_index in range(len(outputs)):
        x1, y1, x2, y2=get_coordinates(img,outputs[prior_index])
        frame = img_raw[y1:y2, x1:x2]
        frame=preprocessing_face(frame)
        encoding  = feature_extraction(frame)
        name,id_stud=face_recognition(encoding)
        save_result(img_path,img_raw,name,id_stud,x1, y1, x2, y2)
    print(f"[*] save result at {img_path}")
    return True





