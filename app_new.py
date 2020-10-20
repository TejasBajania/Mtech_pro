from flask import Flask, render_template, url_for, request, redirect, jsonify
from flask_bootstrap import Bootstrap
import uuid
import os
import requests
import sys
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import numpy as np
from pygame import mixer
HOME_DIR = os.getcwd()
LIB_DIR = os.path.join(HOME_DIR, "mask_rcnn\libraries")
DATA_DIR = os.path.join(HOME_DIR, "data/shapes")
MODEL_DIR = os.path.join(DATA_DIR, "logs")
#os.chdir(LIB_DIR)
sys.path.insert(0, LIB_DIR)
#print(os.getcwd())
# print(os.listdir('./libraries/'))

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.keras import backend as k
import re
from gtts import gTTS 
from collections import Counter

from mrcnn.config import Config
import mrcnn.utils as utils
import mrcnn.model as modellib
import mrcnn.visualize as visualize
from mrcnn.model import log
import mcoco.coco as coco
import skimage
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
Bootstrap(app)

#PIZZA CODE

dataset_train = coco.CocoDataset()
dataset_train.load_coco(DATA_DIR, subset="shapes_train", year="2018")
dataset_train.prepare()

dataset_validate = coco.CocoDataset()
dataset_validate.load_coco(DATA_DIR, subset="shapes_validate", year="2018")
dataset_validate.prepare()

dataset_test = coco.CocoDataset()
dataset_test.load_coco(DATA_DIR, subset="shapes_test", year="2018")
dataset_test.prepare()

image_size = 640
rpn_anchor_template = ( 2, 4, 8, 16, 32) # anchor sizes in pixels
rpn_anchor_scales = tuple(i * (image_size // 16) for i in rpn_anchor_template)

class ShapesConfig(Config):
    """Configuration for training on the shapes dataset.
    """
    NAME = "shapes"

    # Train on 1 GPU and 2 images per GPU. Put multiple images on each
    # GPU if the images are small. Batch size is 2 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 10  # background + 3 shapes (triangles, circles, and squares)

    # Use smaller images for faster training. 
    IMAGE_MAX_DIM = image_size
    IMAGE_MIN_DIM = image_size
    
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = rpn_anchor_scales

    # Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    STEPS_PER_EPOCH = 50

    VALIDATION_STEPS = STEPS_PER_EPOCH / 20
    
config = ShapesConfig()
config.display()


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
inference_config = InferenceConfig()

# MODEL_DIR = "./weights/mask_rcnn_coco.h5"
    # Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=MODEL_DIR)
    
ROOT = os.getcwd()
    
model_path = model.find_last()[1]
    # Load trained weights
def load_model():
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    global graph
    graph = tf.get_default_graph()
    
load_model()
# detected
def getImage(image):
    data = []
    image = skimage.io.imread(image)
    with graph.as_default():
        results = model.detect([image], verbose=1)
        r = results[0]
       
       
        name = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, 
                                          r['scores'])
        # skimage.io.imsave('out.png',name)
        # image.savefig('out.png')
        count_10 = 0
        count_20_new = 0
        count_50_new = 0 
        count_100_new = 0 
        count_200_new = 0
        count_500_new = 0
        count_2000_new = 0
        # '10_old','10_new ','20_old','50_old','50_new ','100_old','100_new ','200_new','500_new ','2000_new']
        for i in r['class_ids']:
            print(class_names[i])
            if class_names[i] == '10_old' or class_names[i] == '10_new ':
                count_10+=1
            elif class_names[i] == '20_old':
                count_20_new+=1
            elif class_names[i] == '50_old' or class_names[i] == '50_new ':
                count_50_new+=1
            elif class_names[i] == '100_old' or class_names[i] == '100_new ':
                count_100_new+=1
            elif class_names[i] == '200_new':
                count_200_new+=1
            elif class_names[i] == '500_new ':
                count_500_new+=1
            elif class_names[i] == '2000_new':
                count_2000_new+=1
        data.append(count_10)
        data.append(count_20_new)
        data.append(count_50_new)
        data.append(count_100_new)
        data.append(count_200_new)
        data.append(count_500_new)
        data.append(count_2000_new)
        detected = [] 
        for cl in r['class_ids']:
            detected.append(dataset_test.class_names[cl])
        detect_dict = {}
        for i in range(1,len(dataset_test.class_names)):
           detect_dict.update({dataset_test.class_names[i]:int(re.search(r'\d+', dataset_test.class_names[i]).group())})
        col = []
        fin_detec = []
        detect_list = []
        for dec in detected:
            col.append(" {}".format(detect_dict[dec]))
        for i in range(len(col)):
            fin_detec.append(str(col[i])+" Rupees ")
        a = dict(Counter(fin_detec))
        for k,v in a.items():
            if v == 1:
                detect_list.append("{} note of {}".format(v,k))
            else:
                detect_list.append("{} notes of {}".format(v,k))
        if detect_list:
            mytext = " The image contains"+str(detect_list)
        else:
            mytext = "The image contains no currency"
        language = 'en'
        myobj = gTTS(text=mytext, lang=language, slow=False) 
        filename = str(uuid.uuid4()) + ".mp3"
        myobj.save("./audio_out/{}".format(filename)) 
        mixer.init()
        mixer.music.load("./audio_out/{}".format(filename))
        mixer.music.play()
        
        return data,'1.png'

class_names = ['BG','10_old','10_new ','20_old','50_old','50_new ','100_old','100_new ','200_new','500_new ','2000_new']

#HOME API
@app.route('/')
def index():
    return render_template('index.html')

#PIZZA API
@app.route('/upload', methods=["GET", "POST"])
def upload():
    if request.method == "POST":

        if request.files:
            image = request.files["image"]
            if(os.path.isfile(HOME_DIR + "./static/1.png")):
                os.remove(HOME_DIR + "./static/1.png")
            res = []
            res,file = getImage(image)
        
            return render_template("view.html",image_name = file, data=res)
    return render_template('upload.html')

#PIZZA API
# @app.route('/check_nodes', methods=["GET", "POST"])
# def check_nodes():
#     if request.method == "POST":

#         if request.files:
#             # image = request.files["image"]
#             image = 
#             if(os.path.isfile(HOME_DIR + "./static/1.png")):
#                 os.remove(HOME_DIR + "./static/1.png")
#             res = []
#             res,file = getImage(image)
        
#             return render_template("view.html",image_name = file, data=res)
#     return render_template('upload.html')
# #MAIN
if __name__ == '__main__':
    app.run(debug = True, use_reloader = False) 
