import cv2
import sys 

sys.path.insert(0,'/home/tejaskumar/College_Project/Instance_project/Mtech_thesis_project/mask_rcnn/libraries')

import mrcnn.model as modellib
from mrcnn.config import Config

cam = cv2.VideoCapture(0) #0=front-cam, 1=back-cam
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1300)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1500)

MODEL_DIR = "/home/tejaskumar/College_Project/Instance_project/mask_rcnn_shapes_0300.h5"

image_size = 64
rpn_anchor_template = (1, 2, 4, 8, 16) # anchor sizes in pixels
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

    STEPS_PER_EPOCH = 10

    VALIDATION_STEPS = STEPS_PER_EPOCH / 20
    
config = ShapesConfig()
config.display()

while True:
    ## read frames
    ret, img = cam.read()
    class InferenceConfig(ShapesConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    #Mask RCNN
    model = modellib.MaskRCNN(mode="inference", 
                            config=inference_config,
                            model_dir=MODEL_DIR)
    cv2.imshow("", img)
    ## press q or Esc to quit    
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break

## close camera
cam.release()
cv2.destroyAllWindows()