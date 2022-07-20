
import os
import torch

from detector import Detector

# Function called by camnetGPU and camnetCPU initilizations
def load_detector(centertrackdir='/home/JL/YOLOX-BYTETRACK', img_width=720, img_height=616):
    args = [#'task=tracking',
           #'--arch=hardnet_85',
           '--load_model={}'.format(os.path.join(centertrackdir, 'models', 'coco_tracking.pth')),
           '--input_w={}'.format(img_width),
           '--input_h={}'.format(img_height)]

    opt = opts().init(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    print(f"opt.task: {opt.task}")
    detector = Detector(opt)

    return detector

# Function called by camnetGPU.runTarget
def run_detector_gpu(detector, image):
  """
  Function to perform object detection on the GPU
  """

  with torch.no_grad():
    ret = detector.run(image)

  # Return the 'results' value, which is a dictionary of the form 
  # {'score': x.xx, 
  #  'class': int, 
  #  'ct': array([xx, yy], dtype=float32), 
  #  'tracking': array([xx, yy], dtype=float32), 
  #  'bbox': array([x1, y1, x2, y2], dtype=float32), 
  #  'tracking_id': int, 
  #  'age': int (likely 1), 
  #  'active': int}
  return ret['results']
