from ast import arg
from collections import deque
import sys
from venv import create

sys.path.insert(0, './YOLOX')
import torch
import numpy as np
import cv2
import time
from utils.couting import *
from YOLOX.yolox.data.data_augment import preproc
from YOLOX.yolox.data.data_augment import ValTransform
from YOLOX.yolox.data.datasets import COCO_CLASSES
from YOLOX.yolox.exp.build import get_exp_by_name,get_exp_by_file
from YOLOX.yolox.utils import postprocess
from utils.visualize import vis
from YOLOX.yolox.utils.visualize import plot_tracking, save_bbox_track_id_list
from YOLOX.yolox.tracker.byte_tracker import BYTETracker
from torch2trt import TRTModule

COCO_MEAN = (0.485, 0.456, 0.406)
COCO_STD = (0.229, 0.224, 0.225)

class Detector():
    def __init__(self, model=None, ckpt=None):
        super(Detector, self).__init__()

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        print("device = ",self.device)
        self.cls_names = COCO_CLASSES

        self.preproc = ValTransform(legacy=False)
        self.exp = get_exp_by_name(model)
        
        self.exp.test_size = (640,640) # Need to change it to 416 x 416 for tiny & Nano

        self.test_size = self.exp.test_size  
        self.model = self.exp.get_model()
        self.model.to(self.device)
        self.model.eval()
        checkpoint = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.exp.test_conf = 0.4 # Testing out conf threshold
        self.exp.nmsthre = 0.3 # Testing out nms threshold

        self.trt_file ="/home/jugaad/JL/YOLOX-ByteTrack/YOLOX/YOLOX_outputs/yolox_s/model_trt.pth" # Please change the path according to the YOLOX backbone
        self.model.head.decode_in_inference = False
        self.decoder = self.model.head.decode_outputs

        self.load_modelTRT()

    def load_modelTRT(self):
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(self.trt_file))
        x = torch.ones(1, 3, self.exp.test_size[0], self.exp.test_size[1]).cuda()
        self.model(x)
        self.model = model_trt

    def detect(self, img):
        img_info = {"id": 0}

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img, _ = self.preproc(img, None, self.test_size)

        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.exp.num_classes,  self.exp.test_conf, self.exp.nmsthre,
                class_agnostic=True
            )
            print("Threshold_NMS",self.exp.nmsthre)
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        info = {}
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            info['boxes'], info['scores'], info['class_ids'],info['box_nums']=None,None,None,0
            return img,info

        #output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)

        info['boxes'] = bboxes
        info['scores'] = scores
        info['class_ids'] = cls
        info['box_nums'] = output.shape[0]

        return vis_res,info

class Args():
    def __init__(self) -> None:
        self.track_thresh = 0.4
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10
        self.mot20 = False
        self.tsize = None
        self.name = 'yolox-s' # Please change the path according to the YOLOX backbone
        self.ckpt = '/home/jugaad/JL/YOLOX-ByteTrack/YOLOX/models/yolox_s.pth' # Please change the path according to the YOLOX backbone
        self.exp_file = None
        

if __name__=='__main__':
    args = Args()
    detector = Detector(model=args.name,ckpt=args.ckpt)
    tracker = BYTETracker(args, frame_rate=30)
    exp = get_exp_by_name(args.name)

    #cap = cv2.VideoCapture('/home/jugaad/JL/YOLOX-ByteTrack/YOLOX/video.avi')  # open one video
    #save_path = '/home/jugaad/JL/YOLOX-ByteTrack/YOLOX/result.txt'
    #save_video_path = '/home/jugaad/JL/YOLOX-ByteTrack/YOLOX/result.avi'
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter(save_video_path, fourcc, 20.0, (1920, 1080))

    
    # used to record the time when we processed last frame
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0

    frame_id = 0
    results = []
    saving_path =[]
    fps = 0

    # create filter class
    filter_class = [2]

    # init variable for counting object
    memory = {}
    angle = -1
    in_count = 0
    out_count = 0
    already_counted = deque(maxlen=50)
    frame_id = 0
    while True:
        _, im = cap.read() # read frame from video
        frame_id += 1
        if im is None:
            break
        
        outputs, img_info = detector.detect(im)
       
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, filter_class)
            
           
            online_tlwhs = []
            online_ids = []
            online_scores = []
            bboxes = []
            print("online_targets = ",online_targets)
            for t in online_targets:
                
                attr = dir(t)
                print("Target",attr)
                tlwh = t.tlwh
                tid = t.track_id        
                tlbr = t.tlbr
                print("Location",t.location)
                print("TLBR",tlbr)
                bboxes.append(tlbr)
                online_tlwhs.append(tlwh)
                print("TLWH",tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                saving_path.append(f"{frame_id}, {tid}, {tlbr[0]:.2f}, {tlbr[1]:.2f}, {tlbr[2]:.2f}, {tlbr[3]:.2f}")
                save_bbox_track_id_list(tlbr[0],tlbr[1],tlbr[2],tlbr[3],save_path)
                results.append(f"{frame_id}, {tid}, {tlwh[0]:.2f}, {tlwh[1]:.2f}, {tlwh[2]:.2f}, {tlwh[3]:.2f},{ t.score:.2f}, -1, -1, -1\n")
                print("Frame: {} Track_id: {} ".format(frame_id,tid))
                

                # couting
                # get midpoint from bbox
                midpoint = tlbr_midpoint(tlwh)
                origin_midpoint = (midpoint[0], im.shape[0] - midpoint[1])  # get midpoint respective to bottom-left

           
                    
            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id ,fps=fps, in_count=in_count, out_count=out_count)
            
        else:
            online_im = img_info['raw_img']

        # font which we will be using to display FPS
        font = cv2.FONT_HERSHEY_SIMPLEX
        # time when we finish processing for this frame
        new_frame_time = time.time()
    
        # Calculating the fps
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
    
        online_im = cv2.resize(online_im,(1920,1080))
        cv2.imshow('demo', online_im)	# imshow
        out.write(online_im)

        cv2.waitKey(1)
        if cv2.getWindowProperty('demo', cv2.WND_PROP_AUTOSIZE) < 1:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
