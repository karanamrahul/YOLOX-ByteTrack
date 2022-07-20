# Developed by: Jugaad Labs

import sys
import os
from torch import det
import yaml
import json
import math

import cv2
from queue import Queue
from threading import Thread
import numpy as np
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from iris_msgs.msg import ObjectDetection, ObjectDetectionArray

# Get Camera Tools path
current_directory = os.path.dirname(os.path.realpath(__file__))
camera_tools_path = os.path.join(current_directory, '../../mirroreye_camera_tools/src')
if camera_tools_path not in sys.path:
    sys.path.append(camera_tools_path)

from calibration_utilities import extract_calibration_from_version
from projection_utilities import WorldProjector, DistortionCorrector

from cohen_sutherland_bbox_clip import cohenSutherlandBBoxClip


class Ob(object):
    """
    Utility class can be used for just an empty object, or with attributes initialized by a dictionary
    """
    def __init__(self, dict = None):
        if dict is not None:
            for k, v in dict.items():
                setattr(self, k, v)


def load_configs(config_path):
    """
    Function to load a YAML file containing configuration information
    """
    print('loading', config_path)
    with open(config_path, 'r') as fp:
        config_dict = yaml.load(fp, Loader=yaml.FullLoader)  
    configs = Ob(config_dict)
    return configs

current_directory = os.path.dirname(os.path.realpath(__file__))
localPaths = load_configs(os.path.join(current_directory, '../../local_paths/LocalPathsConfig.yaml'))

# Get CenterTrack path
if localPaths.bytetrackPath not in sys.path:
    sys.path.append(localPaths.bytetrackPath)

from detector import * # This is the class that actually does the detection

######## YOLOX-L #######
backbone_yolo = 'yolox-l'
chk_point_yolo = '/home/jugaad/JL/YOLOX-ByteTrack/YOLOX/models/yolox_l.pth'

######## YOLOX-M #######
#backbone_yolo = 'yolox-m'
#chk_point_yolo = '/home/jugaad/JL/YOLOX-ByteTrack/YOLOX/models/yolox_m,.pth'

######## YOLOX-S #######
# backbone_yolo = 'yolox-s'
# chk_point_yolo = '/home/jugaad/JL/YOLOX-ByteTrack/YOLOX/models/yolox_s.pth'

# ######## YOLOX-TINY #######
# backbone_yolo = 'yolox-tiny'
# chk_point_yolo = '/home/jugaad/JL/YOLOX-ByteTrack/YOLOX/models/yolox_tiny.pth'

# ######## YOLOX-NANO #######
# backbone_yolo = 'yolox-nano'
# chk_point_yolo = '/home/jugaad/JL/YOLOX-ByteTrack/YOLOX/models/yolox_nano.pth'

print('sys.path:')
print(sys.path)

class Args():
    def __init__(self) -> None:
        self.track_thresh = 0.4
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10
        self.mot20 = False
        self.tsize = None
        self.name = backbone_yolo
        self.ckpt = chk_point_yolo
        self.exp_file = None
        
class CamnetGPU(object):
    def __init__(self, node):
        self.args = Args()
        self.node = node
        self.queue = Queue()
        self.outqueue = self.node.camnet_cpu.queue
        self.epoch_length = 1.0
        self.epoch_start = 0.0
        self.cam_count = 0
        self.drop_count = 0
        self.left_count = 0
        self.right_count = 0
        self.detector = Detector(model=self.args.name,ckpt=self.args.ckpt)
        self.tracker = BYTETracker(self.args,frame_rate=30)
        self.exp = get_exp_by_name(self.args.name)
        self.filter_class = [2]
        
    def runTarget(self):
        print('Starting Detection GPU')
        
        while not rospy.is_shutdown():
            qs = self.queue.qsize() 
            if qs > 3:
                print('clearing queue gpu',qs)
                while self.queue.qsize() > 1:
                    image_msg = self.queue.get()
            image_msg = self.queue.get()

            # Extract data from Image() message
            image_metadata, image_array = self.extract_data_from_image_message(image_msg)

            self.cam_count += 1
            if 'left' in image_metadata['camera']:
                self.left_count += 1
            elif 'right' in image_metadata['camera']:
                self.right_count += 1
            stamp = rospy.Time.now().to_sec()
            span = stamp - self.epoch_start
            if span > self.epoch_length:
                qs = self.queue.qsize()
                lc = self.left_count
                rc = self.right_count
                rate = (lc + rc) / span
                fmt = '\nGPU Processed {} left and {} right in {:.1f}sec {:.1f}fps qsize={}\n'
                print("FPS:",rate)
                print(fmt.format(lc,rc,span,rate,qs))
                self.epoch_start = stamp
                self.left_count = 0
                self.right_count = 0
            

            # Split detection by camera side
            if 'left' in image_metadata['camera']:
                outputs,img_info = self.detector.detect(image_array)
                online_targets = self.tracker.update(outputs[0], [img_info['height'], img_info['width']], self.exp.test_size, self.filter_class)
                
                detections_list = self.convert_online_targets_to_detections(online_targets,outputs,img_info,'left')
                
            elif 'right' in image_metadata['camera']:
                outputs,img_info = self.detector.detect(image_array)
                online_targets = self.tracker.update(outputs[0], [img_info['height'], img_info['width']], self.exp.test_size, self.filter_class)
                detections_list = self.convert_online_targets_to_detections(online_targets,outputs,img_info,'right')
            # Pass detection information to CPU
            if not self.node.configs.debugDetections:
                # Pass only image metadata and detections to the CPU queue (we don't need to handle the full image array anymore)
                self.outqueue.put((image_metadata, detections_list))
            else:
                # Include image in queue for visualizations
                self.outqueue.put((image_metadata, image_array, detections_list))
                
            print("Detection List:",detections_list)
            
    def extract_data_from_image_message(self, image_msg):
        """
        Function to extract needed metadata from an image message and separately create numpy array from actual image data
        """
        img_metadata = {}
        img_metadata['header.stamp'] = image_msg.header.stamp
        img_metadata['header.frame_id'] = image_msg.header.frame_id
        if 'left_wide' in img_metadata['header.frame_id']:
            img_metadata['camera'] = 'left_wide'
        elif 'right_wide' in img_metadata['header.frame_id']:
            img_metadata['camera'] = 'right_wide'
        img_metadata['height'] = image_msg.height
        img_metadata['width'] = image_msg.width

        img_array = np.fromstring(image_msg.data, dtype=np.uint8).reshape(image_msg.height, image_msg.width, 3)

        return img_metadata, img_array



    def convert_online_targets_to_detections(self,online_targets,outputs,img_info,camera):
        """ Need to convert the ByteTrack online targets to the format that the detector expects 
        CenterTrack uses a different format than the detector(ByteTrack)
         {'score': x.xx, 
         'class': int, 
         'ct': array([xx, yy], dtype=float32), 
         'tracking': array([xx, yy], dtype=float32), 
         'bbox': array([x1, y1, x2, y2], dtype=float32), 
         'tracking_id': int, 
         'age': int (likely 1), 
         'active': int} """
         
         
        detections_list = []
        for target in online_targets:
            detection = {}
            detection['score'] = target.score
            detection['class'] = 3
            detection['bbox'] = target.tlbr
            detection['tracking_id'] = target.track_id
            detection['camera'] = camera
            ret = np.asarray(target.tlwh).copy()
            ret[:2] += ret[2:] / 2
            detection['ct'] = ret[:2]
            detection['tracking'] = ret[:2]
            detection['age'] = 1
            detection['active'] = 1
            detections_list.append(detection)
        return detections_list

class CamnetCPU(object):
    def __init__(self, node):
        self.node = node

        # self.roiPub = rospy.Publisher("/camnet/roi_array", ROIArray, queue_size=100)
        # self.markerPub = rospy.Publisher("/camnet/camMarkers", MarkerArray, queue_size=100)
        # self.trackPub = rospy.Publisher("/camnet/camTracks", MarkerArray, queue_size=100)
        # self.left_box_ims_pub = rospy.Publisher("/camnet/left_box_ims", Image, queue_size=100)
        # self.right_box_ims_pub = rospy.Publisher("/camnet/right_box_ims", Image, queue_size=100)
        # self.camMarkerIdCounter = 500000
        # self.trackMarkerIdCounter = 500000 ### CENTERTRACK MOD - NEW LINE
        self.queue = Queue()
        self.epoch_length = 1.0
        self.epoch_start = 0.0
        self.cam_count = 0
        self.drop_count = 0
        self.left_count = 0
        self.right_count = 0
        # self.transforms = loadSavedTransforms(options.transformsPath)
        # self.caminfos = load_camera_info_yamls(options.calibrationsPath)
        # self.detector = load_detector()

        # Initialize needed items for debug visualization
        if self.node.configs.debugDetections:
            #----------------------#
            #     Camera Tools     #
            #----------------------#
            # Select version and extract calibration:
            # Original for older bag files, else choose a newer version
            self.calibration_version = 'v7'
            self.calibration_dict = extract_calibration_from_version(self.calibration_version)
            self.world_projector = {'left_wide': WorldProjector(self.calibration_dict['left_wide']), 
                                    'right_wide': WorldProjector(self.calibration_dict['right_wide'])}
            self.distortion_corrector = {'left_wide': DistortionCorrector(self.calibration_dict['left_wide']), 
                                        'right_wide': DistortionCorrector(self.calibration_dict['right_wide'])}
            

            self.detection_img = np.zeros((self.node.configs.imgH, 2*self.node.configs.imgW, 3), dtype=np.uint8)
            self.centertrack_dict = {'left': {}, 'right': {}}

        # Define publisher for DetectionROIArray message
        self.queue_size = self.node.rosTopics.objectDetectionQueueSize
        self.ObjectDetectionArrayPub = rospy.Publisher(self.node.rosTopics.objectDetectionArrayTopic, 
                                                       ObjectDetectionArray, 
                                                       queue_size=self.queue_size)
        

    def runTarget(self):
        print('Starting Detection CPU')

        while not rospy.is_shutdown():
            qs = self.queue.qsize() 
            if qs > 3:
                print('clearing cpu queue',qs)
                while self.queue.qsize() > 1:
                    self.queue.get()
            
            # Get detection information from GPU
            if not self.node.configs.debugDetections:
                image_metadata, detections_list = self.queue.get()
            else:
                image_metadata, image_array, detections_list = self.queue.get()

            self.cam_count += 1
            if 'left' in image_metadata['camera']:
                self.left_count += 1
            elif 'right' in image_metadata['camera']:
                self.right_count += 1
            stamp = rospy.Time.now().to_sec()
            span = stamp - self.epoch_start
            if span > self.epoch_length:
                qs = self.queue.qsize()
                lc = self.left_count
                rc = self.right_count
                rate = (lc + rc) / span
                fmt = '\nCPU Processed {} left and {} right in {:.1f}sec {:.1f}fps qsize={}\n'
                print(fmt.format(lc,rc,span,rate,qs))
                self.epoch_start = stamp
                self.left_count = 0
                self.right_count = 0
        
            # Filter CenterTrack output passed from GPU
            filtered_detections = self.filter_detections(image_metadata, detections_list)

            # Build ObjectDetectionArray message from the filtered detections
            detection_array_msg = self.build_ObjectDetectionArray_msg(image_metadata, filtered_detections)

            # Publish ObjectDetectionArray message
            self.ObjectDetectionArrayPub.publish(detection_array_msg)
            print(detection_array_msg)
            # Make debug visualization if flag set
            if self.node.configs.debugDetections:
                # Update the CenterTrack dictionary
                self.update_centertrack_dict(image_metadata, filtered_detections)
                # Update the debug visualization
                self.visualize_detections(self.detection_img, image_metadata, image_array)
                # Show the visualization
                cv2.imshow('Detection Visualization', self.detection_img)
                cv2.waitKey(1)

            



    def filter_detections(self, img_metadata, detections_list):
        """ 
        Function to filter the output of CenterTrack from the GPU

        Results from CenterTrack are a list with elements of the form
        {'score': x.xx, 
         'class': int, 
         'ct': array([xx, yy], dtype=float32), 
         'tracking': array([xx, yy], dtype=float32), 
         'bbox': array([x1, y1, x2, y2], dtype=float32), 
         'tracking_id': int, 
         'age': int (likely 1), 
         'active': int} 
        
        Unlike CenterNet, not all classes are represented, only actual detections found
        Note can't use break in these filters since not all classes are included, and class order is not enforced

        We also need to add the camera side to the detection dictionary
        """

        # Get image shape
        h,w, = img_metadata['height'], img_metadata['width']

        filtered_detections = []

        # Loop through [{det1, det2, ...}] list of dictionaries
        for detection in detections_list:
            
            # Unpack detection dictionary for info used in filtering:
            cls_conf = detection['score']
            cls = detection['class']
            bbox = detection['bbox']

            # Filter on classes that are included
            if cls not in self.node.configs.classesIncluded.keys():
                continue 

            # Perform confidence threshold filtering
            # Check if using a uniform confidence threshold
            if self.node.configs.useUniformThreshold == True:
                if cls_conf < self.node.configs.uniformConfidenceThreshold:
                    # Discard detection
                    continue

            # Else use class-specific confidence threshold
            else:
                if cls_conf < self.node.configs.classesIncluded[cls]['threshold']:
                    # Discard detection
                    continue
            
            # If reaching here, we have a confident detection of a relevant class

            # Unpack bbox corners
            x1, y1, x2, y2 = bbox.tolist()

            # Filter for eliminating ego vehicle detections
            # Parameters for defining corner area
            tractor_corner_x = self.node.configs.tractorCornerX
            tractor_corner_y = self.node.configs.tractorCornerY
            # TODO : implement trailer detections filter
            # trailer_corner_x = self.node.configs.trailerCornerX 
            # trailer_corner_y = self.node.configs.trailerCornerY

            
            if 'left' in img_metadata['camera']:
                # Check if top left of bbox is within corner distance of top left of image
                if (x1 < tractor_corner_x and y1 < tractor_corner_y):
                    # Discard detection
                    continue
            elif 'right' in img_metadata['camera']:
                # Check if top right of bbox is within corner distance of top right of image
                if (w - x2 < tractor_corner_x and y1 < tractor_corner_y):
                    # Discard detection
                    continue
            
            # TODO : implement trailer detections filter next
            
            # IoU filter
            iou_continue = False
            for i, filtered_detection in enumerate(filtered_detections):
                # if detection['class'] == filtered_detection['class']:
                #     continue
                iou = self.calcIOU(detection['bbox'], filtered_detection['bbox'])
                if iou > self.node.configs.IoUThreshold:
                    # Once in here, we are either substituting in, or not add
                    iou_continue = True
                    if detection['score'] > filtered_detection['score']:
                        filtered_detections[i] = detection
                        filtered_detections[i]['tracking_id'] = filtered_detection['tracking_id']
            
            if iou_continue:
                continue

            # If we reach here, we passed all filters for this detection

            # Modify tracking_id to reflect which camera side it came from, 
            # if first time encountering this tracking id, its type will be integer
            if isinstance(detection['tracking_id'], int):
                if 'left' in img_metadata['camera']:
                    detection['tracking_id'] = 'L_'+str(detection['tracking_id'])
                elif 'right' in img_metadata['camera']:
                    detection['tracking_id'] = 'R_'+str(detection['tracking_id'])

            filtered_detections.append(detection)
        
        # Return filtered detections for continued processing
        return filtered_detections
    

    def calcIOU(self, bbox1, bbox2):
        # Coordinates of intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        # Areas of boxes
        intersect_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
        bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

        iou = intersect_area / (bbox1_area + bbox2_area - intersect_area)
        return iou


    def build_ObjectDetectionArray_msg(self, img_metadata, detections_list):
        """
        Function to construct an ObjectDetectionArray message containing ObjectDetection messages for each filtered detection.
        
        ObjectDetection.msg contains:
            float64		score
            int32		classID
            string		trackingID
            float64[]	bbox

        ObjectDetectionArray.msg contains:
            Header header
            ObjectDetection[] detection_msgs
        
        The Header of ObjectDetectionArray message will be the same as the Header of the original image message which contains all the synchronization data
        """
        
        detection_msg_list = []
        # Loop over detections and construct a ObjectDetection class for each
        for detection in detections_list:
            detection_msg = ObjectDetection()
            detection_msg.score = detection['score']
            detection_msg.class_id = detection['class']
            detection_msg.tracking_id = detection['tracking_id']
            detection_msg.bbox = detection['bbox'].tolist() # Need to convert from numpy array to python list

            # Add built message to list
            detection_msg_list.append(detection_msg)
        
        # Build ObjectDetectionArray message
        detection_array_msg = ObjectDetectionArray()
        # Use original header data for new message
        detection_array_msg.header.stamp = img_metadata['header.stamp']
        detection_array_msg.header.frame_id = img_metadata['header.frame_id']
        detection_array_msg.detection_msgs = detection_msg_list

        print(f"Ran Object Detection on image frame_id: {img_metadata['header.frame_id']}")

        # Return built ObjectDetectionArray message
        return detection_array_msg
    
    def update_centertrack_dict(self, img_metadata, detections):
        """
        Method to track detections for debug purposes
        Results from CenterTrack are a list with elements of the form
        {'score': x.xx, 
         'class': int, 
         'ct': array([xx, yy], dtype=float32), 
         'tracking': array([xx, yy], dtype=float32), 
         'bbox': array([x1, y1, x2, y2], dtype=float32), 
         'tracking_id': int, 
         'age': int (likely 1), 
         'active': int}
        """
        if img_metadata['camera'] == 'left_wide' :
            side = 'left'
        elif img_metadata['camera'] == 'right_wide':
            side = 'right'

        # Initiate set of tracking_ids detected this time
        detected_ids = set()

        # Loop through detection list
        for detection in detections:
            # Get id
            track_id = detection['tracking_id'] 
            # Add id to set of tracking ids detected this time
            detected_ids.add(track_id)      
            # Check if id has been detected already
            if track_id not in self.centertrack_dict[side].keys():
                # Initiate new entry
                self.centertrack_dict[side][track_id] = {'class': detection['class'],
                                                         'scores': [detection['score']],
                                                         'bboxes' : [detection['bbox']],
                                                         'detection_centers': [detection['ct']],
                                                         'tracking_vectors': [detection['tracking']],
                                                         'bbox_pts': [], # Initiate empty list to populate in next step
                                                         'p_worlds': []} # Initiate empty list to populate in next step
            else:
                # Update existing record
                self.centertrack_dict[side][track_id]['scores'].append(detection['score'])
                self.centertrack_dict[side][track_id]['bboxes'].append(detection['bbox'])
                self.centertrack_dict[side][track_id]['detection_centers'].append(detection['ct'])
                self.centertrack_dict[side][track_id]['tracking_vectors'].append(detection['tracking'])
            
            # Do world projection to determine current p_world value
            # Step 1: Determine single point from latest bounding box. Delegate to a separate method for later sophistication improvement
            bbox_pt = self.transform_bbox_to_point(side, track_id)

            # Project the single bbox_pt pixel coordinates out into the world coordinate system using the corresponding WorldProjector
            p_world = self.project_bbox_pt_to_p_world(img_metadata['camera'], bbox_pt, as_list=True)

            # Append both points to track
            self.centertrack_dict[side][track_id]['bbox_pts'].append(bbox_pt)
            self.centertrack_dict[side][track_id]['p_worlds'].append(p_world)

        # Get set of current track ids in dictionary
        current_track_ids = set(self.centertrack_dict[side].keys())

        # Get track ids not detected this time using set difference operation
        not_detected_ids = current_track_ids.difference(detected_ids)

        # Delete tracks not detected this time
        for track_id in not_detected_ids:
            # Delete item from dictionary
            del self.centertrack_dict[side][track_id]

    def transform_bbox_to_point(self, side, track_id):
        """
        Method to transform a bounding box to a single point.
        We use the prior and current bounding box center to create a motion vector.
        We find the intersection of the 
        """

        # Get the dictionary for this track
        track_dict = self.centertrack_dict[side][track_id]
        window = 25

        # If this is the first detection, determine the bbox_pt with original method of center of bottom edge
        if len(track_dict['bboxes']) <= window:
            # Unpack bounding box
            x1, y1, x2, y2 = track_dict['bboxes'][0]
            # Get bbox point from middle of bottom edge of bounding box
            bbox_pt = (x1+x2)/2.0, y2
        
        else:
            # Get current and previous bounding boxes
            current_bbox = track_dict['bboxes'][-1]
            past_bbox = track_dict['bboxes'][-window]
            past_bboxes = track_dict['bboxes'][-window:-1]
            past_bbox_pt_xs = []
            past_bbox_pt_ys = []
            # Unpack the boxes
            x1_c, y1_c, x2_c, y2_c = current_bbox
            # x1_p, y1_p, x2_p, y2_p = past_bbox
            for bbox in past_bboxes:
                x1_p, y1_p, x2_p, y2_p = bbox
                past_bbox_pt = (x1_p + x2_p)/2.0, y2_p
                past_bbox_pt_xs.append(past_bbox_pt[0])
                past_bbox_pt_ys.append(past_bbox_pt[1])
            # Compute centers
            current_bbox_ct = (x1_c + x2_c)/2.0, (y1_c + y2_c)/2.0
            past_bbox_ct = (x1_p + x2_p)/2.0, (y1_p + y2_p)/2.0
            # Compute center of bottom edges
            current_bbox_pt = (x1_c + x2_c)/2.0, y2_c
            # past_bbox_pt = (x1_p + x2_p)/2.0, y2_p
            past_bbox_pt = self.moving_average(past_bbox_pt_xs, window-2, ma_type='simple'), \
                            self.moving_average(past_bbox_pt_ys, window-2, ma_type='simple')
            # Compute motion vector
            # motion_vector = current_bbox_ct[0] - past_bbox_ct[0], current_bbox_ct[1] - past_bbox_ct[1]
            motion_vector = current_bbox_pt[0] - past_bbox_pt[0], current_bbox_pt[1] - past_bbox_pt[1]
            
            # Determine intersection of motion vector line and box boundary

            # Step 1. Rescale motion_vector to have norm equal to diagonal of the current bounding box
            motion_vector_norm = math.sqrt(motion_vector[0]**2 + motion_vector[1]**2)
            current_bbox_norm = math.sqrt((x2_c - x1_c)**2 + (y2_c - y1_c)**2)
            scale = current_bbox_norm/motion_vector_norm
            motion_vector_scaled = scale*motion_vector[0], scale*motion_vector[1]

            # Step 2. Create line segment from center of current bbox to the end of the rescaled motion vector
            end_point = current_bbox_ct[0] + motion_vector_scaled[0], current_bbox_ct[1] + motion_vector_scaled[1]

            # Step 3. Apply Cohen-Sunderland clipping to find intersection of line segment with bounding box
            bbox_pt = cohenSutherlandBBoxClip(current_bbox, current_bbox_ct, end_point)

        # Return the bbox_pt
        return bbox_pt

    def moving_average(self, x, n, ma_type='simple'):
        """
        Compute an n period moving average.

        type is 'simple' | 'exponential'
        """
        x = np.asarray(x)
        if ma_type == 'simple':
            weights = np.ones(n)
        elif ma_type == 'exponential':
            weights = np.exp(np.linspace(-1., 0., n))

        weights /= weights.sum()

        a = np.convolve(x, weights, mode='full')[:len(x)]
        a[:n] = a[n]
        return a[-1]

    def project_bbox_pt_to_p_world(self, camera, bbox_pt, as_list = True):
        """
        Method to project the single pixel coordinates for an object (bbox_pt) into 
        the 3D world coordinate system using the WorldProjector and Undistorter 
        helper classes for this camera
        """

        # Get the distortion-corrected pixel coordinates
        undistorted_bbox_pt = self.distortion_corrector[camera].undistort_points(bbox_pt)

        # Project the distortion-corrected pixel coordinates to the world coordinate system, type is numpy array
        p_world = self.world_projector[camera].project_image_to_world(undistorted_bbox_pt[0], undistorted_bbox_pt[1], 0)

        # Return world coordinates, as an array or list as requested
        if as_list == True:
            return p_world.tolist()
        else:
            return p_world

    def visualize_detections(self, detection_img, img_metadata, img_array):
        """
        Method to create a visualization of detections for debugging purposes
        self.centertrack_dict['left'] and self.centertrack_dict['right'] are dictionaries,
        Each key: value pair is a track IDs and track dictionary consisting of:
  
        'class': COCO class integer
        'scores': detection confidence score
        'bboxes' : list of bounding boxes
        'detection_centers': list of detection centers
        'tracking_vectors': list of tracking vectors
        'bbox_pts': list of computed bounding box points for projection
        'p_worlds': world-projected point from the bbox_pts

        Entries that are lists have most recent quanties last
        """

        # Convert RBG source image to BGR
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Get camera side to determing horizontal pixel offset
        if img_metadata['camera'] == 'left_wide':
            side = 'left'
            offset = img_metadata['width']
        elif img_metadata['camera'] == 'right_wide':
            side = 'right'
            offset = 0
        
        # Step 1. Insert source image
        if img_metadata['camera'] == 'left_wide':
            detection_img[:, img_metadata['width']:, :] = img_array
        elif img_metadata['camera'] == 'right_wide':
            detection_img[:, :img_metadata['width'], :] = img_array
        
        
        # Step 2. Draw detections in self.centertrack_dict
        for track_id, track_dict in self.centertrack_dict[side].items():
            # Label string using class ID and classesIncluded dictionary in configs
            bbox_label = track_id+self.node.configs.classesIncluded[track_dict['class']]['name']
            # Get bounding box pixels in integers
            x1, y1, x2, y2 = [int(x) for x in track_dict['bboxes'][-1].tolist()]
            # Add offset to horizontal pixel coordinates
            x1 += offset
            x2 += offset
            # Determine which font scale and thickness to use based on width of bbox
            label_scale = 0.6 if (x2 - x1) < 50 else 0.8
            label_thickness = 1 if (x2 - x1) < 50 else 1
            label_font_color = (255, 255, 255) # White rgb(255, 255, 255)   
            # Drawing bounding box
            bbox_color = (255, 255, 0) # Cyan rgb(0, 255, 255)
            bbox_thickness = 1
            # Get pixel extent of label text
            (label_w, label_h), label_base = cv2.getTextSize(bbox_label, cv2.FONT_HERSHEY_DUPLEX, label_scale, label_thickness)
            # Draw bbox rectangle
            cv2.rectangle(detection_img, (int(x1), int(y1)), (int(x2), int(y2)), bbox_color, bbox_thickness)
            # Draw bbox label background and text
            cv2.rectangle(detection_img, (int(x1), int(y1) - label_h - label_base), (int(x1)+label_w, int(y1)), bbox_color, -1)
            cv2.putText(detection_img, bbox_label, (int(x1), int(y1) - label_base ), cv2.FONT_HERSHEY_DUPLEX, 
                        label_scale, label_font_color, label_thickness, cv2.LINE_AA)
            
            
            # Draw bbox point as the center of the bottom edge
            bbox_pt = int((x1 + x2)/2), y2
            cv2.circle(detection_img, bbox_pt, 3, (255, 255, 255), -1) # Filled white dot
            
            # Draw bbox point as the center of the bottom edge
            bbox_pt = int((x1 + x2)/2 + (x2 - offset) / (detection_img.shape[1]/2) * (x2 - x1) / 2), y2
            cv2.circle(detection_img, bbox_pt, 3, (0, 0, 255), -1)
            
            # Draw bbox point as the center of the bottom edge
            bbox_pt = int((x1 + x2)/2 - (x2 - offset) / (detection_img.shape[1]/2) * (x2 - x1) / 4),\
                      int(y1 + (y2 - y1) / 2)
            cv2.circle(detection_img, bbox_pt, 3, (255, 0, 0), -1)           
            
            # Draw bbox point as the computed bbox point
            bbox_pt = int(track_dict['bbox_pts'][-1][0]) + offset, int(track_dict['bbox_pts'][-1][1])
            cv2.circle(detection_img, bbox_pt, 2, (0, 255, 255), -1) # Filled yellow dot
            # Draw line segment from bounding box center to computed bbox point (this is the clipped Cohen Sutherland line segment)
            bbox_ct = int((x1 + x2)/2), int((y1 + y2)/2)
            cv2.line(detection_img, bbox_ct, bbox_pt, (0, 255, 255), thickness=2) # Green tracking vector

            # Draw CenterTrack ct output, adding offset to horizontal coordinate
            ct_pt = track_dict['detection_centers'][-1]
            ct_pt = int(ct_pt[0])+ offset, int(ct_pt[1])
            cv2.circle(detection_img, ct_pt, 2, (0, 165, 255), -1) # Filled orange 
            
            
          
            
            
            # Draw CenterTrack tracking vector
            tracking_vector = track_dict['tracking_vectors'][-1]
            tracking_pt = ct_pt[0] + int(tracking_vector[0]), ct_pt[1] + int(tracking_vector[1])
            cv2.line(detection_img, ct_pt, tracking_pt, (0, 255, 0), thickness=2) # Green tracking vector
        
        # Step 3. Save image
        self.detection_img = detection_img
        
    

import cython

class CamnetNode(object):
    def __init__(self, rosTopics, configs):
        print("--INIT CAMNET NODE--")
        if cython.compiled:
            print("--RUNNING CAMNET PURE IN C--")
        else:
            print("--RUNNING CAMNET PURE IN PYTHON--")
        # if options.enableMarkers:
        #     print('+++camnet markerballs enabled')
        # else:
        #     print('camnet markerballs disabled---')
        # if options.enableBoxIms:
        #     print('+++camnet box images enabled')
        # else:
        #     print('camnet box images disabled---')
        
        print('Initiating Camnet Node')

        # Get ROS topics and configs objects
        self.rosTopics = rosTopics
        self.configs = configs
        print('Object Detection configs:')
        print(self.configs.__dict__)

        # Start GPU and CPU threads
        self.startCamnetCPUThread()
        self.startCamnetGPUThread()

        # Init ROS node
        rospy.init_node('camnet_node', anonymous=True)

        #---------------------#
        #     Subscribers     #
        #---------------------#

        rospy.Subscriber(self.rosTopics.syncedImageTopic.format('left_wide'), Image, self.callback_sync_img)
        rospy.Subscriber(self.rosTopics.syncedImageTopic.format('right_wide'), Image, self.callback_sync_img)

        # Spin node
        rospy.spin()

    #--------------------------#
    #     Callback methods     #
    #--------------------------#  

    def callback_sync_img(self, image_msg):
        self.camnet_gpu.queue.put(image_msg)
    
    #------------------------#
    #     Action methods     #
    #------------------------#     
    
    def startCamnetGPUThread(self):
        self.camnet_gpu = CamnetGPU(self)
        self.camnet_gpu.thread = Thread(target=self.camnet_gpu.runTarget)
        self.camnet_gpu.thread.daemon = True
        self.camnet_gpu.thread.start()        

    def startCamnetCPUThread(self):
        self.camnet_cpu = CamnetCPU(self)
        self.camnet_cpu.thread = Thread(target=self.camnet_cpu.runTarget)
        self.camnet_cpu.thread.daemon = True
        self.camnet_cpu.thread.start()

                
