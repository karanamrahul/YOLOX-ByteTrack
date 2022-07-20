

### Setup Instructions


1. Clone the [YOLOX-ByteTrack](https://github.com/karanamrahul/YOLOX-ByteTrack.git) repository.
2. Run the following command to install the dependencies:
    ```
    cd YOLOX-ByteTrack
    pip install -r requirements.txt

    ```
3. Run the following command to install YOLOX and its dependencies:
    ```
    cd YOLOX
    pip install -v -e .
    ```
4. Clone Torch2TRT repository and install the dependencies:
    ```
    cd ..
    git clone https://github.com/NVIDIA-AI-IOT/torch2trt
    cd torch2trt
    python setup.py install
    ```





### Running the Model

#### Step 1: Download the pre-trained models from the YOLOX repository.

[YOLOX-large](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth)

[YOLOX-small](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth)

[YOLOX-medium](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth)

[YOLOX-tiny](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth)

[YOLOX-Nano](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth)

#### Step 2: Convert the model to TRT.

YOLOX models can be easily conveted to TensorRT models using torch2trt

If you want to convert to trt model, use the flag -n to specify a model name:

```
cd YOLOX
python tools/trt.py -n <YOLOX_MODEL_NAME> -c <YOLOX_CHECKPOINT>

```
For example:

```
python tools/trt.py -n yolox-s -c yolox_s.pth

```

* <YOLOX_MODEL_NAME> can be: yolox-nano, yolox-tiny. yolox-s, yolox-m, yolox-l, yolox-x.

* <YOLOX_CHECKPOINT> is the path to the checkpoint file ( yolox_l.pth ).


#### Step 3: Add bytetrack_module to "IRIS_ROOT/standard_ws/src/object_detection/src/"

Copy the following files from "YOLOX-ByteTrack/bytetrack_module.py" to "IRIS_ROOT/standard_ws/src/object_detection/src/"

```
IRIS_ROOT/standard_ws/src//object_detection/src/bytetrack_module.py
```

Make sure to add the following lines to the below files:


```
IRIS_ROOT/standard_ws/src/object_detection/src/configObjectDetection.yaml
detectionNetwork: 'YOLOX_BYTETRACK'

IRIS_ROOT/standard_ws/src/local_paths/LocalPathsConfig.yaml
bytetrackPath : '/path_to_repository/YOLOX-ByteTrack'

```


#### To check ByteTrack model is loaded correctly, run the following command:

```
python detector.py
    
```
Note: Make sure you have given the path for the demo video file in the detector.py file.
If the model is loaded correctly, you should output of the detector for the demo video



#### Run the highway mode similar to the LOCAL_IRIS repository instructions.
