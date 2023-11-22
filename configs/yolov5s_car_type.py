import os
# ----------------------------------------
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
# --------------------------------------------------

YOLO_DEVICE = "cuda:0"
# YOLO_DEVICE = "cpu"
YOLO_CLASSES = [
    'car', 'motorcycle',  'bus', 'truck'
]
YOLO_NET_CONF = os.path.join(CURRENT_DIR, 'yolov5s_v6.yaml')
YOLO_WEIGHT_PATH = 'weights/llh_car_type.pkl'
YOLO_TARGET_SIZE = (640, 640)
YOLO_PADDING_COLOR = (114, 114, 114)
YOLO_THRESHOLD_CONF = 0.25
YOLO_THRESHOLD_IOU = 0.4
YOLO_MULTI_LABLE = False
YOLO_AGNOSTIC = True
SCALE_FILL = False       # False是前处理时对图像keep_ratio方式resize
BATCH_FILL = False       # 在处理batch时，最大尺寸填充，保证batch推理时每张图像尺寸一样