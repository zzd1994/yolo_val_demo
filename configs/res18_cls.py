import os
# ----------------------------------------
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
# --------------------------------------------------


# YOLO_DEVICE = "cuda:0"
YOLO_DEVICE = "cpu"
YOLO_CLASSES = ['yellow', 'orange', 'green', 'gray', 'red', 'blue', 'white', 'golden', 'brown', 'black', 'other',
                'back', 'front', 'unknown']

# YOLO_NET_CONF = os.path.join(CURRENT_DIR, 'yolov5n_cls_tccd.yaml')
YOLO_WEIGHT_PATH = 'weights/res18_trans_lhh.pth'
YOLO_TARGET_SIZE = (224, 224)       # (w, h)
YOLO_PADDING_COLOR = (114, 114, 114)
YOLO_THRESHOLD_CONF = 0.35      # 这个用不到
SCALE_FILL = True       # False是前处理时对图像keep_ratio方式resize
RESIZE_MODE = 'Center_crop'     # 共三种模式[Letter_box, Scale_fill, Center_crop]
