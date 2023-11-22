import colorsys
import pickle
# ----------------------------------------
import numpy
import cv2
import torch
import thop
import torch.nn.functional as F
# ----------------------------------------
# from .models.general import non_max_suppression, scale_coords
# from .models.yolo import Model
from .models.common import DetectMultiBackend
# --------------------------------------------------
import time

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
# IMAGENET_MEAN = 0., 0., 0.  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation
# IMAGENET_STD = 1, 1, 1  # RGB standard deviation


class YOLOv8Classifier(object):

    @classmethod
    def from_config(cls, config):
        target_size = config.YOLO_TARGET_SIZE
        padding_color = config.YOLO_PADDING_COLOR
        conf_thres = config.YOLO_THRESHOLD_CONF
        device = torch.device(config.YOLO_DEVICE)
        net_conf = config.YOLO_NET_CONF
        weight_path = config.YOLO_WEIGHT_PATH
        classes = config.YOLO_CLASSES
        scaleFill = config.SCALE_FILL
        resize_mode = config.RESIZE_MODE
        # ----------------------------------------
        return cls(target_size, padding_color, conf_thres, device, net_conf, weight_path, classes,
                   scaleFill, resize_mode)

    def __init__(self, target_size, padding_color, conf_thres, device, net_conf, weight_path, classes,
                 scaleFill, resize_mode):
        self._target_size = target_size
        self._padding_color = padding_color
        self._conf_thres = conf_thres
        self._scaleFill = scaleFill
        self._resize_mode = resize_mode
        # ----------------------------------------
        self._device = torch.device(device)

        self._model = DetectMultiBackend(weight_path, yaml=net_conf, nc=len(classes), device=self._device,
                                         data=classes, classify=True, fp16=False)

        self.auto_padding = self._model.pt or self._model.pkl
        self.half = self._model.fp16
        self.stride = self._model.stride

        self._model.eval().to(self._device)
        flops = thop.profile(self._model, inputs=(torch.randn(1, 3, *target_size),), verbose=False)[0] / 1E9 * 2
        print(f'model:{round(flops, 2)}GFLOPs')
        # ----------------------------------------
        self._classes = classes
        num_classes = len(self._classes)
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self._colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    def __call__(self, *args, **kwargs):
        return self.classify(*args, **kwargs)

    def classify(self, image_list: list, *args, **kwargs):
        # s1_time = time.time()
        batch, batch_image_size = self._preprocess(image_list, auto=self.auto_padding, half=self.half)
        # s2_time = time.time()
        # print(f'前处理{round((s2_time-s1_time)*1000, 2)}ms')
        # ----------------------------------------
        # print('batch', batch)
        # print('batch', batch)
        batch_output = self._model(batch, augment=False)
        # print('batch_output', batch_output)
        # print('top', batch_output[0, 58])
        batch_pred = batch_output
        # batch_pred = F.softmax(batch_output, dim=1)  # probabilities
        # print('batch_output', batch_output)
        # print('batch_pred', batch_pred)
        top5i = batch_pred.argsort(1, descending=True).tolist()  # top 5 indices
        # print('top5i', top5i)
        batch_res = []
        for i, index_res in enumerate(top5i):
            pred_cls = self._classes[index_res[0]]
            pred_score = batch_pred[i, index_res[0]].item()
            batch_res.append([pred_cls, pred_score])
        # print('batch_res', batch_res)

        # s3_time = time.time()
        # print(f'模型{round((s3_time - s2_time) * 1000, 2)}ms')
        # batch_pred = non_max_suppression(
        #     batch_output,
        #     conf_thres=self._conf_thres, iou_thres=self._iou_thres,
        #     classes=None, agnostic=False, multi_label=self._multi_label
        # )
        # print(f'后处理{round((time.time() - s3_time) * 1000, 2)}ms')
        # ----------------------------------------
        return batch_res
        # return self._postprocess(batch_pred, batch.shape[-2:], batch_image_size)

    def get_color(self, class_id):
        return self._colors[class_id]

    def get_class_name(self, class_id):
        return self._classes[class_id]

    def _preprocess(self, image_list: list, auto=True, half=True) -> tuple:
        batch_image_tensors = []
        batch_image_size = []
        for image in image_list:
            image_shape = image.shape[:2]
            batch_image_size.append(image_shape)
            scala_ratio = min((self._target_size[0] / image_shape[0]),
                              (self._target_size[1] / image_shape[1]))
            scala_ratio = min(scala_ratio, 1.0)
            image_scaled_shape = (int(round(image_shape[0] * scala_ratio)),
                                  int(round(image_shape[1] * scala_ratio)))
            image_scaled = image

            # ------------------------------
            if self._resize_mode == 'Letter_box':

                delta_height, delta_width = (self._target_size[0] - image_scaled_shape[0],
                                             self._target_size[1] - image_scaled_shape[1])
                delta_height, delta_width = numpy.mod(delta_height, self.stride), \
                    numpy.mod(delta_width, self.stride)
                delta_height, delta_width = delta_height / 2, delta_width / 2
                if image_scaled_shape != image_shape:
                    image_scaled = cv2.resize(image, image_scaled_shape[::-1],
                                              interpolation=cv2.INTER_LINEAR)

                top, bottom = int(round(delta_height - 0.1)), int(round(delta_height + 0.1))
                left, right = int(round(delta_width - 0.1)), int(round(delta_width + 0.1))
                image_padded = cv2.copyMakeBorder(image_scaled, top, bottom, left, right,
                                                  cv2.BORDER_CONSTANT, value=self._padding_color)

            elif self._resize_mode == 'Center_crop':
                image_height, image_wide = image_shape
                image_short_length = min(image_height, image_wide)
                top_crop, left_crop = (image_height - image_short_length) // 2, (image_wide - image_short_length) // 2
                image_crop = image[top_crop:top_crop + image_short_length, left_crop:left_crop + image_short_length]
                image_padded = cv2.resize(image_crop, self._target_size,
                                          interpolation=cv2.INTER_LINEAR)

            # Scale_fill
            else:
                image_padded = cv2.resize(image, self._target_size,
                                          interpolation=cv2.INTER_LINEAR)

            # ------------------------------
            image_torch_format = cv2.cvtColor(image_padded, cv2.COLOR_BGR2RGB)
            image_torch_format = image_torch_format.astype(numpy.float32) / 255.
            image_torch_format = (image_torch_format - IMAGENET_MEAN) / IMAGENET_STD
            image_torch_format = image_torch_format.transpose(2, 0, 1)
            image_torch_format = numpy.ascontiguousarray(image_torch_format)
            image_tensor = torch.from_numpy(image_torch_format).to(self._device)
            image_tensor = image_tensor.half() if half else image_tensor.float()
            # image_tensor /= 255.0

            image_unsqueezed = image_tensor.unsqueeze(0)
            batch_image_tensors.append(image_unsqueezed)
        # ----------------------------------------
        batch = torch.cat(batch_image_tensors, 0)
        return batch, batch_image_size

