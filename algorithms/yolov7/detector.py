import colorsys
# ----------------------------------------
import numpy
import cv2
import torch
# ----------------------------------------
from .utils.general import non_max_suppression, scale_coords
# from .models.common import DetectMultiBackend
from .models.experimental import attempt_load, attempt_load_pkl
# --------------------------------------------------

class YOLOv7Detector(object):

    @classmethod
    def from_config(cls, config):
        target_size = config.YOLO_TARGET_SIZE
        padding_color = config.YOLO_PADDING_COLOR
        conf_thres = config.YOLO_THRESHOLD_CONF
        iou_thres = config.YOLO_THRESHOLD_IOU
        device = torch.device(config.YOLO_DEVICE)
        net_conf = config.YOLO_NET_CONF
        weight_path = config.YOLO_WEIGHT_PATH
        classes = config.YOLO_CLASSES
        multi_label = config.YOLO_MULTI_LABLE
        scaleFill = config.SCALE_FILL
        batchFill = config.BATCH_FILL
        pick_classes = config.PICK_CLASSES
        # ----------------------------------------
        return cls(target_size, padding_color, conf_thres, iou_thres, device, net_conf, weight_path, classes,
                   multi_label, scaleFill, batchFill, pick_classes)

    def __init__(self, target_size, padding_color, conf_thres, iou_thres, device, net_conf, weight_path, classes,
                 multi_label, scaleFill, batchFill, pick_classes):
        self._target_size = target_size
        self._padding_color = padding_color
        self._conf_thres = conf_thres
        self._iou_thres = iou_thres
        self._multi_label = multi_label
        self._scaleFill = scaleFill
        self._batchFill = batchFill
        self._net_conf = net_conf
        self._pick_classes = pick_classes
        # ----------------------------------------
        self._device = torch.device(device)
        # self._model = Model(net_conf, nc=len(classes))
        # print("before _model")
        # self._model = attempt_load(weight_path, map_location=self._device)
        self._model = attempt_load_pkl(weight_path, yaml_path=self._net_conf, nc=len(classes),
                                       device=device, inplace=True, fuse=True)
        # print("after _model")
        # self.auto_padding = self._model.pt or self._model.pkl
        self.auto_padding = True
        # self.half = self._model.fp16
        self.half = False
        # self.stride = self._model.stride
        self.stride = 32
        # with open(weight_path, 'rb') as f:
        #     self._model.load_state_dict(pickle.load(f))
        # self._model.fuse().eval().half().to(self._device)
        self._model.eval().to(self._device)
        # ----------------------------------------
        self._classes = classes
        num_classes = len(self._classes)
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self._colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    def __call__(self, *args, **kwargs):
        return self.det(*args, **kwargs)

    def det(self, image_list: list, *args, **kwargs):
        # s1_time = time.time()
        batch, batch_image_size = self._preprocess(image_list, auto=self.auto_padding, half=self.half)
        # s2_time = time.time()
        # print(f'前处理{round((s2_time-s1_time)*1000, 2)}ms')
        # ----------------------------------------
        batch_output = self._model(batch, augment=False)
        # print('batch_output', len(batch_output), batch_output[0].shape, len(batch_output[1]), batch_output[1][0].shape,
        #       batch_output[1][1].shape, batch_output[1][2].shape)
        # s3_time = time.time()
        # print(f'模型{round((s3_time - s2_time) * 1000, 2)}ms')
        batch_output = batch_output[0]
        batch_pred = non_max_suppression(
            batch_output,
            conf_thres=self._conf_thres, iou_thres=self._iou_thres,
            classes=self._pick_classes, agnostic=False, multi_label=self._multi_label
        )
        # print(f'后处理{round((time.time() - s3_time) * 1000, 2)}ms')
        # ----------------------------------------
        return self._postprocess(batch_pred, batch.shape[-2:], batch_image_size)

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
            scala_ratio = min((self._target_size[0]/image_shape[0]),
                              (self._target_size[1]/image_shape[1]))
            # scala_ratio = min(scala_ratio, 1.0)
            image_scaled_shape = (int(round(image_shape[0] * scala_ratio)),
                                  int(round(image_shape[1] * scala_ratio)))
            image_scaled = image

            # ------------------------------
            delta_height, delta_width = (self._target_size[0]-image_scaled_shape[0],
                                         self._target_size[1]-image_scaled_shape[1])

            if self._scaleFill:
                delta_height, delta_width = 0, 0
                image_scaled_shape = (self._target_size[0], self._target_size[1])
            if not self._batchFill:
                delta_height, delta_width = numpy.mod(delta_height, self.stride),\
                                            numpy.mod(delta_width, self.stride)

            delta_height_half, delta_width_half = delta_height/2, delta_width/2

            if image_scaled_shape != image_shape:
                image_scaled = cv2.resize(image, image_scaled_shape[::-1],
                                          interpolation=cv2.INTER_LINEAR)

            top, bottom = int(round(delta_height_half-0.1)), int(round(delta_height_half+0.1))
            left, right = int(round(delta_width_half - 0.1)), int(round(delta_width_half + 0.1))
            image_padded = cv2.copyMakeBorder(image_scaled, top, bottom, left, right,
                                              cv2.BORDER_CONSTANT, value=self._padding_color)
            # ------------------------------
            image_torch_format = cv2.cvtColor(image_padded, cv2.COLOR_BGR2RGB)\
                                    .transpose(2, 0, 1)
            image_torch_format = numpy.ascontiguousarray(image_torch_format)
            image_tensor = torch.from_numpy(image_torch_format).to(self._device)
            image_tensor = image_tensor.half() if half else image_tensor.float()
            image_tensor /= 255.0
            image_unsqueezed = image_tensor.unsqueeze(0)
            batch_image_tensors.append(image_unsqueezed)
        # ----------------------------------------
        batch = torch.cat(batch_image_tensors, 0)
        return batch, batch_image_size

    def _postprocess(self, batch_pred, input_image_shape, batch_image_size):
        results = []
        for idx, (pred_item, image_size) in enumerate(zip(batch_pred, batch_image_size)):
            result_item = []
            if (pred_item is not None) and len(pred_item):
                pred_item[:, :4] = scale_coords(
                    input_image_shape, pred_item[:, :4], image_size).round()
                for *p1p2, conf, klass_idx in pred_item:
                    klass_id = int(klass_idx.item())
                    klass_name = self._classes[klass_id]
                    x1, y1, x2, y2 = p1p2
                    x1, y1, x2, y2 = int(x1.item()), int(y1.item()),\
                                     int(x2.item()), int(y2.item())
                    xo, yo, w, h = round((x1+x2)/2), round((y1+y2)/2), (x2-x1), (y2-y1)
                    score = round(conf.item(), 2)
                    result_item.append((klass_name, score, (x1, y1, x2, y2), (xo, yo, w, h)))
            results.append(result_item)
        return results

