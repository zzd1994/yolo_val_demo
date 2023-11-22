import colorsys
import pickle
from algorithms.yolox import models
import torchvision
# ----------------------------------------
from .models import YOLOX, YOLOPAFPN, YOLOXHead


# ----------------------------------------
import numpy as np
from numpy.lib.arraypad import pad
# import pycuda
# import pycuda.autoinit
# from pycuda import gpuarray
import time
import cv2
import torch
import torch.distributed as dist
import torch.nn as nn
# import pycuda.driver as cuda
# import pycuda.autoinit

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]


    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)




class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
 
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
 
    def __repr__(self):
        return self.__str__()


class YOLOXDetector(object):
    @classmethod
    def from_config(cls,config):
        target_size = config.YOLOX_TARGET_SIZE
        padding_color = config.YOLOX_PADDING_COLOR
        num_classes = config.YOLOX_NUM_CLASSES
        net_depth = config.YOLOX_DEPTH
        net_width = config.YOLOX_WIDTH
        conf_thres = config.YOLOX_THRESHOLD_CONF
        iou_thres = config.YOLOX_THRESHOLD_IOU
        weight_path = config.YOLOX_WEIGHT_PATH
        classes = config.YOLOX_CLASSES
        nmsthres = config.YOLOX_NMS_THRES
        rgb_means = config.YOLOX_RGB_MEANS
        std = config.YOLOX_STD

        # ----------------------------------------
        return cls(target_size, padding_color, num_classes, net_depth, net_width, conf_thres, iou_thres, weight_path, classes,nmsthres,rgb_means,std)


    def __init__(self,target_size, padding_color, num_classes, net_depth, net_width, conf_thres, iou_thres, weight_path, classes,nmsthres,rgb_means,std):
        self._target_size = target_size
        self._padding_color = padding_color
        self._conf_thres = conf_thres
        self._iou_thres = iou_thres
        self._device = "gpu" #only gpu
        self._classes = classes
        self.num_classes = num_classes
        self.depth = net_depth
        self.width = net_width
        self.weight_path = weight_path
        num_classes = len(self._classes)
        self._nms_thres = nmsthres
        self._mean = rgb_means
        self._std = std
        self._ratio = 0
        self.model = self.get_model()
        self.model.cuda()
        self.model.eval()
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        ckpt = torch.load(self.weight_path,map_location=('cuda:0'))
        self.model.load_state_dict(ckpt["model"])





    def __call__(self, *args, **kwargs):
        return self.det(*args, **kwargs)

    def det(self,img,*args, **kwargs):
        img, ratio = self.preproc(img[0])
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        img = img.cuda()
        outputs = self.model(img)
        output = self.postprocess(outputs, ratio)
        return output

    def get_model(self):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def preproc(self,img,swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = np.ones((self._target_size[0], self._target_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(self._target_size) * 114.0
        #img = np.array(img)
        r = (self._target_size[0] / img.shape[0], self._target_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r[1]), int(img.shape[0] * r[0])),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded_img[: int(img.shape[0] * r[0]), : int(img.shape[1] * r[1])] = resized_img
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def postprocess(self,prediction,ratio):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]
        prediction[:, :, 0] /= ratio[1]
        prediction[:, :, 1] /= ratio[0]
        prediction[:, :, 2] /= ratio[1]
        prediction[:, :, 3] /= ratio[0]

        output = [None for _ in range(len(prediction))]
        results = []
        for i, image_pred in enumerate(prediction):
            result_item = []
            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + self.num_classes], 1, keepdim=True)
            
            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= self._conf_thres).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            
            
            if not detections.size(0):
                continue


            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                self._nms_thres,
            )

            detections = detections[nms_out_index]

            for j in range(len(detections)):
                p1p2 = (detections[j,0].item(),detections[j,1].item(),detections[j,2].item(),detections[j,3].item())
                score = (detections[j,4]*detections[j,5]).item()
                class_id = int(detections[j,6].item())
                klass_name = self._classes[class_id]
                x1, y1, x2, y2 = p1p2
                xo, yo, w, h = round((x1+x2)/2), round((y1+y2)/2), (x2-x1), (y2-y1)
                result_item.append((klass_name,score,(x1, y1, x2, y2), (xo, yo, w, h)))
            results.append(result_item)

            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))
        
            #print("detections : ",detections.cpu().detach().numpy())

        return results











