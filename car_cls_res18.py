import numpy as np
import cv2, os, shutil, random
from tqdm import tqdm
from algorithms.yolov5_7 import YOLOv5Detector, Res18Classifier
import configs.res18_cls as res18_config
import configs.yolov5s_car_type as yolov5_config_car_type


classifier_res18 = Res18Classifier.from_config(res18_config)
detector_car_type = YOLOv5Detector.from_config(yolov5_config_car_type)

colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(yolov5_config_car_type.YOLO_CLASSES))]
colors_dict = dict(zip(yolov5_config_car_type.YOLO_CLASSES, colors))
path = r'D:\datasets\highway_20230816\camera2_images_pick'
img_list = os.listdir(path)
img_list = sorted(img_list, key=lambda x:(int((x.split('_')[1]))*100000)+int((x.split('_')[2].strip('.jpg'))))

back_roi = [(837, 334), (881, 350), (935, 361), (992, 380), (1037, 391), (1107, 420), (1176, 457), (1221, 512), (1208, 584), (1164, 665), (1116, 739), (1049, 813), (935, 881), (785, 970), (489, 1110), (23, 1356), (36, 1437), (2205, 1434), (2167, 721), (2088, 638), (1851, 545), (1589, 422), (1385, 370), (1181, 330), (1073, 318), (981, 305), (892, 283)]
back_roi = np.expand_dims(back_roi, 1)
for img_name in tqdm(img_list[60:]):
    print(img_name)
    img_path = os.path.join(path, img_name)
    img = cv2.imread(img_path)
    img_copy = img.copy()

    pred = detector_car_type.det([img])[0]
    # print('pred', pred)
    img_crop_list = []
    car_crop_xy = []
    towards_list = []

    for klass_name, score, (x1, y1, x2, y2), (xo, yo, w, h) in pred:
        # if klass_name not in ['car']:
        #     continue
        img_crop = img[y1:y2, x1:x2]
        car_crop_xy.append([x1, y1, x2, y2])
        img_crop_list.append(img_crop)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), colors_dict[klass_name], 3)
        roi_flag = cv2.pointPolygonTest(back_roi, (xo, yo), False)
        if roi_flag > 0:
            towards_list.append('back')
        else:
            towards_list.append('front')

    pred_cls = classifier_res18.classify(img_crop_list)
    print('pred_cls', pred_cls)
    for i, (color_pred, color_score, towards_pred, towards_score) in enumerate(pred_cls):
        x1, y1, x2, y2 = car_crop_xy[i]
        cv2.putText(img_copy, f'{color_pred}: {round(color_score, 2)}', (x1+10, y1+10), 0, 0.7, (0,255,0), 2)
        cv2.putText(img_copy, f'{towards_list[i]}', (x1+10, y1+40), 0, 0.7, (0,255,0), 2)
        # cv2.putText(img_copy, f'{towards_pred}: {round(towards_score, 2)}', (x1+10, y1+40), 0, 0.7, (0,255,0), 2)

    cv2.namedWindow('img_copy', 0)
    cv2.imshow('img_copy', img_copy)
    cv2.waitKey(0)

