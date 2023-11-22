import cv2


class YOLOv5ROIWrapper(object):

    def __init__(self, detector):
        self._detector = detector

    def __call__(self, *args, **kwargs):
        return self.det(*args, **kwargs)

    def det(self, input_list: list, roi_list: list, *args, **kwargs):
        batch_image, batch_delta_point = self._preprocess(input_list, roi_list)
        # ----------------------------------------
        pred_list = self._detector.det(batch_image)
        # ----------------------------------------
        output_list = self._postprocess(pred_list, batch_delta_point)
        return output_list

    def _preprocess(self, input_list: list, roi_list):
        batch_image = []
        batch_delta_point = []
        for image, roi_region in zip(input_list, roi_list):
            x1, y1, x2, y2 = roi_region
            croped_img = image[y1:y2, x1:x2]
            delta_point = (x1, y1)
            batch_image.append(croped_img)
            batch_delta_point.append(delta_point)
        return batch_image, batch_delta_point

    def _postprocess(self, pred_list: list, batch_delta_point: list):
        output_list = []
        for pred, delta_point in zip(pred_list, batch_delta_point):
            dx, dy = delta_point
            pred_new = [(klass_name, score, (x1+dx, y1+dy, x2+dx, y2+dy), (xo+dx, yo+dy, w, h))
                        for klass_name, score, (x1, y1, x2, y2), (xo, yo, w, h) in pred]
            output_list.append(pred_new)
        return output_list

    @property
    def get_class_name(self):
        return self._detector.get_class_name

    @property
    def get_color(self):
        return self._detector.get_color




