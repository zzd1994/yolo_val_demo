import numpy as np
# ----------------------------------------
from .deep_sort import nn_matching
from .deep_sort.detection import Detection
from .deep_sort.tracker import Tracker
from .utils import preprocessing
from .tools import generate_detections
# ----------------------------------------
from .deep_sort.track import TrackState
# --------------------------------------------------


class DeepSORTTrackProxy(object):

    def __init__(self, track):
        self._track = track

    @property
    def track_id(self):
        return self._track.track_id

    @property
    def bbox(self):
        x1, y1, x2, y2 = self._track.to_tlbr()
        # ----------------------------------------
        xo = round((x1+x2) / 2)
        yo = round((y1+y2) / 2)
        # ----------------------------------------
        x1 = round(x1)
        y1 = round(y1)
        x2 = round(x2)
        y2 = round(y2)
        # ----------------------------------------
        w = x2 - x1
        h = y2 - y1
        # ----------------------------------------
        return (x1, y1, x2, y2), (xo, yo, w, h)

    @property
    def time_since_update(self):
        return self._track.time_since_update

    @time_since_update.setter
    def time_since_update(self, value):
        self._track.time_since_update = value

    @property
    def is_tentative(self):
        return self._track.is_tentative

    @property
    def is_confirmed(self):
        return self._track.is_confirmed

    @property
    def is_deleted(self):
        return self._track.is_deleted

    @property
    def max_age(self):
        return self._track._max_age

    @max_age.setter
    def max_age(self, value):
        self._track._max_age = value

    @property
    def mark_missed(self):
        return self._track.mark_missed

    @property
    def state(self):
        return self._track.state


class DeepSORTTracker(object):

    encoder_model_path = None
    encoder = None

    @classmethod
    def from_config(cls, config):
        return cls(
            distance_metric_type=config.DISTANCE_METRIC_TYPE,
            matching_threshold=config.MATCHING_THRESHOLD,
            nn_budget=config.NN_BUDGET,
            encoder_model_path=config.ENCODER_MODEL_PATH,
            detection_confidence=config.DETECTION_CONFIDENCE,
            nms_max_bbox_overlap=config.NMS_MAX_BBOX_OVERLAP
        )

    def __init__(self, distance_metric_type, matching_threshold, nn_budget,
                 encoder_model_path, detection_confidence, nms_max_bbox_overlap):
        self._distance_metric_type = distance_metric_type
        self._matching_threshold = matching_threshold
        self._nn_budget = nn_budget
        self._tracker = None
        # ------------------------------
        # self._encoder = self._setup_encoder(encoder_model_path)
        # self._encoder_model_path = type(self).encoder_model_path
        # ------------------------------
        self._detection_confidence = detection_confidence
        self._nms_max_bbox_overlap = nms_max_bbox_overlap
        # ----------------------------------------
        self._setup_tracker()

    def update_by_obj_list(self, frame, obj_list):
        if not obj_list:
            return False, []
        # ----------------------------------------
        p1wh_list = [
            (x1, y1, w, h)
            for class_name, score, (x1, y1, x2, y2), (xo, yo, w, h) in obj_list
        ]
        # ----------------------------------------
        feature_array = self._encoder_allone(frame, p1wh_list)
        detection_gen = (Detection(p1wh, self._detection_confidence, feature)
                         for p1wh, feature in zip(p1wh_list, feature_array))
        # ----------------------------------------
        det_p1wh_list = []
        det_score_list = []
        detection_list = []
        for detection in detection_gen:
            det_p1wh_list.append(detection.tlwh)
            det_score_list.append(detection.confidence)
            detection_list.append(detection)
        det_p1wh_array = np.array(det_p1wh_list)
        det_score_array = np.array(det_score_list)
        # ----------------------------------------
        filtered_idx_list = preprocessing.non_max_suppression(
            det_p1wh_array,
            self._nms_max_bbox_overlap,
            det_score_array
        )
        filtered_detection_list = [detection_list[idx] for idx in filtered_idx_list]
        # ----------------------------------------
        self._tracker.predict()
        self._tracker.update(filtered_detection_list)

    def iter_tracks(self):
        for track in self._tracker.tracks:
            yield DeepSORTTrackProxy(track)

    def _setup_tracker(self):
        metric = nn_matching.NearestNeighborDistanceMetric(
            metric=self._distance_metric_type,
            matching_threshold=self._matching_threshold,
            budget=self._nn_budget
        )
        self._tracker = Tracker(metric)

    def _encoder_allone(self,frame,list):
        a = np.ones((len(list),128),dtype=np.float32)
        return a

    @classmethod
    def _setup_encoder(cls, encoder_model_path):
        if not getattr(cls, "encoder"):
            encoder = generate_detections.create_box_encoder(
                model_filename=encoder_model_path,
                batch_size=1,
            )
            cls.encoder = encoder
            cls.encoder_model_path = encoder_model_path
        # ----------------------------------------
        return cls.encoder

