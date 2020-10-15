import copy
from itertools import count
from alphaction.structures.bounding_box import BoxList
import numpy as np
import queue
from tqdm import tqdm

import torch
from alphaction.config import cfg as base_cfg
from alphaction.modeling.detector import build_detection_model
from alphaction.utils.checkpoint import ActionCheckpointer
from alphaction.dataset.transforms import video_transforms as T
from alphaction.dataset.transforms import object_transforms as OT
from alphaction.structures.memory_pool import MemoryPool
from alphaction.dataset.collate_batch import batch_different_videos
from alphaction.utils.IA_helper import has_memory, has_object
from video_detection_loader import VideoDetectionLoader
from detector.apis import get_detector
from bisect import bisect_right
import torch.multiprocessing as mp


def convert_boxlist(maskrcnn_boxlist):
    box_tensor = maskrcnn_boxlist.bbox
    size = maskrcnn_boxlist.size
    mode = maskrcnn_boxlist.mode
    bbox = BoxList(box_tensor, size, mode)
    for field in maskrcnn_boxlist.fields():
        bbox.add_field(field, maskrcnn_boxlist.get_field(field))
    return bbox

class AVAPredictor(object):
    def __init__(
            self,
            cfg_file_path,
            model_weight_url,
            detect_rate,
            common_cate,
            device,
            exclude_class=[],
    ):
        # TODO: add exclude class
        cfg = base_cfg.clone()
        cfg.merge_from_file(cfg_file_path)
        cfg.MODEL.WEIGHT = model_weight_url
        cfg.MODEL.IA_STRUCTURE.MEMORY_RATE *= detect_rate
        if common_cate:
            cfg.MODEL.ROI_ACTION_HEAD.NUM_CLASSES = 15
            cfg.MODEL.ROI_ACTION_HEAD.NUM_PERSON_MOVEMENT_CLASSES = 6
            cfg.MODEL.ROI_ACTION_HEAD.NUM_OBJECT_MANIPULATION_CLASSES = 5
            cfg.MODEL.ROI_ACTION_HEAD.NUM_PERSON_INTERACTION_CLASSES = 4
        cfg.freeze()
        self.cfg = cfg

        self.model = build_detection_model(cfg)
        self.model.eval()
        self.model.to(device)
        self.has_memory = has_memory(cfg.MODEL.IA_STRUCTURE)
        self.mem_len = cfg.MODEL.IA_STRUCTURE.LENGTH
        self.mem_rate = cfg.MODEL.IA_STRUCTURE.MEMORY_RATE
        self.has_object = has_object(cfg.MODEL.IA_STRUCTURE)

        checkpointer = ActionCheckpointer(cfg, self.model)
        self.mem_pool = MemoryPool()
        self.object_pool = MemoryPool()
        self.mem_timestamps = []
        self.obj_timestamps = []
        self.pred_pos = 0
        print("Loading action model weight from {}.".format(cfg.MODEL.WEIGHT))
        _ = checkpointer.load(cfg.MODEL.WEIGHT)
        print("Action model weight successfully loaded.")

        self.transforms, self.person_transforms, self.object_transforms = self.build_transform()

        self.device = device
        self.cpu_device = torch.device("cpu")
        self.exclude_class = exclude_class

    def build_transform(self):
        cfg = self.cfg

        transform = T.Compose(
            [
                T.TemporalCrop(cfg.INPUT.FRAME_NUM, cfg.INPUT.FRAME_SAMPLE_RATE),
                T.Resize(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST),
                T.ToTensor(),
                T.Normalize(
                    mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr=cfg.INPUT.TO_BGR
                ),
                T.SlowFastCrop(cfg.INPUT.TAU, cfg.INPUT.ALPHA, False),
            ]
        )

        person_transforms = OT.Resize()

        object_transform = OT.Compose([
            OT.PickTop(cfg.MODEL.IA_STRUCTURE.MAX_OBJECT),
            OT.Resize(),
        ])

        return transform, person_transforms, object_transform

    def update_feature(self, video_data, boxes, objects, timestamp, transform_randoms):
        """Updates memory features pool and object features pool

        Given the video data, person boxes, and objects boxes, this method update the memory
        features pool and the object features pool with respect to the timestamp. These features
        will be retrieved later for action prediction.

        Args
            video_data(List(Tensor)): The input video data.
            boxes(BoxList): Detected person boxes
            objects(BoxList): Detected object boxes
            timestamp(int): The timestamp of center frame. In seconds
            transform_randoms(dict): The random transforms
        """
        if self.mem_timestamps:
            assert timestamp > self.mem_timestamps[-1], "features are expected to be updated in order."

        slow_clips = batch_different_videos([video_data[0]], self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        fast_clips = batch_different_videos([video_data[1]], self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        slow_clips = slow_clips.to(self.device)
        fast_clips = fast_clips.to(self.device)
        boxes = [self.person_transforms(boxes, transform_randoms).to(self.device)]
        if objects is not None:
            objects = [self.object_transforms(objects, transform_randoms).to(self.device)]

        with torch.no_grad():
            feature = self.model(slow_clips, fast_clips, boxes, objects, part_forward=0)
            person_feature = [ft.to(self.cpu_device) for ft in feature[0]][0]
            if feature[1] is None:
                object_feature = None
            else:
                object_feature = [ft.to(self.cpu_device) for ft in feature[1]][0]

        self.mem_pool["SingleVideo", timestamp] = person_feature
        self.mem_timestamps.append(timestamp)
        if object_feature is not None:
            self.object_pool["SingleVideo", timestamp] = object_feature
            self.obj_timestamps.append(timestamp)

    def check_ready_timestamp(self):
        if self.mem_timestamps:
            last_timestamp = self.mem_timestamps[-1]
            if self.has_memory:
                before, after = self.mem_len
                last_ready = last_timestamp - after * self.mem_rate
                ready_num = bisect_right(self.mem_timestamps, last_ready)
                return ready_num - self.pred_pos
            else:
                return len(self.mem_timestamps) - self.pred_pos
        else:
            return 0

    def clear_feature(self, timestamp=None):
        # this function is usually called after compute_prediction
        # to clear features that will not be used in the future.
        # timestamp should be the consistent with the one parsed into compute_prediction
        # note: after this function is called, predictions for clip with timestamp larger than the argument will be unavailable.
        if timestamp is None:
            self.mem_pool = MemoryPool()
            self.object_pool = MemoryPool()
            self.mem_timestamps = []
            self.obj_timestamps = []
            self.pred_pos = 0
            return

        if self.has_memory:
            before, after = self.mem_len
            last_unused = timestamp - before * self.mem_rate
        else:
            last_unused = timestamp

        mem_to_release = bisect_right(self.mem_timestamps, last_unused)
        for t in self.mem_timestamps[:mem_to_release]:
            del self.mem_pool["SingleVideo", t]
        self.mem_timestamps = self.mem_timestamps[mem_to_release:]
        self.pred_pos -= mem_to_release
        self.pred_pos = max(self.pred_pos, 0)

        obj_to_release = bisect_right(self.obj_timestamps, timestamp)
        for t in self.obj_timestamps[:obj_to_release]:
            del self.object_pool["SingleVideo", t]
        self.obj_timestamps = self.obj_timestamps[obj_to_release:]

    def compute_prediction(self, timestamp, vid_size):
        """Compute the actions score at a timestamp

        Using the previous computed person features and object features to compute
        action scores for each person at given timestamp.
        Note that you should at least update the features of given timestamp before
        using these method. Although this method can be safely used if you only updated
        the given timestamp. The result will be better if you updated more nearby timestamps
        since more memory features will be taken into account.

        Args:
            timestamp(int): The timestamp to be compute. In seconds
            vid_size(tuple): The size of video

        Returns:
            prediction(BoxList): The prediction results with boxes and scores.
        """
        current_feat_p = [self.mem_pool["SingleVideo", timestamp].to(self.device)]
        if ("SingleVideo", timestamp) in self.object_pool:
            current_feat_o = [self.object_pool["SingleVideo", timestamp].to(self.device)]
        else:
            current_feat_o = None

        extras = dict(
            person_pool=self.mem_pool,
            movie_ids=["SingleVideo"],
            timestamps=[timestamp],
            current_feat_p=current_feat_p,
            current_feat_o=current_feat_o,
        )

        with torch.no_grad():
            output = self.model(None, None, None, None, extras=extras, part_forward=1)
            output = [o.resize(vid_size).to(self.cpu_device) for o in output]

        prediction = output[0]

        self.pred_pos += 1

        return prediction

class AVAPredictorWorker(object):
    """Worker class for AVA prediction

    The AVA action predictor need person boxes, object boxes, and a stack of video frames to work.
    Thus, this worker contains three parts.
    coco_det: provide object boxes
    ava_predictor: Given person boxes and object boxes, predict actions for each person
    det_loader: load video data and provide person boxes

    This class will launch a new process for action prediction.
    """
    def __init__(self, cfg):

        self.realtime = cfg.realtime

        # Action Predictor
        cfg_file_path = cfg.cfg_path
        model_weight_url = cfg.weight_path
        self.ava_predictor = AVAPredictor(
            cfg_file_path,
            model_weight_url,
            cfg.detect_rate,
            cfg.common_cate,
            cfg.device,
        )

        # Object Detector
        if self.ava_predictor.has_object:
            object_cfg = copy.deepcopy(cfg)
            object_cfg.detector = "yolo"
            self.coco_det = get_detector(object_cfg)
        else:
            self.coco_det = None

        self.track_queue = mp.Queue(maxsize=1)
        self.input_queue = mp.Queue(maxsize=512)
        self.output_queue = mp.Queue()

        # Video Detection Loader
        self.predictor_process = mp.Value("i", 0)
        det_loader = VideoDetectionLoader(cfg, self.track_queue, self.input_queue, self.predictor_process)
        det_loader.start()

        self.timestamps = []
        self.frame_stack = []
        self.extra_stack = []
        ava_cfg = self.ava_predictor.cfg
        self.frame_buffer_numbers = ava_cfg.INPUT.FRAME_NUM * ava_cfg.INPUT.FRAME_SAMPLE_RATE

        # detection interval should be 1 second like AVA,
        # one reason is that our model with memory feature is trained with that.
        # Since we may not be able to reach 25 fps, so the strategy here is based on the
        # number of frames. The duration of these frames may be varied.
        self.last_milli = 0
        self.detect_rate = cfg.detect_rate
        self.interval = 1000//self.detect_rate

        self.vid_transforms = self.ava_predictor.transforms

        self._stopped = mp.Value('b', False)
        self._task_done = mp.Value('b', False)

        self.prediction_worker = mp.Process(target=self._compute_prediction, args=())
        self.prediction_worker.start()

    def add_task(self, extra, video_size):
        if not self.stopped:
            self.input_queue.put((extra, video_size))

    def terminate(self):
        # end threads
        self._stopped.value = True
        self.predictor_process.value = -1
        # clear queues
        self.stop()

    def stop(self):
        self.prediction_worker.join()
        # clear queues
        self.clear_queues()

    def clear_queues(self):
        self.clear(self.input_queue)

    def clear(self, q):
        while not q.empty():
            try:
                q.get(timeout=1)
            except queue.Empty:
                break
            except FileNotFoundError:
                continue

    def read(self):
        '''
        Read action detection results
        '''
        if self.stopped:
            return None
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None

    def read_track(self):
        '''
        Read tracking results
        '''
        if not self.stopped:
            return self.track_queue.get()

    def compute_prediction(self):
        assert self.realtime == False, "AVAPredictorWorker.compute_prediction() can not be used in realtime mode"
        self._task_done.value = True

    def _compute_prediction(self):
        '''The main loop of action prediction worker

        The main task of this separate process is compute the action score.
        However it behaves differently depends on whether it is in realtime mode.
        In realtime mode, it will compute the action scores right after the feature update.
        In video mode, the prediction won't be done until an explicit call of compute_prediction()
        '''

        empty_flag = False
        pred_num_cnt = 0

        for i in count():
            if self.stopped:
                # tqdm.write("Avaworker stopped")
                return
            # if all video data have been processed and compute_prediction() has been called
            # compute predictions
            if self.task_done == True and empty_flag:
                tqdm.write("Feature extraction finished. Now showing action prediction progress bar [ ready point count / total prediction point ]")
                for center_timestamp, video_size, ids in tqdm(self.timestamps[pred_num_cnt:], initial=pred_num_cnt, total=len(self.timestamps), desc="Action Prediction"):
                    feature_index = center_timestamp // self.interval
                    predictions = self.ava_predictor.compute_prediction(feature_index, video_size)
                    self.output_queue.put((predictions, center_timestamp, ids))
                    self.ava_predictor.clear_feature(feature_index)
                self.ava_predictor.clear_feature()
                tqdm.write("Action prediction is done.")
                self.output_queue.put("done")
                self._task_done.value = False

            try:
                extra, video_size = self.input_queue.get(timeout=1)
            except queue.Empty:
                continue

            if extra == "Done":
                empty_flag = True
                self.predictor_process.value = -1
                continue

            frame, cur_millis, boxes, scores, ids = extra


            self.frame_stack.append(frame)
            self.extra_stack.append((cur_millis, boxes, scores, ids))
            self.frame_stack = self.frame_stack[-self.frame_buffer_numbers:]
            self.extra_stack = self.extra_stack[-self.frame_buffer_numbers:]

            # Predict action once per interval
            if len(self.frame_stack) >= self.frame_buffer_numbers and cur_millis > self.last_milli + self.interval:
                if not self.realtime:
                    self.predictor_process.value = int(cur_millis)
                self.last_milli = cur_millis
                frame_arr = np.stack(self.frame_stack)[..., ::-1]
                center_index = self.frame_buffer_numbers // 2
                center_timestamp, person_boxes, person_scores, person_ids = self.extra_stack[center_index]
                if person_boxes is None or len(person_boxes) == 0:
                    continue

                if self.coco_det is not None:
                    kframe = self.frame_stack[center_index]
                    center_timestamp = int(center_timestamp)

                    kframe_data = self.coco_det.image_preprocess(kframe)
                    im_dim_list_k = kframe.shape[1], kframe.shape[0]
                    im_dim_list_k = torch.FloatTensor(im_dim_list_k).repeat(1, 2)
                    dets = self.coco_det.images_detection(kframe_data, im_dim_list_k)
                    if isinstance(dets, int) or dets.shape[0] == 0:
                        obj_boxes = torch.zeros((0,4))
                    else:
                        obj_boxes = dets[:, 1:5].cpu()
                    obj_boxes = BoxList(obj_boxes, video_size, "xyxy").clip_to_image()
                else:
                    obj_boxes = None

                video_data, _, transform_randoms = self.vid_transforms(frame_arr, None)

                person_box = BoxList(person_boxes, video_size, "xyxy").clip_to_image()

                feature_index = center_timestamp // self.interval

                self.ava_predictor.update_feature(video_data,
                                                  person_box,
                                                  obj_boxes,
                                                  feature_index,
                                                  transform_randoms)

                if self.realtime:
                    predictions = self.ava_predictor.compute_prediction(feature_index, video_size)
                    #print(len(predictions.get_field("scores")), person_ids)
                    self.output_queue.put((predictions, center_timestamp, person_ids[:, 0]))
                    self.ava_predictor.clear_feature(feature_index)
                    pred_num_cnt += 1
                else:
                    # if not realtime, timestamps will be saved and the predictions will be computed later.
                    self.timestamps.append((center_timestamp, video_size, person_ids[:, 0]))
                    ready_num = self.ava_predictor.check_ready_timestamp()
                    for timestamp_idx in range(pred_num_cnt, pred_num_cnt + ready_num):
                        center_timestamp, video_size, ids = self.timestamps[timestamp_idx]
                        feature_index = center_timestamp // self.interval
                        predictions = self.ava_predictor.compute_prediction(feature_index, video_size)
                        self.output_queue.put((predictions, center_timestamp, ids))
                        self.ava_predictor.clear_feature(feature_index)
                    pred_num_cnt = pred_num_cnt + ready_num

    @property
    def stopped(self):
        return self._stopped.value

    @property
    def task_done(self):
        return self._task_done.value
