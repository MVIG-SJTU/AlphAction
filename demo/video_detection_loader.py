from itertools import count
from threading import Thread
from queue import Queue

import cv2
import numpy as np
from time import sleep
from tqdm import tqdm
import queue

import torch
import torch.multiprocessing as mp
from torchvision.transforms import functional as F

from detector.apis import get_detector

class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image


class VideoDetectionLoader(object):
    '''
    This Class takes the video from the source (video file or camera) and tracks the person.
    '''
    def __init__(self, cfg, track_queue, action_queue, predictor_process):
        self.detector = get_detector(cfg)
        self.input_path = cfg.input_path

        self.start_mill = cfg.start
        self.duration_mill = cfg.duration
        self.realtime = cfg.realtime

        stream = cv2.VideoCapture(self.input_path)
        assert stream.isOpened(), 'Cannot capture source'
        self.fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
        self.fps = stream.get(cv2.CAP_PROP_FPS)
        self.frameSize = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.videoinfo = {'fourcc': self.fourcc, 'fps': self.fps, 'frameSize': self.frameSize}
        stream.release()

        self._stopped = mp.Value('b', False)
        self.track_queue = track_queue
        self.action_queue = action_queue
        self.predictor_process = predictor_process

    def start_worker(self, target):
        p = mp.Process(target=target, args=())
        p.start()
        return p

    def start(self):
        # start a thread to pre process images for object detection
        self.image_preprocess_worker = self.start_worker(self.frame_preprocess)
        return self

    def stop(self):
        # end threads
        self.image_preprocess_worker.join()
        # clear queues
        self.clear_queues()

    def terminate(self):
        self._stopped.value = True
        self.stop()

    def clear_queues(self):
        self.clear(self.track_queue)
        self.clear(self.action_queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def wait_and_put(self, queue, item):
        if not self.stopped:
            queue.put(item)

    def wait_and_get(self, queue):
        if not self.stopped:
            return queue.get()
    
    def wait_till_empty(self, queue):
        if not queue.empty():
            number_of_items = queue.qsize()
            print("{} item(s) to be processed".format(number_of_items))
            rest = number_of_items
            for i in tqdm(range(number_of_items)):
                if rest + i < number_of_items:
                    continue
                else:
                    rest = queue.qsize()
                    sleep(0.1)

            while True:
                if queue.empty():
                    print("Process completed")
                    return
                else:
                    sleep(0.1)

    def frame_preprocess(self):
        stream = cv2.VideoCapture(self.input_path)
        assert stream.isOpened(), 'Cannot capture source'

        if not self.realtime:
            stream.set(cv2.CAP_PROP_POS_MSEC, self.start_mill)

        cur_millis = 0

        # keep looping infinitely
        for i in count():
            if self.stopped or (self.realtime and self.predictor_process.value == -1):
                stream.release()
                print("Video detection loader stopped")
                return
            if not self.track_queue.full():
                # otherwise, ensure the queue has room in it
                # The frame is in BGR format
                (grabbed, frame) = stream.read()
                last_millis = cur_millis
                cur_millis = stream.get(cv2.CAP_PROP_POS_MSEC)

                if not self.realtime and self.duration_mill != -1 and cur_millis > self.start_mill + self.duration_mill:
                    grabbed = False

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.wait_and_put(self.track_queue, (None, None, None, None))
                    self.wait_and_put(self.action_queue, ("Done", self.videoinfo["frameSize"]))
                    #self.wait_till_empty(self.action_queue)
                    #self.wait_till_empty(self.track_queue)
                    print("Wait for feature preprocess")

                    # This process needs to be finished after the predictor process
                    # Otherwise, it will cause FileNotFoundError, if predictor is
                    # overwhelmed
                    cur_process = 0
                    for j in tqdm(range(int(last_millis) - self.start_mill)):
                        if self.stopped:
                            break

                        if j <= cur_process:
                            continue
                        else:
                            predictor_process = self.predictor_process.value
                            if predictor_process == -1:
                                cur_process = last_millis + 1 - self.start_mill
                            else:
                                cur_process = predictor_process - self.start_mill
                            sleep(0.1)

                    stream.release()
                    print("End of video loader")
                    return

                # expected frame shape like (1,3,h,w) or (3,h,w)
                img_k = self.detector.image_preprocess(frame)

                if isinstance(img_k, np.ndarray):
                    img_k = torch.from_numpy(img_k)
                # add one dimension at the front for batch if image shape (3,h,w)
                if img_k.dim() == 3:
                    img_k = img_k.unsqueeze(0)

                im_dim_list_k = frame.shape[1], frame.shape[0]

                orig_img = frame[:, :, ::-1]
                im_name = str(i) + '.jpg'

                with torch.no_grad():
                    # Record original image resolution
                    im_dim_list_k = torch.FloatTensor(im_dim_list_k).repeat(1, 2)
                img_det = self.image_detection((img_k, orig_img, im_name, im_dim_list_k))
                self.image_postprocess(img_det, (frame, cur_millis))

    def image_detection(self, inputs):
        img, orig_img, im_name, im_dim_list = inputs
        if img is None or self.stopped:
            return (None, None, None, None)

        with torch.no_grad():
            dets = self.detector.images_detection(img, im_dim_list)
            if isinstance(dets, int) or dets.shape[0] == 0:
                return (orig_img, None, None, None)
            if isinstance(dets, np.ndarray):
                dets = torch.from_numpy(dets)
            dets = dets.cpu()
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]
            ids = dets[:, 6:7]

        boxes_k = boxes[dets[:, 0] == 0]
        if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
            return (orig_img, None, None, None)

        return (orig_img, boxes_k, scores[dets[:, 0] == 0], ids[dets[:, 0] == 0])

    def image_postprocess(self, inputs, extra):
        with torch.no_grad():
            (orig_img, boxes, scores, ids) = inputs
            if orig_img is None or self.stopped:
                self.wait_and_put(self.track_queue, (None, None, None, None))
                return

            # all parameters to be used in ava
            frame, cur_millis = extra
            input = (frame, cur_millis, boxes, scores, ids)

            # Passing these information to AVAPredictorWorker
            self.action_queue.put((input, self.videoinfo["frameSize"]))

            # Only return the tracking results to main thread
            self.wait_and_put(self.track_queue, (orig_img, boxes, scores, ids))

    def read_track(self):
        return self.wait_and_get(self.track_queue)

    def read_action(self):
        return self.wait_and_get(self.action_queue)

    @property
    def stopped(self):
        return self._stopped.value

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]
