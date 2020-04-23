import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw
import torch.multiprocessing as mp

cv2.setNumThreads(0)

def cv2_video_info(video_path):
    vid = cv2.VideoCapture(video_path)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = vid.get(cv2.CAP_PROP_FPS)
    frame_num = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    vid.release()
    return dict(
        width=int(width),
        height=int(height),
        fps=fps,
        frame_num=int(frame_num),
    )


class AVAVisualizer(object):
    # category names are modified for better visualization
    CATEGORIES = [
        "bend/bow",
        "crawl", #
        "crouch/kneel",
        "dance",
        "fall down",
        "get up",
        "jump/leap",
        "lie/sleep",
        "martial art",
        "run/jog",
        "sit",
        "stand",
        "swim",
        "walk",
        "answer phone",
        "brush teeth", #
        "carry/hold sth.",
        "catch sth.", #
        "chop", #
        "climb",
        "clink glass", #
        "close",
        "cook", #
        "cut",
        "dig", #
        "dress/put on clothing",
        "drink",
        "drive",
        "eat",
        "enter",
        "exit", #
        "extract", #
        "fishing", #
        "hit sth.",
        "kick sth.", #
        "lift/pick up",
        "listen to sth.",
        "open",
        "paint", #
        "play board game", #
        "play musical instrument",
        "play with pets", #
        "point to sth.",
        "press", #
        "pull sth.",
        "push sth.",
        "put down",
        "read",
        "ride",
        "row boat", #
        "sail boat",
        "shoot",
        "shovel", #
        "smoke",
        "stir", #
        "take a photo",
        "look at a cellphone",
        "throw",
        "touch sth.",
        "turn",
        "watch screen",
        "work on a computer",
        "write",
        "fight/hit sb.",
        "give/serve sth. to sb.",
        "grab sb.",
        "hand clap",
        "hand shake",
        "hand wave",
        "hug sb.",
        "kick sb.", #
        "kiss sb.",
        "lift sb.",
        "listen to sb.",
        "play with kids", #
        "push sb.",
        "sing",
        "take sth. from sb.",
        "talk",
        "watch sb.",
    ]
    COMMON_CATES = [
        'dance',
        'run/jog',
        'sit',
        'stand',
        'swim',
        'walk',
        'answer phone',
        'carry/hold sth.',
        'drive',
        'play musical instrument',
        'ride',
        'fight/hit sb.',
        'listen to sb.',
        'talk',
        'watch sb.'
    ]
    EXCLUSION = [
        "crawl",
        "brush teeth",
        "catch sth.",
        "chop",
        "clink glass",
        "cook",
        "dig",
        "exit",
        "extract",
        "fishing",
        "kick sth.",
        "paint",
        "play board game",
        "play with pets",
        "press",
        "row boat",
        "shovel",
        "stir",
        "kick sb.",
        "play with kids",
    ]
    def __init__(
            self,
            video_path,
            output_path,
            realtime,
            start,
            duration,
            show_time,
            confidence_threshold=0.5,
            exclude_class=None,
            common_cate=False,
    ):
        self.vid_info = cv2_video_info(video_path)
        fps = self.vid_info["fps"]
        if fps == 0 or fps > 100:
            print(
                "Warning: The detected frame rate {} could be wrong. The behavior of this demo code can be abnormal.".format(
                    fps))

        self.realtime = realtime
        self.start = start
        self.duration = duration
        self.show_time =  show_time
        self.confidence_threshold = confidence_threshold
        if common_cate:
            self.cate_to_show = self.COMMON_CATES
            self.category_split = (6, 11)
        else:
            self.cate_to_show = self.CATEGORIES
            self.category_split = (14, 63)
        self.cls2label = {class_name: i for i, class_name in enumerate(self.cate_to_show)}
        if exclude_class is None:
            exclude_class = self.EXCLUSION
        self.exclude_id = [self.cls2label[cls_name] for cls_name in exclude_class if cls_name in self.cls2label]

        self.width = self.vid_info["width"]
        self.height = self.vid_info["height"]
        long_side = min(self.width, self.height)
        self.font_size = max(int(round((long_side / 40))), 1)
        self.box_width = max(int(round(long_side / 180)), 1)
        self.font = ImageFont.truetype("./Roboto-Bold.ttf", self.font_size)

        self.box_color = (191, 40, 41)
        self.category_colors = ((176, 85, 234), (87, 118, 198), (52, 189, 199))
        self.category_trans = int(0.6 * 255)

        self.action_dictionary = dict()

        if realtime:
            # Output Video
            width = self.vid_info["width"]
            height = self.vid_info["height"]
            fps = self.vid_info["fps"]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.out_vid = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            self.frame_queue = mp.JoinableQueue(512)
            self.result_queue = mp.JoinableQueue()
            self.track_queue = mp.JoinableQueue()
            self.done_queue = mp.Queue()
            self.frame_loader = mp.Process(target=self._load_frame, args=(video_path,))
            self.frame_loader.start()
            self.video_writer = mp.Process(target=self._wirte_frame, args=(output_path,))
            self.video_writer.start()

    def realtime_write_frame(self, result, orig_img, boxes, scores, ids):
        orig_img = orig_img[:, :, ::-1]

        if result is not None:
            result, timestamp, result_ids = result
            update_boxes = result.bbox
            update_scores = result.get_field("scores")
            update_ids = result_ids
            if update_boxes is not None:
                self.update_action_dictionary(update_scores, update_ids)

        if boxes is not None:
            last_visual_mask = self.visual_result(boxes, ids)
            orig_img = self.visual_frame(orig_img, last_visual_mask)

        cv2.imshow("my webcam", orig_img)
        self.out_vid.write(orig_img)

        if cv2.waitKey(1) == 27:
            return False
        return True

    def _load_frame(self, video_path):
        vid = cv2.VideoCapture(video_path)
        vid.set(cv2.CAP_PROP_POS_MSEC, self.start)
        vid_avail = True
        while True:
            vid_avail, frame = vid.read()
            if not vid_avail:
                break
            mills = vid.get(cv2.CAP_PROP_POS_MSEC)
            if self.duration != -1 and mills > self.start + self.duration:
                break
            self.frame_queue.put((frame, mills))

        vid.release()
        self.frame_queue.put("DONE")
        self.frame_queue.join()
        self.frame_queue.close()
        print("load frame closed")

    def _wirte_frame(self, output_path):
        width = self.vid_info["width"]
        height = self.vid_info["height"]
        fps = self.vid_info["fps"]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_vid = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        has_frame = True

        result = self.result_queue.get()
        timestamp = float('inf')
        result_ids = None
        if not isinstance(result, str):
            result, timestamp, result_ids = result
        while has_frame:
            track_result = self.track_queue.get()

            # read frame
            data = self.frame_queue.get()
            self.frame_queue.task_done()

            if isinstance(result, str) and data == "DONE":
                self.track_queue.task_done()
                self.result_queue.task_done()
                break

            # note that the timestamp should be in milliseconds
            frame, mills = data

            if self.show_time:
                frame = self.visual_timestampe(frame, mills)
            if mills - timestamp + 0.5 > 0:
                # print("renew action_dict:{}".format(self.action_dictionary))
                boxes = result.bbox
                scores = result.get_field("scores")
                ids = result_ids

                self.result_queue.task_done()
                result = self.result_queue.get()
                if not isinstance(result, str):
                    result, timestamp, result_ids = result
                else:
                    timestamp = float('inf')
            else:
                boxes, ids = track_result
                scores = None

            if boxes is not None:
                self.update_action_dictionary(scores, ids)
                last_visual_mask = self.visual_result(boxes, ids)
                new_frame = self.visual_frame(frame, last_visual_mask)
                out_vid.write(new_frame)
            else:
                out_vid.write(frame)

            self.track_queue.task_done()
            self.done_queue.put(True)

        out_vid.release()
        print("write frame closed")

    def hou_min_sec(self, total_millis):
        total_millis = int(total_millis)
        millis = total_millis % 1000
        total_millis /= 1000
        seconds = total_millis % 60
        total_millis /= 60
        minutes = total_millis % 60
        total_millis /= 60
        hours = total_millis
        return ("%02d:%02d:%02d.%03d" % (hours, minutes, seconds, millis))

    def visual_timestampe(self, frame, mills):
        time_text = self.hou_min_sec(mills)
        img = Image.fromarray(frame[..., ::-1])
        img = img.convert("RGBA")

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        trans_draw = ImageDraw.Draw(overlay)
        text_width, text_height = trans_draw.textsize(time_text, font=self.font)
        width_pad = max(self.font_size // 2, 1)
        rec_height = int(round(1.8 * text_height))
        height_pad = round((rec_height - text_height) / 2)

        r_x1 = 0
        r_y2 = img.height
        r_x2 = r_x1 + text_width + width_pad * 2
        r_y1 = r_y2 - rec_height
        rec_pos = (r_x1, r_y1, r_x2, r_y2)
        text_pos = (r_x1 + width_pad, r_y1 + height_pad)

        trans_draw.rectangle(rec_pos, fill=(0, 0, 0, self.category_trans))
        trans_draw.text(text_pos, time_text, fill=(255, 255, 255, self.category_trans), font=self.font, align="center")

        img = Image.alpha_composite(img, overlay)

        img = img.convert("RGB")

        return np.array(img)[..., ::-1]

    def update_action_dictionary(self, scores, ids):
        # Update action_dictionary
        if scores is not None:
            for score, id in zip(scores, ids):
                show_idx = torch.nonzero(score >= self.confidence_threshold).squeeze(1)
                captions = []
                bg_colors = []

                #captions.append("id: {}".format(int(id)))
                #bg_colors.append(0)

                for category_id in show_idx:
                    if category_id in self.exclude_id:
                        continue
                    label = self.cate_to_show[category_id]
                    conf = " %.2f" % score[category_id]
                    caption = label + conf
                    captions.append(caption)
                    if category_id < self.category_split[0]:
                        bg_colors.append(0)
                    elif category_id < self.category_split[1]:
                        bg_colors.append(1)
                    else:
                        bg_colors.append(2)

                self.action_dictionary[int(id)] = {
                    "captions": captions,
                    "bg_colors": bg_colors,
                }

    def visual_result(self, boxes, ids):
        bboxes = boxes
        ids = ids

        result_vis = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(result_vis)

        for box in bboxes:
            draw.rectangle(box.tolist(), outline=self.box_color + (255,), width=self.box_width)

        for box, id in zip(bboxes, ids):
            caption_and_color = self.action_dictionary.get(int(id), None)

            if caption_and_color is None:
                captions = []
                bg_colors = []
            else:
                captions = caption_and_color['captions']
                bg_colors = caption_and_color['bg_colors']

            if len(captions) == 0:
                continue
            x1, y1, x2, y2 = box.tolist()
            overlay = Image.new("RGBA", result_vis.size, (0, 0, 0, 0))
            trans_draw = ImageDraw.Draw(overlay)
            caption_sizes = [trans_draw.textsize(caption, font=self.font) for caption in captions]
            caption_widths, caption_heights = list(zip(*caption_sizes))
            max_height = max(caption_heights)
            rec_height = int(round(1.8 * max_height))
            space_height = int(round(0.2 * max_height))
            total_height = (rec_height + space_height) * (len(captions) - 1) + rec_height
            width_pad = max(self.font_size // 2, 1)
            start_y = max(round(y1) - total_height, space_height)

            for i, caption in enumerate(captions):
                r_x1 = round(x1)
                r_y1 = start_y + (rec_height + space_height) * i
                r_x2 = r_x1 + caption_widths[i] + width_pad * 2
                r_y2 = r_y1 + rec_height
                rec_pos = (r_x1, r_y1, r_x2, r_y2)

                height_pad = round((rec_height - caption_heights[i]) / 2)
                text_pos = (r_x1 + width_pad, r_y1 + height_pad)

                trans_draw.rectangle(rec_pos, fill=self.category_colors[bg_colors[i]] + (self.category_trans,))
                trans_draw.text(text_pos, caption, fill=(255, 255, 255, self.category_trans), font=self.font,
                                align="center")

            result_vis = Image.alpha_composite(result_vis, overlay)

        return result_vis

    def visual_frame(self, frame, visual_mask):
        img = Image.fromarray(frame[..., ::-1])
        img = img.convert("RGBA")
        img = Image.alpha_composite(img, visual_mask)

        img = img.convert("RGB")

        return np.array(img)[..., ::-1]

    def visual_frame_old(self, frame, result):
        bboxes = result.bbox
        scores = result.get_field("scores")
        img = Image.fromarray(frame[..., ::-1])
        img = img.convert("RGBA")

        draw = ImageDraw.Draw(img)
        for box in bboxes:
            draw.rectangle(box.tolist(), outline=self.box_color + (255,), width=self.box_width)

        for box, score in zip(bboxes, scores):
            show_idx = torch.nonzero(score >= self.confidence_threshold).squeeze(1)
            captions = []
            bg_colors = []
            for category_id in show_idx:
                if category_id in self.exclude_id:
                    continue
                label = self.cate_to_show[category_id]
                conf = " %.2f" % score[category_id]
                caption = label + conf
                captions.append(caption)
                if category_id <= self.category_split[0]:
                    bg_colors.append(0)
                elif category_id <= self.category_split[1]:
                    bg_colors.append(1)
                else:
                    bg_colors.append(2)

            x1, y1, x2, y2 = box.tolist()
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            trans_draw = ImageDraw.Draw(overlay)
            caption_sizes = [trans_draw.textsize(caption, font=self.font) for caption in captions]
            caption_widths, caption_heights = list(zip(*caption_sizes))
            max_height = max(caption_heights)
            rec_height = int(round(1.8 * max_height))
            space_height = int(round(0.2 * max_height))
            total_height = (rec_height + space_height) * (len(captions) - 1) + rec_height
            width_pad = max(self.font_size // 2, 1)
            start_y = max(round(y1) - total_height, space_height)

            for i, caption in enumerate(captions):
                r_x1 = round(x1)
                r_y1 = start_y + (rec_height + space_height) * i
                r_x2 = r_x1 + caption_widths[i] + width_pad * 2
                r_y2 = r_y1 + rec_height
                rec_pos = (r_x1, r_y1, r_x2, r_y2)

                height_pad = round((rec_height - caption_heights[i]) / 2)
                text_pos = (r_x1 + width_pad, r_y1 + height_pad)

                trans_draw.rectangle(rec_pos, fill=self.category_colors[bg_colors[i]] + (self.category_trans,))
                trans_draw.text(text_pos, caption, fill=(255, 255, 255, self.category_trans), font=self.font,
                                align="center")

            img = Image.alpha_composite(img, overlay)

        img = img.convert("RGB")

        return np.array(img)[..., ::-1]

    def send(self, result):
        self.result_queue.put(result)

    def send_track(self, result):
        self.track_queue.put(result)

    def close(self):
        if self.realtime:
            self.out_vid.release()
        else:
            self.result_queue.join()
            self.result_queue.close()

            self.track_queue.join()
            self.track_queue.close()

    def progress_bar(self, total):
        # get initial
        cnt = 0
        while not self.done_queue.empty():
            _ = self.done_queue.get()
            cnt += 1
        pbar = tqdm(total=total, initial=cnt)
        # update bar
        while cnt < total:
            _ = self.done_queue.get()
            cnt += 1
            pbar.update(1)
        # close bar
        pbar.close()
