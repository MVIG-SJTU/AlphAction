"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "data"
    DATASETS = {
        "ava_video_train_v2.2": {
            "video_root": "AVA/clips/trainval",
            "ann_file": "AVA/annotations/ava_train_v2.2_min.json",
            "box_file": "",
            "eval_file_paths": {
                "csv_gt_file": "AVA/annotations/ava_train_v2.2.csv",
                "labelmap_file": "AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
                "exclusion_file": "AVA/annotations/ava_train_excluded_timestamps_v2.2.csv",
            },
            "object_file": "AVA/boxes/ava_train_det_object_bbox.json",
        },
        "ava_video_val_v2.2": {
            "video_root": "AVA/clips/trainval",
            "ann_file": "AVA/annotations/ava_val_v2.2_min.json",
            "box_file": "AVA/boxes/ava_val_det_person_bbox.json",
            "eval_file_paths": {
                "csv_gt_file": "AVA/annotations/ava_val_v2.2.csv",
                "labelmap_file": "AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
                "exclusion_file": "AVA/annotations/ava_val_excluded_timestamps_v2.2.csv",
            },
            "object_file": "AVA/boxes/ava_val_det_object_bbox.json",
        },
    }

    @staticmethod
    def get(name):
        if "ava_video" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            if attrs["box_file"]=="":
                box_file = ""
            else:
                box_file = os.path.join(data_dir, attrs["box_file"])
            args = dict(
                video_root=os.path.join(data_dir, attrs["video_root"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
                box_file=box_file,
                eval_file_paths={key: os.path.join(data_dir, attrs["eval_file_paths"][key]) for key in
                                 attrs["eval_file_paths"]},
                object_file=os.path.join(data_dir, attrs["object_file"]),
            )
            return dict(
                factory="AVAVideoDataset",
                args=args
            )
        raise RuntimeError("Dataset not available: {}".format(name))