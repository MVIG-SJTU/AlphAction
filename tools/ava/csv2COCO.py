import pandas as pd
import json
from PIL import Image
import time
import os
from tqdm import tqdm
import itertools
import argparse


def csv2COCOJson(csv_path, movie_list, img_root, json_path, min_json_path):
    ann_df = pd.read_csv(csv_path, header=None)

    movie_ids = {}
    with open(movie_list) as movief:
        for idx, line in enumerate(movief):
            name = line[:line.find('.')]
            movie_ids[name] = idx

    movie_infos = {}
    iter_num = len(ann_df)
    for rows in tqdm(ann_df.itertuples(), total=iter_num, desc='Calculating  info'):
        _, movie_name, timestamp, x1, y1, x2, y2, action_id, person_id = rows
        if movie_name not in movie_infos:
            movie_infos[movie_name] = {}
            movie_infos[movie_name]['img_infos'] = {}
            img_path = os.path.join(movie_name, '{}.jpg'.format(timestamp))
            movie_infos[movie_name]['size'] = Image.open(os.path.join(img_root, img_path)).size
            movie_info = movie_infos[movie_name]
            img_infos = movie_info['img_infos']
            width, height = movie_info['size']
            movie_id = movie_ids[movie_name] * 10000
            for tid in range(902, 1799):
                img_id = movie_id + tid
                img_path = os.path.join(movie_name, '{}.jpg'.format(tid))
                video_path = os.path.join(movie_name, '{}.mp4'.format(tid))
                img_infos[tid] = {
                    'id': img_id,
                    'img_path': img_path,
                    'video_path': video_path,
                    'height': height,
                    'width': width,
                    'movie': movie_name,
                    'timestamp': tid,
                    'annotations': {},
                }
        img_info = movie_infos[movie_name]['img_infos'][timestamp]
        if person_id not in img_info['annotations']:
            box_id = img_info['id'] * 1000 + person_id
            box_w, box_h = x2 - x1, y2 - y1
            width = img_info['width']
            height = img_info['height']
            real_x1, real_y1 = x1 * width, y1 * height
            real_box_w, real_box_h = box_w * width, box_h * height
            area = real_box_w * real_box_h
            img_info['annotations'][person_id] = {
                'id': box_id,
                'image_id': img_info['id'],
                'category_id': 1,
                'action_ids': [],
                'person_id': person_id,
                'bbox': list(map(lambda x: round(x, 2), [real_x1, real_y1, real_box_w, real_box_h])),
                'area': round(area, 5),
                'keypoints': [],
                'iscrowd': 0,
            }
        box_info = img_info['annotations'][person_id]
        box_info['action_ids'].append(action_id)

    tic = time.time()
    print("Writing into json file...")
    jsondata = {}
    jsondata['categories'] = [{'supercategory': 'person',
                               'id': 1,
                               'name': 'person'}]

    anns = [img_info.pop('annotations').values() for movie_info in movie_infos.values() for img_info in
            movie_info['img_infos'].values()]
    anns = list(itertools.chain.from_iterable(anns))
    jsondata['annotations'] = anns
    imgs = [movie_info['img_infos'].values() for movie_info in movie_infos.values()]
    imgs = list(itertools.chain.from_iterable(imgs))
    jsondata['images'] = imgs
    with open(json_path, 'w') as jsonf:
        json.dump(jsondata, jsonf, indent=4)
    print("Write json dataset into json file {} successfully.".format(json_path))
    with open(min_json_path, 'w') as jsonminf:
        json.dump(jsondata, jsonminf)
    print("Write json dataset with no indent into json file {} successfully.".format(min_json_path))
    print('Done (t={:0.2f}s)'.format(time.time() - tic))


def genCOCOJson(movie_list, img_root, json_path, min_json_path):
    movie_ids = {}
    with open(movie_list) as movief:
        for idx, line in enumerate(movief):
            name = line[:line.find('.')]
            movie_ids[name] = idx

    movie_infos = {}

    for movie_name in tqdm(movie_ids):
        movie_infos[movie_name] = {}
        movie_infos[movie_name]['img_infos'] = {}
        img_path = os.path.join(movie_name, '902.jpg')
        movie_infos[movie_name]['size'] = Image.open(os.path.join(img_root, img_path)).size
        movie_info = movie_infos[movie_name]
        img_infos = movie_info['img_infos']
        width, height = movie_info['size']
        movie_id = movie_ids[movie_name] * 10000
        for tid in range(902, 1799):
            img_id = movie_id + tid
            img_path = os.path.join(movie_name, '{}.jpg'.format(tid))
            video_path = os.path.join(movie_name, '{}.mp4'.format(tid))
            img_infos[tid] = {
                'id': img_id,
                'img_path': img_path,
                'video_path': video_path,
                'height': height,
                'width': width,
                'movie': movie_name,
                'timestamp': tid,
                'annotations': {},
            }

    tic = time.time()
    print("Writing into json file...")
    jsondata = {}
    jsondata['categories'] = [{'supercategory': 'person',
                               'id': 1,
                               'name': 'person'}]

    anns = [img_info.pop('annotations').values() for movie_info in movie_infos.values() for img_info in
            movie_info['img_infos'].values()]
    anns = list(itertools.chain.from_iterable(anns))
    jsondata['annotations'] = anns
    imgs = [movie_info['img_infos'].values() for movie_info in movie_infos.values()]
    imgs = list(itertools.chain.from_iterable(imgs))
    jsondata['images'] = imgs
    with open(json_path, 'w') as jsonf:
        json.dump(jsondata, jsonf, indent=4)
    print("Write json dataset into json file {} successfully.".format(json_path))
    with open(min_json_path, 'w') as jsonminf:
        json.dump(jsondata, jsonminf)
    print("Write json dataset with no indent into json file {} successfully.".format(min_json_path))
    print('Done (t={:0.2f}s)'.format(time.time() - tic))


def main():
    parser = argparse.ArgumentParser(description="Generate coco format json for AVA.")
    parser.add_argument(
        "--csv_path",
        default="",
        help="path to csv file",
        type=str,
    )
    parser.add_argument(
        "--movie_list",
        required=True,
        help="path to movie list",
        type=str,
    )
    parser.add_argument(
        "--img_root",
        required=True,
        help="root directory of extracted key frames",
        type=str,
    )
    parser.add_argument(
        "--json_path",
        default="",
        help="path of output json",
        type=str,
    )
    parser.add_argument(
        "--min_json_path",
        default="",
        help="path of output minimized json",
        type=str,
    )
    args = parser.parse_args()

    if args.json_path=="":
        if args.csv_path == "":
            json_path = "test.json"
        else:
            dotpos = args.csv_path.rfind('.')
            if dotpos < 0:
                csv_name = args.csv_path
            else:
                csv_name = args.csv_path[:dotpos]
            json_path = csv_name + '.json'
    else:
        json_path = args.json_path

    if args.min_json_path=="":
        dotpos = json_path.rfind('.')
        if dotpos < 0:
            json_name = json_path
        else:
            json_name = json_path[:dotpos]
        min_json_path = json_name + '_min.json'
    else:
        min_json_path = args.min_json_path

    if args.csv_path == "":
        genCOCOJson(args.movie_list, args.img_root, json_path, min_json_path)
    else:
        csv2COCOJson(args.csv_path, args.movie_list, args.img_root, json_path, min_json_path)

if __name__ == '__main__':
    main()