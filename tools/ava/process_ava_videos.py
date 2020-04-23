import os
import json
from multiprocessing import Pool
import tqdm
import argparse
import subprocess


def slice_movie_yuv(movie_path, clip_root, midframe_root="", start_sec=895, end_sec=1805, targ_fps=25, targ_size=360):
    # targ_fps should be int
    probe_args = ["ffprobe", "-show_format", "-show_streams", "-of", "json", movie_path]
    p = subprocess.Popen(probe_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        return "Message from {}:\nffprobe error!".format(movie_path)
    video_stream = json.loads(out.decode("utf8"))["streams"][0]
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    if min(width, height) <= targ_size:
        new_width, new_height = width, height
    else:
        if height > width:
            new_width = targ_size
            new_height = int(round(new_width * height / width / 2) * 2)
        else:
            new_height = targ_size
            new_width = int(round(new_height * width / height / 2) * 2)
    vid_name = os.path.basename(movie_path)
    vid_id = vid_name[:vid_name.find('.')]
    targ_dir = os.path.join(clip_root, vid_id)
    os.makedirs(targ_dir, exist_ok=True)
    if midframe_root != "":
        frame_targ_dir = os.path.join(midframe_root, vid_id)
        os.makedirs(frame_targ_dir, exist_ok=True)

    args1_input = ['ffmpeg', '-ss', str(start_sec), '-t', str(end_sec - start_sec + 1), '-i', movie_path]
    filter_args = "fps=fps={}".format(targ_fps)
    if min(width, height) > targ_size:
        filter_args = 'scale={}:{}, '.format(new_width, new_height) + filter_args
    args1_output = ['-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-filter:v', filter_args, 'pipe:']
    args1 = args1_input + args1_output
    stdout_stream = subprocess.PIPE
    total_err_file = os.path.join(targ_dir, "decode.err")
    total_err_f_obj = open(total_err_file, "wb")
    process1 = subprocess.Popen(
        args1, stdout=stdout_stream, stderr=total_err_f_obj
    )

    width, height = new_width, new_height

    frame_size = int(width * height * 1.5)
    clip_size = targ_fps * frame_size

    err_msg_list = []
    enc_err_sec = []
    frame_err_sec = []

    for cur_sec in range(start_sec, end_sec + 1):
        # yuv420p each pixel need only 1.5 bytes.
        in_bytes = process1.stdout.read(clip_size)
        if not in_bytes:
            err_msg_list.append("Warning: No more data after timestamp {}.".format(cur_sec))
            break

        actual_frame_num = int(len(in_bytes) / frame_size)
        if actual_frame_num < targ_fps:
            err_msg_list.append("Warning: Timestamp {} has only {} frames.".format(cur_sec, actual_frame_num))

        out_filename = os.path.join(targ_dir, "{}.mp4".format(cur_sec))

        args = ["ffmpeg", '-f', 'rawvideo', '-pix_fmt', 'yuv420p',
                '-r', str(targ_fps), '-s', '{}x{}'.format(width, height),
                '-i', 'pipe:', '-pix_fmt', 'yuv420p', out_filename, '-y']
        process2 = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=None, stderr=subprocess.PIPE
        )
        out, err = process2.communicate(input=in_bytes)
        err_str = err.decode("utf8")
        if "error" in err_str.lower():
            enc_err_sec.append(cur_sec)

        if midframe_root != "":
            midframe_filename = os.path.join(frame_targ_dir, "{}.jpg".format(cur_sec))
            args = ["ffmpeg", '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-s', '{}x{}'.format(width, height),
                    '-i', 'pipe:', '-pix_fmt', 'yuv420p', midframe_filename, '-y']
            process2 = subprocess.Popen(
                args, stdin=subprocess.PIPE, stdout=None, stderr=subprocess.PIPE
            )
            out, err = process2.communicate(input=in_bytes[:frame_size])
            err_str = err.decode("utf8")
            if "error" in err_str.lower():
                frame_err_sec.append(cur_sec)

    in_bytes = process1.stdout.read()
    more_frame = int(len(in_bytes) / frame_size)
    if more_frame > 0:
        err_msg_list.append("Warning: {} frames has been dropped.".format(more_frame))

    process1.wait()
    with open(total_err_file, 'r') as total_err_f_obj:
        err_str = total_err_f_obj.read()
    if "error" in err_str.lower():
        err_msg_list.append("Error happens when decoding the raw video.")
    else:
        os.remove(total_err_file)

    if len(enc_err_sec) > 0:
        err_msg_list.append(
            "Error in encoding short clips. Some of the error timestamps are {}.".format(enc_err_sec[:5]))

    if len(frame_err_sec) > 0:
        err_msg_list.append(
            "Error in encoding key frames. Some of the error timestamps are {}.".format(frame_err_sec[:5]))

    if len(err_msg_list) > 0:
        err_msg = '\n'.join(err_msg_list)
        err_msg = "Message from {}:\n".format(movie_path) + err_msg
    else:
        err_msg = ''
    return err_msg


def multiprocess_wrapper(args):
    args, kwargs = args
    return slice_movie_yuv(*args, **kwargs)

def main():
    parser = argparse.ArgumentParser(description="Script for processing AVA videos.")
    parser.add_argument(
        "--movie_root",
        required=True,
        help="root directory of downloaded movies",
        type=str,
    )
    parser.add_argument(
        "--clip_root",
        required=True,
        help="root directory to store segmented video clips",
        type=str,
    )
    parser.add_argument(
        "--kframe_root",
        default="",
        help="root directory to store extracted key frames",
        type=str,
    )
    parser.add_argument(
        "--process_num",
        default=4,
        help="the number of processes",
        type=int,
    )
    args = parser.parse_args()

    movie_path_list = []
    clip_root_list = []
    kwargs_list = []

    movie_names = os.listdir(args.movie_root)
    for movie_name in movie_names:
        movie_path = os.path.join(args.movie_root, movie_name)
        movie_path_list.append(movie_path)
        clip_root_list.append(args.clip_root)
        kwargs_list.append(dict(midframe_root=args.kframe_root))

    pool = Pool(args.process_num)
    for ret_msg in tqdm.tqdm(
            pool.imap_unordered(multiprocess_wrapper, zip(zip(movie_path_list, clip_root_list), kwargs_list)),
            total=len(movie_path_list)):
        if ret_msg != "":
            tqdm.tqdm.write(ret_msg)

if __name__ == '__main__':
    main()