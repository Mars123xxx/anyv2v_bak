import numpy as np
from PIL import Image
import cv2
import os
from moviepy.video.io.VideoFileClip import VideoFileClip

def save_frames(video_name,inverval,frame_count=16):
    video_path = f'./demo/{video_name}'

    # 打开视频文件
    cap = cv2.VideoCapture(video_path+'.mp4')

    # 检查视频是否打开成功
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    c = 0
    c_ = 0
    count = 0
    while True:
        # 读取下一帧
        ret, frame = cap.read()

        # 如果正确读取帧，ret为True
        if not ret:
            print("Reached end of video or failed to read frame.")
            break

        if count == c:
            # 构建输出文件名
            frame_filename = os.path.join(video_path, f"{count:05d}.png")
            c += inverval
            c_ += 1
            # 保存帧
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")
        if c_ == frame_count:
            break

        count += 1

    # 释放视频捕获对象
    cap.release()

def extract_frames(video_name,frame_count=16):
    video_path = f'./demo/{video_name}'
    clip = VideoFileClip(video_path+'.mp4')
    duration = clip.duration
    frames = []

    # Calculate the time interval at which to extract frames
    times = np.linspace(0, duration, frame_count, endpoint=False)

    for t in times:
        # Extract the frame at the specific timestamp
        frame_ = clip.get_frame(t)
        # Convert the frame (numpy array) to a PIL Image
        pil_img = Image.fromarray(frame_)
        frames.append(pil_img)

    for idx, frame in enumerate(frames):
        frame.save(f'{video_path}/{idx:05d}.png')


video_name = 'Face_10'

# 默认方法，正态采样
# extract_frames(video_name)

# 根据特定步长抽取

save_frames(video_name,10)