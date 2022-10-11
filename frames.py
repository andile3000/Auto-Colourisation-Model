import cv2
start_time = 0
end_time = 60
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
ffmpeg_extract_subclip("match.mp4", start_time, end_time, targetname="test.mp4")

video_full = cv2.VideoCapture('match.mp4')
video_short = cv2.VideoCapture('test.mp4')
success_1,image_1 = video_full.read()
success_2,image_2 = video_short.read()
count = 0
while success_1:
  cv2.imwrite("./dataset/train/0/frame%d.jpg" % count, image_1)     # save frame as JPEG file
  success_1,image_1 = video_full.read()
  print('Read a new frame: ', success_1)
  count += 1

count = 0
while success_2:
  cv2.imwrite("./dataset/test/0/frame%d.jpg" % count, image_2)     # save frame as JPEG file
  success_2,image_2 = video_short.read()
  print('Read a new frame: ', success_2)
  count += 1