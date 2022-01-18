"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
Licensed under GNU Lesser General Public License v3.0
"""


import os
from dlclive import benchmark_videos


def main():

    # create example video from posetrack dataset:
    # ffmpeg -start_number 1 -i 00000%d.jpg -vcodec mpeg4 test.avi
    video_file_1 = os.path.join("00985_mpii/test.avi")

    # record test video with your webcam and put it into poetlive directory:
    #video_file_2 = os.path.join("sample.avi")

    # put POET checkpoint into same directory
    checkpoint = os.path.join("checkpoint.pth")

    # run benchmark videos
    print("\n Running inference...\n")
    benchmark_videos(checkpoint, video_file_1, display=True, resize=0.4, save_video=True, display_radius=5, poet=True)
    #benchmark_videos(checkpoint, video_file_2, display=True, resize=0.4, save_video=True, poet=True)



if __name__ == "__main__":
    main()