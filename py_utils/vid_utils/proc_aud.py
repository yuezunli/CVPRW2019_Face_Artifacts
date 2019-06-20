from moviepy.editor import *


def audio_transfer(source, target, out_path):
    source_video = VideoFileClip(source)
    target_video = VideoFileClip(target)
    source_audio = source_video.audio
    processed_video = target_video.set_audio(source_audio)
    processed_video.write_videofile(out_path, audio_codec="aac")

