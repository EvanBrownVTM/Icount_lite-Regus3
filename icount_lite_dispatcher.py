"""
this file will be run by Unix service
depending on configSrc.py either icount_lite.py or icount_lite_video.py will be run
"""
import utils_lite.configSrc as cfg
import icount_lite_video_tf1

while True:
	icount_lite_video_tf1.main()
