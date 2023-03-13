"""
this file will be run by Unix service
depending on configSrc.py either icount_lite.py or icount_lite_video.py will be run
"""
import utils_lite.configSrc as cfg

if cfg.postprocess_video_mode:
    import icount_lite_video_tf1
    icount_lite_video_tf1.main()
else:
    import icount_lite
    icount_lite.main()