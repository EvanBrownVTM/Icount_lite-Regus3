cat yolov4x-mish.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=512/width=320/' | sed -e '9s/height=512/height=320/' > yolov4x-mish-320.cfg
ln -sf yolov4x-mish.weights yolov4x-mish-320.weights

