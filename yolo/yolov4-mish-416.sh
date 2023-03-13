cat yolov4-mish.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=512/width=416/' | sed -e '9s/height=512/height=416/' > yolov4-mish-416.cfg
ln -sf yolov4-mish.weights yolov4-mish-416.weights

