cat yolov4-csp.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=512/width=416/' | sed -e '9s/height=512/height=416/' > yolov4-csp-416.cfg
ln -sf yolov4-csp.weights yolov4-csp-416.weights

