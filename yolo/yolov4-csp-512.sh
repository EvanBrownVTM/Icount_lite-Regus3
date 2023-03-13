cat yolov4-csp.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=512/width=256/' | sed -e '9s/height=512/height=256/' > yolov4-csp-256.cfg
ln -sf yolov4-csp.weights yolov4-csp-256.weights

