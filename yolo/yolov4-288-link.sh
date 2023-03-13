cat yolov4.cfg | sed -e '2s/batch=64/batch=1/' | sed -e '7s/width=608/width=288/' | sed -e '8s/height=608/height=288/' > yolov4-288.cfg
ln -sf yolov4.weights yolov4-288.weights

