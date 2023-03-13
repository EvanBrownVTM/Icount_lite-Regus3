cat yolov4.cfg | sed -e '2s/batch=64/batch=1/' | sed -e '7s/width=608/width=608/' | sed -e '8s/height=608/height=608/' > yolov4-608.cfg
ln -sf yolov4.weights yolov4-608.weights

