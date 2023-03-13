#!/bin/bash
set -e

cat yolov4-tiny.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=416/width=288/' | sed -e '9s/height=416/height=288/' > yolov4-tiny-288.cfg
echo >> yolov4-tiny-288.cfg
ln -sf yolov4-tiny.weights yolov4-tiny-288.weights

