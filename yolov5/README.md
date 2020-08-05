Data:

../trian.csv

../train

step1:

python convert.py

train:

CUDA_VISIBLE_DEVICES=0 python3 train.py --img 1024 --batch 4 --epochs 100 --data ./data/w0.yaml --cfg ./models/yolov5x.yaml --name yolov5x_fold0_1024 --weights ./yolov5x.pt

based on:
https://github.com/ultralytics/yolov5

LICENSE:
https://github.com/ultralytics/yolov5/blob/master/LICENSE
