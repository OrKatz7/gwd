CUDA_VISIBLE_DEVICES=4 python3 train.py --img 800 --batch 4 --epochs 100 --data ./data/w0.yaml --cfg ./models/yolov5x.yaml --name yolov5x_fold0_800 --weights ./yolov5x.pt --multi-scale
