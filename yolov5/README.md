
# Data
```bash
$ kaggle competitions download -c global-wheat-detection
     
├── data  
│   ├── train.csv      
│   ├── train
│   ├── test
├── yolov5
│   ├── train.py           
│   ├── test.py       
.....
```

# step1
```bash
$ python convert.py
```
python convert.py


# train
```bash
$ CUDA_VISIBLE_DEVICES=0 python3 train.py --img 1024 --batch 4 --epochs 100 --data ./data/w0.yaml --cfg ./models/yolov5x.yaml --name yolov5x_fold0_1024 --weights ./yolov5x.pt
```
# Inference
https://www.kaggle.com/orkatz2/yolov5-fake-or-real-single-model-l-b-0-753

https://www.kaggle.com/orkatz2/yolov5x-k-folds-lb-0-748

# based on:
https://github.com/ultralytics/yolov5

# LICENSE:
https://github.com/ultralytics/yolov5/blob/master/LICENSE
