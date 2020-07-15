pip uninstall -y mmdet mmcv torch torchvision 
conda install -y cython==0.28.5
#pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install -U "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
pip install torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
#pip install torch==1.5.0 torchvision==0.6.0
#conda install -y  pytorch cudatoolkit=10.2 torchvision -c pytorch
pip install -q  terminaltables==3.1.0 Pillow==6.2.2
git clone --recursive 'https://github.com/open-mmlab/mmdetection.git'
cd mmdetection
rm -rf build
pip install -v -e .
#python setup.py install
#python setup.py develop
#pip install -U "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"

#pip install -q mmcv==0.2.16
pip install -q mmcv==0.6.2
