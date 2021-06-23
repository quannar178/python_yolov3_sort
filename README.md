# YOLOv3 + SORT - YOU ONLY LOOK ONCE + SIMPER OBJECT REALTIME TRACKING

## 1. Structure git branch

There are 4 branch:

- `master`: YOLO in computer.
- `colab`: YOLO in google colab.
- `sort`: SORT in computer.
- `colab_sort`: SORT in google colab. (working)

## 2. Structure folder, file + meaning

- YOLOv3_TF2: store information about yolo
  - cfg: file config network (download: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
  - data: store coco name (lable) (download: https://github.com/pjreddie/darknet/blob/master/data/coco.names)
  - weight: store weight for network (download: https://pjreddie.com/media/files/yolov3.weights)
  - image: path img for detect (+ tracking)
  - videos: path video for detect (+ tracking)
- `convert_weight.py`: convert file weight to tf
- `utils.py`: some useful function
- `yolov3.py`: network
- `image.py`: detect on image
- `video.py`: detect on video
- `tracking.py`: detect + tracking on Video (sure :vv only branch for sort)

## 3. Usage

1. Install requirement

```sh

pip install -r requirements.txt

```

2. Install coco.names, yolov3.cfg. yolov3.weights --> move to correct folder

3. Prepace video or image

4. Change path_video or img_video, then run
