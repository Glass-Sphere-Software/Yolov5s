cd yolov5
#python train.py --img 1080 --batch 2 --epochs 16 --data=../coco.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name yolov5s_results
python train.py --img 1080 --batch 2 --epochs 16 --data=../coco.yaml --cfg ./models/yolov5s.yaml --weights runs/train/yolov5s_results36/weights/last.pt --name yolov5s_results
pause
