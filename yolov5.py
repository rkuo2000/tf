# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
import torch

model = torch.hub.load('ultralytics/yolov5','yolov5s',pretrained=True)

imgs = ['https://ultralytics.com/images/zidane.jpg']

results = model(imgs)

results.print()

results.save()

print(results.xyxy[0])

print(results.pandas().xyxy[0])
