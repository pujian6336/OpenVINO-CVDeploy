### OpenVINO部署精度与速度
在Intel(R) Core(TM) i7-10875H CPU上进行速度与精度的测试，精度测试阈值为：conf_thres:0.001，iou_thres0.7，与yolov8官方精度测试阈值保持一致；速度测试阈值使用yolov8预测默认阈值，conf_thres:0.25，iou_thres0.45。

| Model  |size<br><sup>(pixes) |  Mode | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | speed<br>(FPS) |
|:-------:|:-------:|:-------:| :-------:|:-------:|:-------: |
|YOLOv8n|640|FP32<br>INT8 |37.3<br>37.0|52.6<br>52.3|55<br>80|
|YOLOv8s|640|FP32<br>INT8 |44.9<br>44.7|61.5<br>61.3|20<br>35|
|YOLOv8m|640|FP32<br>INT8 |50.2<br>49.9|67.0<br>66.8|7<br>14|
|||||||
|YOLOv5nu|640|FP32<br>INT8 |34.5<br>34.2|49.7<br>49.3|63<br>75|
|YOLOv5su|640|FP32<br>INT8 |43.0<br>42.8|59.7<br>59.5|25<br>39|
|YOLOv5mu|640|FP32<br>INT8 |48.9<br>48.7|65.7<br>65.5|10<br>16|
|YOLOv5n6u|1280|FP32<br>INT8 |42.2<br>42.0|58.6<br>58.3|14<br>20|

