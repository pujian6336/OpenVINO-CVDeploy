### OpenVINO部署精度与速度
在Intel(R) Core(TM) i7-10875H CPU上进行速度与精度的测试，精度测试阈值为：conf_thres:0.25,iou_thres0.65。

| Model  |size<br><sup>(pixes) |  Mode | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | speed<br>(usePlugin)<br>(FPS) | speed<br>(FPS) |
|:-------:|:-------:|:-------:| :-------:|:-------:|:-------: |:-------:|
|YOLOv5n|640|FP32<br>INT8 |28.0<br>27.5|45.6<br>45.0|103<br>-|90<br>113|
|YOLOv5s|640|FP32<br>INT8 |37.5<br>37.0|56.7<br>56.6|37<br>-|36<br>54|
|YOLOv5m|640|FP32<br>INT8 |45.2<br>44.9|63.9<br>63.8|13<br>-|13<br>22|
|YOLOv5n6|1280|FP32<br>INT8 |35.8<br>35.3|54.2<br>53.8|13<br>-|13<br>24|
|YOLOv5s6|1280|FP32<br>INT8 |44.5<br>44.0|63.5<br>63.3|8<br>-|7<br>12|
