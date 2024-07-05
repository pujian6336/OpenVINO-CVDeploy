### OpenVINO部署精度与速度
在Intel(R) Core(TM) i7-10875H CPU上进行速度与精度的测试，精度测试使用yolov10检测吃出的300个目标进行测试；速度测试阈值，conf_thres:0.25。

| Model  |size<br><sup>(pixes) |  Mode | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | speed<br>(FPS) |
|:-------:|:-------:|:-------:| :-------:|:-------:|:-------: |
|YOLOv10n|640|FP32<br>INT8 |38.4<br>38.1|53.6<br>53.2|77<br>102|
|YOLOv10s|640|FP32<br>INT8 |46.2<br>45.6|63.0<br>62.5|27<br>43|
|YOLOv10m|640|FP32<br>INT8 |50.7<br>49.6|67.7<br>66.7|10<br>19|
