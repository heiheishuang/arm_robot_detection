# arm_robot_detection

## Camera
用于获取像素平面上的点，在空间座标系下的位置。

### 3D to 2D
```
camera = Camera(internal, external)
print(camera.cal_world2pix(p_end=vec_end))
```

### 2D to 3D
```
# 加载单应性变换
camera.load_param_one_mat(H_c2w)
print(camera.cal_pix2world_by_homo(depth=-67, p_camera=vec_camera))
```

## Yolo
用于检测平面上的垃圾，并给对应像素平面的位置。

### 环境配置
依据[Yolov5](https://github.com/ultralytics/yolov5)配置。

### 使用说明
```
weights = Path("./pretrain_weights/best.pt")
# 相机的编号
source = 0
run(weights=weights, source=source)
```
