# 在rk3588部署yolov5检测，使用npu推理，并用ROS2调用

> rk3588环境：Ubuntu22.04、ROS2 humble、python3.10

## PC端操作

### PC端环境准备
> 所用PC环境：Ubuntu22.04，x86_64
> 参考：[https://docs.radxa.com/rock5/rock5b/app-development/rknn_install#pc-%E7%AB%AF%E4%B8%8B%E8%BD%BD-rknn-toolkit2-%E4%BB%93%E5%BA%93](https://docs.radxa.com/rock5/rock5b/app-development/rknn_install#pc-%E7%AB%AF%E4%B8%8B%E8%BD%BD-rknn-toolkit2-%E4%BB%93%E5%BA%93)

- 下载 RKNN 仓库
建议新建一个目录用来存放 RKNN 仓库，例如新建一个名称为 Projects 的文件夹，并将 RKNN-Toolkit2 和 RKNN Model Zoo 仓库存放至该目录下。
```bash
# 新建 Projects 文件夹
mkdir Projects && cd Projects
# 下载 RKNN-Toolkit2 仓库
git clone -b v2.3.0 https://github.com/airockchip/rknn-toolkit2.git
# 下载 RKNN Model Zoo 仓库
git clone -b v2.3.0 https://github.com/airockchip/rknn_model_zoo.git
```

- （可选但建议）创建venv虚拟环境
    ```bash
    sudo apt isntall python3.8-venv python3.8-distutils
    python3.8 -m ensurepip --upgrade
    python3.8 -m venv venv/rknn_py3.8
    ```

- PC 端安装 RKNN-Toolkit2  
激活 conda rknn 环境后，进入 rknn-toolkit2 目录，根据您的架构平台和 Python 版本选择相应的 requirements_cpxx.txt 安装依赖， 并通过 wheel 包安装 RKNN-Toolkit2，这里以 64 位 x86 架构 Python3.8 环境的 PC 为例子，参考命令如下:
```bash
# 进入 rknn-toolkit2 目录
cd rknn-toolkit2/rknn-toolkit2/packages/x86_64/
# 请根据不同的 python 版本，选择不同的 requirements 文件, 这里以 python3.8 为例子
pip3 install -r requirements_cp38-2.3.0.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 请根据不同的 python 版本及处理器架构，选择不同的 wheel 安装包文件
pip3 install ./rknn_toolkit2-2.3.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

- 验证是否安装成功
执行以下命令，若没有报错，则代表 RKNN-Toolkit2 环境安装成功。
```bash
$ python3
>>> from rknn.api import RKNN
```

### PC端模型转化（pt转onnx）：
> 具体记不清，为防止错误，贴上参考博客。  
> 参考  
[https://blog.csdn.net/m0_57315535/article/details/128250096](https://blog.csdn.net/m0_57315535/article/details/128250096)  
[https://blog.csdn.net/m0_55217834/article/details/130583886](https://blog.csdn.net/m0_55217834/article/details/130583886)

- （可选）创建venv虚拟环境，用于训练yolov5模型
    ```bash
    python3.8 -m venv ~/venv/yolov5-v5
    ```

- 克隆或下载 yolov5 指定版本：c5360f6e commit，后面以yolov5-c5360代替文件夹名称
    [https://github.com/ultralytics/yolov5/tree/c5360f6e7009eb4d05f14d1cc9dae0963e949213](https://github.com/ultralytics/yolov5/tree/c5360f6e7009eb4d05f14d1cc9dae0963e949213)

- 进入训练的虚拟环境，下载依赖
    ```bash
    # yolov5-v5虚拟环境下
    source ~/venv/yolov5-v5/bin/activate
    cd yolov5-c5360
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    # 训练过程中若出现依赖问题，调整numpy、pillow和pytorch版本，以下requirements.txt作为参考：
    # pip install -r requirements.txt

    # base ----------------------------------------
    matplotlib>=3.2.2
    # numpy>=1.18.5
    numpy==1.23.5
    opencv-python>=4.1.2
    # Pillow>=8.0.0
    Pillow==9.5.0
    PyYAML>=5.3.1
    scipy>=1.4.1
    # torch>=1.7.0
    torch==1.7.1
    torchvision>=0.8.1
    tqdm>=4.41.0

    # logging -------------------------------------
    tensorboard>=2.4.1
    # wandb

    # plotting ------------------------------------
    seaborn>=0.11.0
    pandas

    # export --------------------------------------
    # coremltools>=4.1
    onnx>=1.9.0
    # scikit-learn==0.19.2  # for coreml quantization
    # tensorflow==2.4.1  # for TFLite export

    # extras --------------------------------------
    # Cython  # for pycocotools https://github.com/cocodataset/cocoapi/issues/172
    # pycocotools>=2.0  # COCO mAP
    # albumentations>=1.0.3
    thop  # FLOPs computation
    ```
- yolov5-v5虚拟环境下训练自己的pt模型，第一次尝试最好使用yolov5 v5.0版本的预训练模型yolov5s.pt进行测试：
    [https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt](https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt)

- 使用 export.py 将pt模型转为onnx  
    先修改models/yolo.py的forward函数，注意训练时要用回原来的forward函数，转模型时用修改后的：
    ```python
    '''
    def forward(self, x):  # 训练用
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)   
    '''

    def forward(self, x):  # 转换onnx用
        z = []  # inference output
        for i in range(self.nl):
            x[i] = torch.sigmoid(self.m[i](x[i]))  # conv
        return x
    ```

- 选择好要转的模型路径，开始转换
    ```bash
    # yolov5-v5虚拟环境下
    python export.py --weights runs/train/exp4/weights/best.pt --include onnx --opset 12 --batch-size 1
    ```

- （可选，如果不使用自训练模型pt转onnx）下载 yolov5s_relu.onnx 模型：
    ```bash
    cd rknn_model_zoo/examples/yolov5/model
    # 下载预训练好的 yolov5s_relu.onnx 模型
    bash download_model.sh
    ```

### PC端模型转化（onnx转rknn）：
- 如使用venv虚拟环境，请先激活rknn环境
    ```bash
    source ~/venv/rknn/bin/activate
    ```

- 克隆或下载 rknn_model_zoo 代码仓库
    ```bash
    git clone -b v2.3.0 https://github.com/airockchip/rknn_model_zoo.git
    ```

- 使用 rknn-toolkit2 转换成 yolov5s_relu.rknn：
    ```bash
    cd rknn_model_zoo/examples/yolov5/python
    python3 convert.py <onnx_model> <TARGET_PLATFORM> <dtype> <output_rknn_path>
    # python3 convert.py ../model/yolov5s_relu.onnx rk3588 i8 ../model/yolov5s_relu_rk3588.rknn
    ```
    > 参数解析：  
    > <onnx_model>: 指定 ONNX 模型路径  
    > <TARGET_PLATFORM>: 指定 NPU 平台名称。可选 rk3562, rk3566, rk3568, rk3576, rk3588, rk1808, rv1109, rv1126  
    > <dtype>: 指定为 i8 或 fp。i8 用于 int8 量化，fp 用于 fp16 量化。默认为 i8  
    > <output_rknn_path>: 指定 RKNN 模型的保存路径  

- 将 rknn 模型拷贝到板端

## 板端操作

### 板端环境准备
> 参考：[https://docs.radxa.com/rock5/rock5b/app-development/rknn_install#pc-%E7%AB%AF%E4%B8%8B%E8%BD%BD-rknn-toolkit2-%E4%BB%93%E5%BA%93](https://docs.radxa.com/rock5/rock5b/app-development/rknn_install#pc-%E7%AB%AF%E4%B8%8B%E8%BD%BD-rknn-toolkit2-%E4%BB%93%E5%BA%93)  

- 镜像烧录：  
    本项目烧录镜像为    
    `Orangepi5plus_1.2.0_ubuntu_jammy_desktop_xfce_linux5.10.160.img`

- 烧录完成后，进入系统查看NPU驱动：  
    ```bash
    sudo cat /sys/kernel/debug/rknpu/version
    ```  
    输出：
    `RKNPU driver: v0.9.6`

- 安装 python3.10:  
    后续使用ROS2 humble需要python3.10  
    ```bash
    sudo apt install software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt install python3.10
    ```

- 创建 venv 虚拟环境：
    ```bash
    sudo apt isntall python3.10-venv python3.10-distutils
    python3.10 -m ensurepip --upgrade
    python3.10 -m venv venv/rknn_py3.10
    ```

- 激活 venv 虚拟环境：
    ```bash
    source venv/rknn_py3.10/bin/activate
    ```

#### （可选但推荐）虚拟环境单独安装 rknn_toolkit-lite2 wheel
> 参考 [https://docs.radxa.com/rock5/rock5b/app-development/rknn_install](https://docs.radxa.com/rock5/rock5b/app-development/rknn_install)

- 下载仓库 [https://github.com/airockchip/rknn-toolkit2/tree/master](https://github.com/airockchip/rknn-toolkit2/tree/master)
```bash
cd rknn-toolkit2/rknn-toolkit-lite2/packages/
```
- 根据板端系统的 python 版本将对应的.whl文件复制到板端，这里是python3.8

- 进入虚拟环境后使用 pip3 安装
```bash
pip3 install rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- 执行以下命令，若没有报错，则代表 rknn_toolkit-lite2 环境安装成功。
```bash
$ python3
>>> from rknnlite.api import RKNNLite as RKNN
``` 

### 板端推理yolov5
> 参考[https://docs.radxa.com/rock5/rock5b/app-development/rknn_toolkit_lite2_yolov5](https://docs.radxa.com/rock5/rock5b/app-development/rknn_toolkit_lite2_yolov5)
- 下载仓库 [https://github.com/airockchip/rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo)

- 修改 `rknn_model_zoo/py_utils/rknn_executor.py` 代码，最好备份原版代码
```python
from rknnlite.api import RKNNLite as RKNN

class RKNN_model_container():
    def __init__(self, model_path, target=None, device_id=None) -> None:
        rknn = RKNN()
        rknn.load_rknn(model_path)
        ret = rknn.init_runtime()
        self.rknn = rknn

    def run(self, inputs):
        if self.rknn is None:
            print("ERROR: rknn has been released")
            return []

        if isinstance(inputs, list) or isinstance(inputs, tuple):
            pass
        else:
            inputs = [inputs]

        result = self.rknn.inference(inputs=inputs)

        return result

    def release(self):
        self.rknn.release()
        self.rknn = None
```

- 修改 `rknn_model_zoo/examples/yolov5/python/yolov5.py` 262 行的代码, 最好备份原版代码
```python
262 outputs = model.run([np.expand_dims(input_data, 0)])
```

- 进入虚拟环境

- 安装依赖环境
```bash
pip3 install opencv-python-headless
```

- 运行 yolov5 示例代码
```bash
cd rknn_model_zoo/examples/yolov5/python
python3 yolov5.py --model_path <your model path> --img_save
```
如使用的是自己转换的模型，需从 PC 端拷贝到板端，并用 --model_path 参数指定模型路径
> --model_path: 指定 rknn 模型路径
> --img_folder: 进行推理的图片库, 默认 ../model
> --img_save: 是否保存推理结果图到 ./result，默认 False

- 所有推理结果保存在 ./result 中

## 引入ROS2，并使用摄像头输入图像
- 克隆或下载本仓库
```bash
git clone https://github.com/neckprotecter/yolov5_rknn_ws.git
```

- 准备usb摄像头，下载usbROS2功能包
```bash
sudo apt-get install ros-humble-usb-cam
# 启动摄像头节点
ros2 run usb_cam usb_cam_node_exe
```

- 进入rknn_py3.10虚拟环境，编译运行
```bash
sudo chmod 777 ./make.sh ./detect.sh
# 编译
./make
# 运行
./detect.sh
```

### 相关参数修改介绍
- rknn模型文件放在 `src\yolov5_rknn_ros\models`
- 在 [detect.launch.py](src\yolov5_rknn_ros\launch\detect.launch.py) 中修改导入的模型路径 model_path 和摄像头设备号 device_id
