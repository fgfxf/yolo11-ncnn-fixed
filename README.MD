# Abstract

解决：ncnn yolov11 乱框，输出维度出错，无框问题

Solution: ncnn yolov11 has random boxes, incorrect output dimensions, and no boxes

# 0x00: model export

首先是模型转换问题

最开始，我是用python的ultralytics导出为onnx格式文件(.onnx)，这个文件在python的pytorch和C++的onnxruntime库中推理良好，于是我就直接onnx转为ncnn格式(.bin和.param)

这会导致ncnn的输出维度出错。![20241225120514](https://github.com/fgfxf/yolo11-ncnn-fixed/blob/fb0cfb46418686a753a51f1216a532357465858a/20241225120514.jpg)

Firstly, there is the issue of model transformation

At first, I exported the file in onnx format (. onnx) using Python's ultralytics. This file worked well in Python's pytorch and C++'s onnxruntime library, so I directly converted onnx to ncnn format (. bin and. param)

This will cause an error in the output dimension of ncnn.

```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
success = model.export(format="onnx")
```

```powershell
onnx2ncnn.exe best.onnx yolov11.param yolov11.bin
```

另外，Tencent官方也建议，不要用onnx转换，而是用pnnx转换。不过我也没有用pnnx转换，因为ultralytics官方有自己的模型转换方式，直接会导出适合ncnn的bin和param文件

```powershell
yolo export model=yolo11n.pt format=ncnn
```

In addition, Tencent officials also suggest not using onnx conversion, but using pnnx conversion. However, I did not use pnnx conversion because ultralytics has its own model conversion method, which directly exports bin and param files suitable for ncnn

执行完命令后，你就在yolo11n_ncnn_model文件夹下拿到了model.ncnn.bin和model.ncnn.param文件了

![image-20241225120120407](https://github.com/fgfxf/yolo11-ncnn-fixed/blob/0cfe6dc7274667f7e57a97d137d186ce88a670d5/image-20241225120120407.png)

# 0x01:  Picture resize

图片的大小必须符合要求，最好是640*640

在python中，人家都给你封装好了，你输入图片的大小无严格要求也不会出错。例如：

The size of the image must meet the requirements, preferably 640 * 640

In Python, they have already packaged it for you, so there is no strict requirement for the size of the input image and there will be no errors. For example:

```python
import ncnn
import cv2
import numpy as np
from ncnn.model_zoo import get_model

image_handler  = cv2.imread('../bus.jpg')

net2 = get_model("mobilenet_ssd", use_gpu=False)
objects = net2(image_handler)
```

这不会导致任何问题，但是在C++中，无论是onnxruntime还是ncnn，都需要把图片的输入大小resize成符合要求的大小。

此外，即使是在python中，使用原始yolov11进行基于pytorch的模型推理，也是需要用户自行修改图片维度的。

例如(代码以torch_directml为例，cuda版torch同理)：

This will not cause any problems, but in C++, both onnxruntime and ncnn require resizing the input size of the image to meet the requirements.

In addition, even in Python, using the original YOLOV11 for PyTorch based model inference requires users to modify the image dimensions themselves.

For example (taking torch-directml as an example, the CUDA version of torch is the same):

```python
import cv2
import numpy as np
import torch
import torch_directml
import ultralytics
from torchvision import transforms

device = torch_directml.device(0)
model = ultralytics.YOLO("yolo11n.pt",task="detect").to(device).model
model.eval()
INFER_SIZE = (640, 640)
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为张量
    transforms.Resize(INFER_SIZE),  # YOLO 通常要求输入大小为 640x640
    transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
])

cap = cv2.VideoCapture(0)
ret, frame = cap.read()  #read a frame
resized_frame = cv2.resize(frame, INFER_SIZE)  #resize
input_tensor = transform(resized_frame).unsqueeze(0).to(device)
outputs = model(input_tensor)     #output is (1,1,84,8400)
predictions = outputs[0].cpu().numpy()
predictions = predictions[0]   #predictions   is (84,8400)
#随后的处理略
#do something
```

因此NCNN也必须要处理图片维度，在NCNN官方中有yolov8的代码处理案例，核心代码如下：

[ncnn/examples/yolov8.cpp at master · Tencent/ncnn](https://github.com/Tencent/ncnn/blob/master/examples/yolov8.cpp)

```c++
const char* imagepath = "bus.jpg";
cv::Mat bgr= cv::imread(imagepath, 1);
ncnn::Net yolov8;
yolov8.load_param("yolo11n_ncnn_model\\model.ncnn.param");
yolov8.load_model("yolo11n_ncnn_model\\model.ncnn.bin");
const int target_size = 640;
const float prob_threshold = 0.15f;
const float nms_threshold = prob_threshold;
int img_w = bgr.cols;
int img_h = bgr.rows;

// letterbox pad to multiple of MAX_STRIDE
int w = img_w;
int h = img_h;
float scale = 1.f;
if (w > h)
{
    scale = (float)target_size / w;
    w = target_size;
    h = h * scale;
}
else
{
    scale = (float)target_size / h;
    h = target_size;
    w = w * scale;
}

ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data,ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

int wpad = (target_size + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
int hpad = (target_size + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
ncnn::Mat in_pad;
ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
```

# 0x02: substract_mean_normalize

这一步非常重要，ncnn官方的案例如下，这个是错误的！！！

This step is very important. The official case of ncnn is as follows, which is incorrect!!!

This step is very important. The official case of ncnn is as follows, which is incorrect!!!

This step is very important. The official case of ncnn is as follows, which is incorrect!!!

```c++
const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
in_pad.substract_mean_normalize(0, norm_vals);
```

修正如下：

fixed as follow:

```c++
const float norm_vals[3] = { 1 / 0.348492 / 0xff, 1 / 0.3070657 / 0xff, 1 / 0.28770673 / 0xff };
in_pad.substract_mean_normalize(0, norm_vals);
```

真的很奇怪，这个substract_mean_normalize的设定对推理结果影响非常大。

设置错误的substract_mean_normalize会导致乱框和无框的发生。官方给出的案例也不适合yolov11

# 0x03: NCNN inference

现在，它应该工作正常了。

![20241225121335](https://github.com/fgfxf/yolo11-ncnn-fixed/blob/0cfe6dc7274667f7e57a97d137d186ce88a670d5/20241225121335.jpg)

fgfxf  于 河南科技大学
fgfxf, Henan University of Science and Technology
2024.12.25

