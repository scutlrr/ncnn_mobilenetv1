# ncnn_mobilenetv1
pytorch->onnx->ncnn  
环境搭建参考：  
https://blog.csdn.net/weixin_42448226/article/details/104951934  
https://blog.csdn.net/weixin_42448226/article/details/104985789  

参数与模型下载：  
链接: https://pan.baidu.com/s/13unsZN-XixHaQXJudkz5qw 提取码: 2gwt  
(请将mobilenetv1.bin和mobilenetv1.param.bin文件放置在ncnn_mobilenetv1/model目录下)  


## 开发环境

- Windows10
- Visual Studio 2017
- ncnn最新版
- Opencv 3.4.2


## 推理

- 模型
  - **mobilenetv1 **：<https://github.com/Tencent/ncnn/wiki/use-ncnn-with-alexnet.zh>
  -  参考nihui大佬的ncnn组件使用指北


- main.cpp

  ```c++
  //  加载模型
  ncnn::Net net;
  net.load_param_bin("../model/mobilenetv1.param.bin");
  net.load_model("../model/mobilenetv1.bin");
  // forward
  ncnn::Extractor ex = net.create_extractor();
  ex.set_light_smode(true);
  ex.input(mobilenetv1_param_id::BLOB_input_1, matIn);
  ex.extract(squeezenet_v1_1_param_id::BLOB_248, matOut);
  ```
