# Automatic Road Detection via U-NET Neural Network

## Architecture

Input: 3x112x160

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 112, 160]           1,728
       BatchNorm2d-2         [-1, 64, 112, 160]             128
              ReLU-3         [-1, 64, 112, 160]               0
            Conv2d-4         [-1, 64, 112, 160]          36,864
       BatchNorm2d-5         [-1, 64, 112, 160]             128
              ReLU-6         [-1, 64, 112, 160]               0
        DoubleConv-7         [-1, 64, 112, 160]               0
         MaxPool2d-8           [-1, 64, 56, 80]               0
         DownBlock-9  [[-1, 64, 56, 80], [-1, 64, 112, 160]]  0
           Conv2d-10          [-1, 128, 56, 80]          73,728
      BatchNorm2d-11          [-1, 128, 56, 80]             256
             ReLU-12          [-1, 128, 56, 80]               0
           Conv2d-13          [-1, 128, 56, 80]         147,456
      BatchNorm2d-14          [-1, 128, 56, 80]             256
             ReLU-15          [-1, 128, 56, 80]               0
       DoubleConv-16          [-1, 128, 56, 80]               0
        MaxPool2d-17          [-1, 128, 28, 40]               0
        DownBlock-18  [[-1, 128, 28, 40], [-1, 128, 56, 80]]  0
           Conv2d-19          [-1, 256, 28, 40]         294,912
      BatchNorm2d-20          [-1, 256, 28, 40]             512
             ReLU-21          [-1, 256, 28, 40]               0
           Conv2d-22          [-1, 256, 28, 40]         589,824
      BatchNorm2d-23          [-1, 256, 28, 40]             512
             ReLU-24          [-1, 256, 28, 40]               0
       DoubleConv-25          [-1, 256, 28, 40]               0
        MaxPool2d-26          [-1, 256, 14, 20]               0
        DownBlock-27  [[-1, 256, 14, 20], [-1, 256, 28, 40]]  0
           Conv2d-28          [-1, 512, 14, 20]       1,179,648
      BatchNorm2d-29          [-1, 512, 14, 20]           1,024
             ReLU-30          [-1, 512, 14, 20]               0
           Conv2d-31          [-1, 512, 14, 20]       2,359,296
      BatchNorm2d-32          [-1, 512, 14, 20]           1,024
             ReLU-33          [-1, 512, 14, 20]               0
       DoubleConv-34          [-1, 512, 14, 20]               0
        MaxPool2d-35           [-1, 512, 7, 10]               0
        DownBlock-36  [[-1, 512, 7, 10], [-1, 512, 14, 20]]   0
           Conv2d-37          [-1, 1024, 7, 10]       4,718,592
      BatchNorm2d-38          [-1, 1024, 7, 10]           2,048
             ReLU-39          [-1, 1024, 7, 10]               0
           Conv2d-40          [-1, 1024, 7, 10]       9,437,184
      BatchNorm2d-41          [-1, 1024, 7, 10]           2,048
             ReLU-42          [-1, 1024, 7, 10]               0
       DoubleConv-43          [-1, 1024, 7, 10]               0
  ConvTranspose2d-44          [-1, 512, 14, 20]       2,097,664
           Conv2d-45          [-1, 512, 14, 20]       4,718,592
      BatchNorm2d-46          [-1, 512, 14, 20]           1,024
             ReLU-47          [-1, 512, 14, 20]               0
           Conv2d-48          [-1, 512, 14, 20]       2,359,296
      BatchNorm2d-49          [-1, 512, 14, 20]           1,024
             ReLU-50          [-1, 512, 14, 20]               0
       DoubleConv-51          [-1, 512, 14, 20]               0
          UpBlock-52          [-1, 512, 14, 20]               0
  ConvTranspose2d-53          [-1, 256, 28, 40]         524,544
           Conv2d-54          [-1, 256, 28, 40]       1,179,648
      BatchNorm2d-55          [-1, 256, 28, 40]             512
             ReLU-56          [-1, 256, 28, 40]               0
           Conv2d-57          [-1, 256, 28, 40]         589,824
      BatchNorm2d-58          [-1, 256, 28, 40]             512
             ReLU-59          [-1, 256, 28, 40]               0
       DoubleConv-60          [-1, 256, 28, 40]               0
          UpBlock-61          [-1, 256, 28, 40]               0
  ConvTranspose2d-62          [-1, 128, 56, 80]         131,200
           Conv2d-63          [-1, 128, 56, 80]         294,912
      BatchNorm2d-64          [-1, 128, 56, 80]             256
             ReLU-65          [-1, 128, 56, 80]               0
           Conv2d-66          [-1, 128, 56, 80]         147,456
      BatchNorm2d-67          [-1, 128, 56, 80]             256
             ReLU-68          [-1, 128, 56, 80]               0
       DoubleConv-69          [-1, 128, 56, 80]               0
          UpBlock-70          [-1, 128, 56, 80]               0
  ConvTranspose2d-71         [-1, 64, 112, 160]          32,832
           Conv2d-72         [-1, 64, 112, 160]          73,728
      BatchNorm2d-73         [-1, 64, 112, 160]             128
             ReLU-74         [-1, 64, 112, 160]               0
           Conv2d-75         [-1, 64, 112, 160]          36,864
      BatchNorm2d-76         [-1, 64, 112, 160]             128
             ReLU-77         [-1, 64, 112, 160]               0
       DoubleConv-78         [-1, 64, 112, 160]               0
          UpBlock-79         [-1, 64, 112, 160]               0
           Conv2d-80          [-1, 1, 112, 160]              65
================================================================
Total params: 31,037,633
Trainable params: 31,037,633
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.21
Forward/backward pass size (MB): 3,331,729.43
Params size (MB): 118.40
Estimated Total Size (MB): 3,331,848.04
----------------------------------------------------------------
```

## Dataset

Save dataset under folder `data`. <br>
Data: [kaggle.com](https://www.kaggle.com/datasets/ipythonx/tgrs-road)

## Sources

- *G. Cheng, Y. Wang, S. Xu, H. Wang, S. Xiang and C. Pan, "Automatic Road Detection and Centerline Extraction via Cascaded End-to-End Convolutional Neural Network," in IEEE Transactions on Geoscience and Remote Sensing, vol. 55, no. 6, pp. 3322-3337, June 2017, doi: 10.1109/TGRS.2017.2669341.*
- *Sambyal, N., Saini, P., Syal, R., & Gupta, V. (2020). Modified U-Net architecture for semantic segmentation of diabetic retinopathy images. Biocybernetics and Biomedical Engineering, 40(3), 1094–1109. https://doi.org/10.1016/j.bbe.2020.05.006*
- *[Aladdin Persson]. (2021, January 21). U-NET Paper Walkthrough [Video]. YouTube. https://youtu.be/oLvmLJkmXuc*
- *[Aladdin Persson]. (2021, February 2). PyTorch Image Segmentation Tutorial with U-NET: Everything from scratch baby [Video]. YouTube. https://youtu.be/IHq1t7NxS8k*
- *[DigitalSreeni]. (2021, May 27). 219 - Understanding U-Net architecture and building it from scratch [Video]. YouTube. https://youtu.be/GAYJ81M58y8*

## License
Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg