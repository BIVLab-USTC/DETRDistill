<div align="center">
<h1> DETRDistill </h1>
<h3>A Universal Knowledge Distillation Framework for DETR-families</h3>
<br>Jiahao Chang, Shuo Wang, Hai-Ming Xu, Zehui Chen, Chenhongyi Yang, Feng Zhao. 
<br>

<div><a href="https://arxiv.org/pdf/2211.10156.pdf">[Paper] </a></div> 

<center>
<img src='main.jpg'>
</center>

</div>

## Install MMDetection and MS COCO2017
  - Our codes are based on [MMDetection](https://github.com/open-mmlab/mmdetection). Please follow the installation of MMDetection and make sure you can run it successfully.
```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.0
cd our project
pip install -r requirements/build.txt
pip install -v -e .
```
- Unzip COCO dataset into data/coco/
- Modify pretraining R-101/R-50 and our teacher model weight path

## Train

```
#multi GPU
bash tools/dist_train.sh  cfg_distill/deformdetr_r101_2x_distill_r50_LayerbyLayer_CL_teachergroup.py  8
```


## Transfer
```
# Tansfer the saved model into mmdet model
python pth_transfer.py --ckpt_path $ckpt --output_path $new_mmdet_ckpt
```
## Performance

### MS COCO Val set
|    Model    |  Backbone  | mAP |                            config                            |                          weight                          | code | log |
| :---------: | :--------: | :-----------: | :----------------------------------------------------------: | :------------------------------------------------------: | :--: |:--: |
|  Deformable DETR  | ResNet-101  |     44.8     | [config]() | [baidu]() |  | |
|  Deformable DETR  | ResNet-50  |     44.1     | [config]() | [baidu]() |  | |
|  Deformable DETR-Distill  | ResNet-50  |     46.6(+2.5)    | [config]() | [baidu]() |  | |
|  Deformable DETR  | ResNet-18  |     40.0     | [config]() | [baidu]() |  | |
|  Deformable DETR-Distill  | ResNet-18  |     43.3(+3.3)    | [config]() | [baidu]() |  | |


## NOTE

This repository is an initial draft, we will release more code in the future.

## Acknowledgement

Our code is based on the project [MMDetection](https://github.com/open-mmlab/mmdetection).
Thanks to the work [FGD](https://github.com/yzd-v/FGD).