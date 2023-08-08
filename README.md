# Automatic Network Pruning via Hilbert-Schmidt Independence Criterion Lasso under Information Bottleneck Principle

> Automatic Network Pruning via Hilbert-Schmidt Independence Criterion Lasso under Information Bottleneck Principle
> Song Guo, Lei Zhang, Xiawu Zheng, Yan Wang, Yuchao Li, Fei Chao, ShengChuan Zhang, Chenglin Wu, Rongrong Ji
> ICCV 2023

### Model Pruning

##### 1. VGG-16
pruning ratio (FLOPs): 66%
```shell
python main.py \
--model vgg16\
--dataset cifar10\
--target 107000000 \
--ckpt [pre-trained model dir] \
--data_path [dataset path]\
--omega 40\
--tolerance 0.01\
--alpha 5e-5
```
##### 2. ResNet56
pruning ratio (FLOPs): 55%
```shell
python main.py \
--model resnet56\
--dataset cifar10\
--target 57000000 \
--ckpt [pre-trained model dir] \
--data_path [dataset path]\
--omega 5\
--tolerance 0.01\
--alpha 8e-4
```
##### 3. ResNet110 
pruning ratio (FLOPs): 63%
```shell
python main.py \
--model resnet110\
--dataset cifar10\
--target 96000000 \
--ckpt [pre-trained model dir] \
--data_path [dataset path]\
--omega 5\
--tolerance 0.01\
--alpha 8e-9
```
##### 4. GoogLeNet
pruning ratio (FLOPs): 63%
```shell
python main.py \
--model googlenet\
--dataset cifar10\
--target 568000000 \
--ckpt [pre-trained model dir] \
--data_path [dataset path]\
--omega 9\
--tolerance 0.01\
--alpha 4e-8
```
##### 5. ResNet50
pruning ratio (FLOPs): 62%

```shell
python main.py \
--model resnet50\
--dataset imagenet\
--target 1550000000 \
--ckpt [pre-trained model dir] \
--data_path [dataset path]\
--omega 1\
--tolerance 0.01\
--alpha 7e-5
```
### Model Training
##### 1. VGG-16
```shell
python train.py \
--model vgg16\
--dataset cifar10\
--lr 0.1\
--batch_size 256 \
--ckpt_path [pruned model dir]\
--data_path [dataset path]
```
##### 2. ResNet-50
```shell
python train.py \
--model resnet50\
--dataset imagenet\
--lr 0.01\
--batch_size 128 \
--ckpt_path [pruned model dir]\
--data_path [dataset path]
```
## Pre-trained Models 

Additionally, we provide the pre-trained models used in our experiments. 


### CIFAR-10:
 [Vgg-16](https://drive.google.com/file/d/1gIeELUSnopEbFesdioxx0WSxs0jUQd16/view?usp=sharing) 
| [ResNet56](https://drive.google.com/file/d/1smOu2BHaU8-ngwv4RMpULKQlvKCrqqO9/view?usp=sharing) 
| [ResNet110](https://drive.google.com/file/d/1smOu2BHaU8-ngwv4RMpULKQlvKCrqqO9/view?usp=sharing)  
| [GoogLeNet](https://drive.google.com/file/d/1smOu2BHaU8-ngwv4RMpULKQlvKCrqqO9/view?usp=sharing) 

### ImageNet:
 [ResNet50](https://drive.google.com/file/d/1C73Qfvst09XtSbw_X77qLhMrnV4Rn1Uh/view?usp=sharing)

## Acknowledgments

Our implementation partially reuses [Lasso's code](https://github.com/lippman1125/channel_pruning_lasso) | [HRank's code](https://github.com/lmbxmu/HRank) | [ITPruner's code](https://github.com/MAC-AutoML/ITPruner).
