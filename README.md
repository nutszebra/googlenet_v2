# What's this
Implementation of GoogLeNet-v2 [[1]][Paper] by chainer


# Dependencies

    git clone https://github.com/nutszebra/googlenet_v2.git
    cd googlenet_v2
    git submodule init
    git submodule update

# How to run
    python main.py -p ./ -g 0 


# Details about my implementation

* Data augmentation  
Train: Pictures are randomly resized in the range of [256, 512], then 224x224 patches are extracted randomly and are normalized locally. Horizontal flipping is applied with 0.5 probability.  
Test: Pictures are resized to 384x384, then they are normalized locally. Single image test is used to calculate total accuracy. 

* Auxiliary classifiers  
No implementation

* Learning rate  
The description about the schedule of learning rate can't be found in [[1]][Paper], so  as [[2]][Paper] said, learning rate are multiplied by 0.96 at every 8 epochs. Initial learning rate is 0.045 acoording to [[1]][Paper].

* Weight decay  
The description about weight decay can't be found in [[1]][Paper], so weight decay is 4.0*10^-5 as [[3]][Paper3] says.

* Separable conv  
Normal convolution is used.


# Cifar10 result

| network              | depth  | total accuracy (%) |
|:---------------------|--------|-------------------:|
| my implementation    | 32     | 94.89              |

<img src="https://github.com/nutszebra/googlenet_v2/blob/master/loss.jpg" alt="loss" title="loss">
<img src="https://github.com/nutszebra/googlenet_v2/blob/master/accuracy.jpg" alt="total accuracy" title="total accuracy">

# References  
Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift [[1]][Paper]  
Going Deeper with Convolutions [[2]][Paper2]  
Rethinking the Inception Architecture for Computer Vision [[3]][Paper3]  

[paper]: https://arxiv.org/abs/1502.03167 "Paper"  
[paper2]: https://arxiv.org/abs/1409.4842 "Paper2"  
[paper3]: https://arxiv.org/abs/1512.00567 "Paper3"  
