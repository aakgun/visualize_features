# Visualizing CNN features

## Description

Simple iTorch example for visualizing CNN features as described in ["Visualizing and Understanding Convolutional Networks"](http://arxiv.org/abs/1311.2901) by Matthew Zeiler and Rob Fergus.

The implementation is quite quick-and-dirty: it seems to work, but I can't exclude the presence of methodological mistakes which by luck don't prevent the implementation to show something apparently meaningful.

## Requirements

* Torch 7
* `nn` package
* `image` package
* iTorch

## Resources needed

To run the example without modifications, you will need the OverFeat Torch wrapper from [jhjin/overfeat-torch](https://github.com/jhjin/overfeat-torch).

In short, you will need to:
* Clone the repository.
* Follow the instruction to download the weights files, and put the `net_weight_0` file in the `models` directory (you can ignore the other files in the tgz file).
* Compile the `ParamBank.c` file through the provided `Makefile`.
* Put `ParamBank.lua` and the compiled `libParamBank.so` in the same directory as my `visualize_features_example.ipynb` notebook.

Also you will need to `bee.jpg` test image [from OverFeat repository](https://raw.githubusercontent.com/sermanet/OverFeat/master/samples/bee.jpg) and put it into the repository root directory. However, it is straightforward to edit the notebook to choose any other picture you'd like.

## Compatiblity with other networks

You should also be able to use any other model, provided that:
* Layers are wrapped in a `nn.Sequential` container.
* It only has `nn.SpatialConvolutionMM`, `nn.SpatialMaxPooling`, `nn.ReLU` layers.
* Convolutional layers must be _proper_ convolutions; i.e. not 1x1 convolutions for fully-connected layers.

However, if you look at the code, you'll see easily how to adapt it for other layers. I just don't need them now.
