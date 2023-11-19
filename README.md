# Decoder-free disentanglement: An attempt to scale unsupervised disentanglement to realistic data

This repository is a copy of https://github.com/facebookresearch/disentangling-correlated-factors for benchmarking disentanglement algorithms 
with correlated factors.
This repository contains the adjustments made by Jasper Toussaint as part of a research project at the University of Tübingen in 2023 to enable
1. Various forms of decoder-free disentanglement
2. Disentanglement using intermediate feature maps
3. Disentanglement using a pre-trained decoder
4. Using the [PUG datasets](https://github.com/facebookresearch/PUG)

<!-- enabling disentanglement without decoder, using intermediate feature maps and a pre-trained decoder. 
Moreover, the PUG datasets can be used for training. 
This repository contains a copy of the following repository: https://github.com/facebookresearch/disentangling-correlated-factors,
together with adjustments to enable-->
<!-- We quote from there: *The repository contains a general-purpose pytorch-based framework library and benchmarking suite to facilitate 
research on methods for learning disentangled representations. 
It was conceived especially for evaluating robustness under correlated factors, and contains all the code, benchmarks and method 
implementations used in the paper.*-->
The underlying repository contains a very comprehensive README on the structure of this repository. 
Thus, we only describe the adjustments here. 
For most files, where an adjustment was made, the original content is located in `file_original` and the adjustments in `file`, for example, 
`adagvae_original.py` and `adagvae.py`.
In this way, the underlying repository could be used directly.

## 1. Various forms of decoder-free disentanglement

In `dent/losses/adagvae.py` and `dent/losses/betavae.py` the computation of the reconstruction loss `rec_loss` is commented out and removed as 
input to the loss function, which enables decoder-free disentanglement.

In the same files we introduce the `reg_mode` parameter and we set `reg_mode = 'minimal_support'` or  `reg_mode = 'minimal_support'` 
to perform (decoder-free) disentanglement with either of the scale losses described in Appendix C of 
[Disentanglement of Correlated Factors via Hausdorff Factorized Support](https://openreview.net/forum?id=OKcJhpQiGiX). 
The scale loss computation is added as well and the parameter `reg_range` is added as a hyperparameter of the minimal support scale loss.

For performing decoder-free disentanglement with a scale loss and Hausdorff factorized support (HFS) as introduced in 
[Disentanglement of Correlated Factors via Hausdorff Factorized Support](https://openreview.net/forum?id=OKcJhpQiGiX), we use 
`dent/losses/factorizedsupportvae.py` where we added the scale loss details as described above.

HFS parameters?

## 2. Disentanglement using intermediate feature maps

## 3. Disentanglement using a pre-trained decoder

## 4. Using the [PUG datasets](https://github.com/facebookresearch/PUG)
