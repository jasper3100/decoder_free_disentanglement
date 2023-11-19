# Decoder-free disentanglement: An attempt to scale unsupervised disentanglement to realistic data

This repository is a copy of https://github.com/facebookresearch/disentangling-correlated-factors which allows benchmarking disentanglement algorithms with correlated factors.
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
In this way, the underlying repository could be used directly without modifying many file paths.

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

## 2. Disentanglement using intermediate feature maps

In `dent/models/encoder/locatello.py` and `dent/models/decoder/locatello.py` respectively, we return the intermediate feature maps. We adjust the model `dent/models/vae_locatello.py` to use those intermediate features. The base trainer `dent/trainers/basetrainer.py` then passes those intermediate features to the loss function `dent/losses/factorizedsupportvae.py`, which is adapted correspondingly. The weight of using a specific intermediate feature map in the reconstruction loss is set manually in `dent/losses/factorizedsupportvae.py`.

Different loss functions were chosen by adjusting the `rec_loss` parameter in `base.yaml`: `bernoulli` for the BCE loss, `gaussian` for the MSE loss and `laplace` for the L1 loss. 
<!-- This determines the distribution parameter of reconstruction_loss in losses/utils.py -->

For using BCE with sigmoid normalization, we use `F.binary_cross_entropy_with_logits` in `dent/losses/utils.py`. Min-max normalization is implemented in `dent/models/utils.py` and it is directly applied to the intermediate feature maps in `dent/models/encoder/locatello.py` and `dent/models/decoder/locatello.py`.

For adjusting the reconstruction loss during training, for example, to introduce intermediate features after a certain number of epochs, we use an `if epoch >= n` statement in the computation of the reconstruction loss in `dent/models/decoder/locatello.py`. 

## 3. Disentanglement using a pre-trained decoder

To freeze the decoder after epoch *n*, we include *epoch* as an argument in `basetrainer.py` and set `param.requires_grad = False` if *epoch > n* (in `self.model.decoder.parameters()`).

To use a fresh encoder after epoch *n*, we initialize a new encoder in `basetrainer.py`. 

## 4. Using the [PUG datasets](https://github.com/facebookresearch/PUG)

The dataloaders for three of the PUG datasets can be found in `datasets`. The root directory should be set to the location where the data should be stored. In `datasets/utils.py` we add the datasets.
