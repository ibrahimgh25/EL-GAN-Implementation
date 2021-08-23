# EL-GAN
This repository offers an implementation for the Embedded loss GAN. The network was introduced in the paper [EL-GAN: Embedding Loss Driven Generative
Adversarial Networks for Lane Detection](https://arxiv.org/pdf/1806.05525) by Mohsen Ghafoorian, Cedric Nugteren, Nora Baka, Olaf Booij, Michael Hofmann.

## Introduction
The model introduced relied heavily on the Denseblocks which were used in  both the generator and the discriminator, it also introduced a novel approach to the discriminator, where the loss was caculated as a distance between the embeddings of the fake labels (labels produced by the generator), and the real labels (ground truth). They called this loss the embedding loss, they also passed not only the labels to the discriminator, but also the original images the labels respond to, creating a discriminator with "two heads".

## Generator
The generator in the model was a FC-DenseNet (fully connected dense network), which was introdued in [this](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w13/papers/Jegou_The_One_Hundred_CVPR_2017_paper.pdf) paper. The architecture of the generator could be seen in the image below.
![Generator Architecture]()
<p align="center">
  <img src="resources/generator_architecture.png" width="350" alt="The fully-connected densenet used for the generator from EL-GAN">
</p>
## Thanks
The repository was built on top of this [repository](https://github.com/baldassarreFe/pytorch-densenet-tiramisu) owned by Federico Baldassarre - the referenced repository offered an implementation for FCDenseNets and DenseNets. I would like to thank the author, whose repository not only helped me for the implementation and getting a better understanding for DenseNets, but also with organizing and presenting a project on GitHub.
