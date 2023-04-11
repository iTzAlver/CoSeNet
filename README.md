# CoSeNet: An Excellent Approach for Optimal Segmentation of Correlation Matrices.

<p align="center">
    <img src="https://raw.githubusercontent.com/iTzAlver/CoSeNet/blob/master/doc/multimedia/cosenet_logo.png" width="400px">
</p>

<p align="center">
    <a href="https://github.com/iTzAlver/CoSeNet/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/iTzAlver/CoSeNet?color=purple&style=plastic" /></a>
    <a href="https://github.com/iTzAlver/CoSeNet/tree/master/test">
        <img src="https://img.shields.io/badge/tests-passed-green?color=green&style=plastic" /></a>
    <a href="https://github.com/iTzAlver/CoSeNet/blob/master/requirements.txt">
        <img src="https://img.shields.io/badge/requirements-pypi-red?color=red&style=plastic" /></a>
    <a href="https://htmlpreview.github.io/?https://github.com/iTzAlver/CoSeNet/blob/main/doc/CoSeNet.html">
        <img src="https://img.shields.io/badge/doc-available-green?color=yellow&style=plastic" /></a>
    <a href="https://github.com/iTzAlver/CoSeNet/releases/tag/1.1.0-release">
        <img src="https://img.shields.io/badge/release-1.1.0-white?color=white&style=plastic" /></a>
</p>

<p align="center">
    <a href="https://github.com/iTzAlver/BaseNet-API/">
        <img src="https://img.shields.io/badge/dependencies-BaseNetAPI-red?color=orange&style=for-the-badge" /></a>
    <a href="https://www.ray.io/">
        <img src="https://img.shields.io/badge/dependencies-ray-red?color=blue&style=for-the-badge" /></a>
</p>

# CoSeNet - 1.1.0.

The proposed approach is known as CoSeNet (Correlation Segmentation
Network), and is based on a four-layer architecture that includes several processing layers: an input
layer, formatting, re-scaling and a final segmentation layer. The proposed model is able to effectively
identify correlated segments in such matrices, better than previous approaches for similar problems.
Internally, the proposed model utilizes an overlapping technique and makes use of pre-trained
Machine Learning (ML) algorithms, which makes it robust and generalizable. CoSeNet model also
includes a method that optimizes the parameters of the re-scaling layer using a heuristic algorithm
and a fitness metric based on the Window Difference metric. The results of our model are binary
matrices with the noise removed and can be used in a variety of applications and the compromise
solutions between efficiency, memory and speed of the proposed deployment model are chosen.

## About ##

Author: A.Palomo-Alonso (alberto.palomo@uah.es)\
Universidad de Alcalá.\
Escuela Politécnica Superior.\
Departamento de Teoría De la Señal y Comunicaciones (TDSC).\
ISDEFE Chair of Research.

## What's new?

### < 1.0.0
1. All included:
   * CoSeNet: Main class.
   * Fitness method: Genetic and PSO algorithms.
   * Solve: Pipeline solver.
   * Pre-trained models: Ridge and MLP for 16 throughput.

## Install

To install the API you need to install the following software:

   * CUDA: To use the GPU training (in case you your GPU for trining a model).
   * Ray: For distributed computing, used by the fit method.

You will need to install all the Python packages shown in the ````requirements.txt```` file. Once done, you can install 
the package via pip.
   
      pip install cosenet

## Architecture

We propose a novel approach for optimal segmentation of correlation matrices, based on a complete
sequential architecture which involves different processing layers, each implementing several algorithms. Specifically,
the proposed approach consists of four layers, which can be grouped into 4 categories. The first layer is an input/output
layer, responsible for inputting and outputting the correlation matrices to be processed or exported by the architecture.
The second layer consists of different procedures essential to optimally prepare the input data. The third layer,
Metaheuristic, normalizes the input and output data using classical algorithms. Finally, the fourth layer is formed by
different Machine Learning (ML) algorithms able to accurately identify the boundaries in the provided correlation
matrix. Thus, the proposed architecture is able to process square correlation matrices of any scale and size, using a ML
model capable of identifying segments with high performance, even for highly noisy data. The proposed approach is
able to adapt any matrix, regardless of its size, to the ML model with excellent performance. The proposed architecture
also runs faster on general-purpose processors, making it a more practical solution for real-world applications.
The performance of the algorithm has been evaluated with a highly nonlinear and noisy database. The problem
proposed in the comparative is a problem of text segmentation by topics. We obtain random articles from Wikipedia,
concatenate them and divide them by sentences. With a language model (BERT) we generate a sentence similarity
coefficient, used as correlation value and correlation matrices are generated with these values sentence by sentence.
The effectiveness in identifying correlated group segmentation and its superiority to some state-of-the-art algorithms
such as unsupervised, Community Detection and Deep Clustering have been tested, reaching improvements of 6% -
22% in terms of performance. The pipeline aims to propose a unified solution to the problem, with the possibility of
performing fine-tuning with a few samples from the database.

### Fast Usage:

You can use the package easily by importing the main class:

    from cosenet import CoSeNet

Then, you can create an instance of the class and use the methods:

    number_of_matrix_to_solve = 600
    matrix_size = 100
    model = CoSeNet()
    x = np.random.rand(number_of_matrix_to_solve, matrix_size, matrix_size)
    solved_matrix, predicted_segmentation = model.solve(x)

To fit a highly non-linear database you can use the fit method:

    # x has a shape of (number_of_matrix_to_solve, matrix_size, matrix_size)
    x = load_my_numpy_data()
    # y has a shape of (number_of_matrix_to_solve, matrix_size) and it is a binary matrix.
    y = load_my_numpy_segmentation_boundaries()
    model.fit(50, x_train, y_train, 'genetic')

Note that the fit method is a distributed method, so you need to have Ray installed.
You also need to notica that the ``y`` matrix is a binary matrix, where the 1s are the boundaries of the segments and 0
otherwise.

### Documentation:

You can find the documentation of the package in the repository in the doc folder.

    https://github.com/iTzAlver/cosenet/tree/master/doc

You can also find the research article in the following link:

    https://arxiv.org/AWAITING-DOI

### Cite as

~~~
Awaiting for citation (ArXiv)
~~~
