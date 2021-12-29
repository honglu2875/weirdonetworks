# weirdonetworks
This repo includes:
- a collection of non-conventional neural networks born in my study (under nets/),
- an experiment on measuring the stability under transformations of transfer-learning using popular pre-trained models of computer visions. See [THIS NOTEBOOK](https://github.com/honglu2875/weirdonetworks/blob/main/stability_measuring.ipynb)

The NNs are written to fit Deepmind's Sonnet package because of personal preference. The networks are written as a subclass of sonnet.Module

---
*(Draft-state research notes below. Work in progress.)*
## Experiment 1:
VGG16, ResNet50, MobileNetv3Large, EfficientNetB7 from keras applications. Got the binaries and applied transfer learning by attaching to another 512 Dense layer and trained on 4000 samples of [cats and dogs dataset](https://www.tensorflow.org/tutorials/images/transfer_learning) (binary classification) enhanced with random deformation (rotation, transformation, sheering). All of them achieved > 90% accuracy on non-deformed images(of course!). But applying deformation destroys the stability.

Proposed two quantities: transformation variance and transformation difference (see math notes at the end, need to compile LaTeX). 

Evaluated each model on 100 picture samples (horizontal axis). For each sample applied 10 rotations (evenly between 0 and 360 degrees) and calculated discretized transformation variance and difference. Below are charts. 
x-axis: sample numbers
orange line: transformation difference
blue line: transformation variance
green line: the average error of prediction under rotation

VGG16![VGG16](https://cdn.discordapp.com/attachments/830931439612723221/925870924291510292/VGG16.png)

ResNet50![ResNet50](https://cdn.discordapp.com/attachments/830931439612723221/925870924077617162/ResNet.png)

MobileNetv3Large![MobileNetv3Large](https://cdn.discordapp.com/attachments/830931439612723221/925870923867881542/MobileNet.png)

EfficientNetB7![EfficientNetB7](https://cdn.discordapp.com/attachments/830931439612723221/925870923628822548/EfficientNet.png)

Observations: accuracies seem to have to do with rotation stability but not entirely correlated. VGG is the worst.

## Experiment 2 (TO BE ADDED)
Initialize the CNN on a state invariant under rotations and then apply **projected** gradients (projected to the orthonormal complement of gradients of *transformation variance*) in the training can achieve good transformation invariant **even if the model is only trained using non-transformed input!** (Result to be posted. Work on going.)



&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---
# Some non-conventional networks:
(The motivation of keeping this library is to document my hand-written neural networks while experimenting.)


## 1. Quadratic (or higher degree) MLP
``` from weirdonetworks.nets import QuadMLP```

The idea is very straightforward. It is a raw application of the invariant theory. Suppose the input is [x_1,x_2,...,x_n], we first generate all degree-d monomials [x_{i_1}...x_{i_d}] and then feed it through MLP. Why? If there is a compact group G acting on the variables x_1,...,x_n, the ring of invariants will be finitely generated, and they will (universally) approximate G-invariant functions under reasonable conditions. By using all the degree-d monomials, the hope is that it also "learns" the invariant functions (need experiments. May not work!). But starting with simple MLP, as they are piecewise linear, there is no guarantee that the network will be "stable" in whatever sense under the group action when it sees a piece of unseen data.

Need experiment
- whether the network is able to "choose" invariant polynomials
- choose a smaller set of monomial? Also how to get rid of redundancy.
- find a way to measure how "stable" the trained network is under the group action
- use Gröbner basis to only find invariant basis? 

## 2. Synchronous MLP
``` from weirdonetworks.nets import SMLP```

The idea is to literally average over the group action. But the averaging process happens **inside** the network. Given an input x=[x_1,x_2,...,x_n], we generate the set of orbits under the group action G: {gx|g\in G}. Now we feed each gx to the **same** neural network. And the output goes through a couple subsequent dense layers.

Need experiment
- what if the output goes through symmetric functions first, and then dense layers?

## (TO BE ADDED)
---

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---
# p.s. some math notes on the [experiment](https://github.com/honglu2875/weirdonetworks/blob/main/stability_measuring.ipynb).
*(RESEARCH PROJECT IN-PROGRESS)*
```
Here we propose two metric to describe the stability of a model under a transformation.


## Notation setup
In our small scale setup, we assume that in the last layer we use sigmoid to classify binary classes. Also assume the input space is n-dimensional.

- $F: R^n\rightarrow R^n$ a piecewise-linear function according to a trained model. 

- $S=\{s: R^n\rightarrow R^n\}$ the (topological) space (most often manifolds) of transformations (endomorphisms) that we want to apply. In this notebook, we sample the rotations. It is NOT closed under compositions! For example, rotating turns some pixels out of bound and a further rotation forces us to fill in boundary values. 

- Assume $S$ is compact and has volume $1$ ($\int_Sds=1$).

Remark (vent?):
1. In the literature, people open say that $S$ forms a group and cite the group axioms. In a lot of scenarios outside of computer vision, this is indeed suitable. But when we talk about transformations on "images", the whole math falls apart:
    - Because 1. images are usually represented in a rectangular frame and 2. images are pixelated, the image transformations almost never form a group!
    - Without group axioms we might still have a monoid action, but even the "action" part falls apart: $$f\circ g(x) \neq (f\circ g)(x), \qquad f,g\in S$$

Is it "almost" a group action? No, not even close. Gladly, most papers involving image transforms do not essentially use the group action on functions, therefore not much is invalidated. This is simply a very dangerous practise that's worth being pointed out.



## Transformation variance
Fix an input $x\in S$, we define the transformation variance as follows:
$$ v=\int_S \left((f\circ s')(x)-\int_S(f\circ s)(x)ds\right)^2 ds' $$
When $S$ is discrete with normalized discrete measure (evaluate to $1/|S|$ on every point), this is literally the variance of the set.

This has a direct meaning: when we transform the image, the predicted probability should fluctuate as little as possible. It has its limit: it does not take how violence the fluctuation is into consideration. This is reflected the most in the example when the rotation of an image {\bf should} yield a different answer (e.g., recognizing 6 and 9 in MNIST). Such information is encoded in the derivatives under transformations.

## Transformation difference
Fix an input $x\in S$, assume $S$ forms a Riemannian manifold (locally Euclidean with a compatible measure induced by the metric), we define the transformation difference as follows:
$$d=\int_S\|\nabla_s(f\circ s)\|_nds,$$
where the gradient is taken over $s\in S$ and $\|\cdot\|_n$ is the $\mathcal l_n$-norm. In this notebook, $\mathcal l_1$-norm is used. This measures how fierce the fluctuation of value it is when we transform the image through $S$.

Note that this definition does not generalize to discrete sets to automatically include a discrete version. But an approximate version of the gradient and the integral is easy to formulate.
```
