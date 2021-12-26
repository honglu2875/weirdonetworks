# weirdonetworks
This repo includes:
- a collection of non-conventional neural networks born out of my imagination,
- an experiment on measuring the stability under transformations of transfer-learning using popular pre-trained models of computer visions. 

Non-conventional NNs are written to fit Deepmind's Sonnet package because of personal preference. The networks are written as a subclass of sonnet.Module

---
# Neural networks for MNIST
So the main theme here is image recognition. Despite MNIST being extremely simple (some say as simple as "hello, world" in C) which I mildly disagree, it is at least a great benchmark, source of inspirations and a perfect testing problem.

I always vent about two things: 1. BatchNorm, 2. Conv2D.
- I won't use BatchNorm because I do not aim to find a network that can recognize a bunch of digits 100% correctly without being able to recognize a single frame reasonably well. If there exists an abstraction (neural-network-like model) of human recognition, it has to be able to deal with single frames as well as a collection of frames.
- I understand that convolutional networks can be good in real-world engineering sense. But I'm not a huge believer of convolution networks, as they are ultimately a special form of linear transformation which bears no theoretic difference than MLP. My personal bias is that it sometimes works well perhaps because of the domain knowledge (inspired by human vision). But... whatever. 

Some thoughts:

Engineers like to scale up, use big networks, etc. It's powerful, and a lot of the brightest minds have been working around it. However, as a mathematician, I am particularly attracted by the **inverse** of scaling:
1. How can we find the minimal network that solves certain problem reasonably well.
2. How can we design a small network that still consists of a lot of important properties (e.g., symmetry of the input).

The first question is analytic, and will have to do with the loss landscape and the properties of functions at the global minimum. 

The second question is geometric. A group acting on the input data and we need to answer what the best way to have a "network" that already incorporates the symmetry. There are a lot of studies done (enlarge the training set, graph neural networks, parameter sharing, etc.), but they have a bit of ad hoc flavor and perhaps we have not exhausted all the useful solutions yet. Enlarging the training set by sampling/generating the group orbits is the industry standard. But is the whole network "stable" under the group action (|| F(gx) - F(x) || < epsilon) for *arbitrary* input? Without intrinsic structural reasons, my guess is that the training set needs to be sufficiently big (and the network might not be sufficiently overparametrized?) and very well-distributed.

The networks:
(note: the following models are mathematically equivalent to raw MLP models. The question is whether the domain knowledge about symmetry helps, and whether the networks are born stable under the symmetry.)

## 1. Quadratic (or higher degree) MLP
The idea is very straightforward. It is a raw application of the invariant theory. Suppose the input is [x_1,x_2,...,x_n], we first generate all degree-d monomials [x_{i_1}...x_{i_d}] and then feed it through MLP. Why? If there is a compact group G acting on the variables x_1,...,x_n, the ring of invariants will be finitely generated, and they will (universally) approximate G-invariant functions under reasonable conditions. By using all the degree-d monomials, the hope is that it also "learns" the invariant functions (need experiments. May not work!). But starting with simple MLP, as they are piecewise linear, there is no guarantee that the network will be "stable" in whatever sense under the group action when it sees a piece of unseen data.

Need experiment
- whether the network is able to "choose" invariant polynomials
- choose a smaller set of monomial? Also how to get rid of redundancy.
- find a way to measure how "stable" the trained network is under the group action
- use Gröbner basis to only find invariant basis? 

## 2. Synchronous MLP
The idea is to literally average over the group action. But the averaging process happens **inside** the network. Given an input x=[x_1,x_2,...,x_n], we generate the set of orbits under the group action G: {gx|g\in G}. Now we feed each gx to the **same** neural network. And the output goes through a couple subsequent dense layers.

Need experiment
- what if the output goes through symmetric functions first, and then dense layers?

---
Experiments on stability of models under transformation

