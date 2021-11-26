# weirdonetworks
This is a collection of non-conventional neural networks born out of my imagination. Maybe none of them has theoretic value but who knows! Collecting them in a GitHub repo makes notebook experiments much easier (and I do have a lot of experiments written!). For certain networks, the training code is special and will be included as well.

Everything is written to fit Deepmind's Sonnet package, because I like the abstraction and the freedom. The networks are written as a subclass of sonnet.Module

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

The second question is geometric. A group acting on the input data and we need to answer what the best way to have a "network" that already incorporates the symmetry. There are a lot of studies done (enlarge the training set, graph neural networks, parameter sharing), but they have a bit ad hoc flavor and perhaps we do not exhaust all the good solutions yet. Enlarging the training set by sampling/generating the group orbits is the industrial standard method. But how do we know that the training set is well-distributed in the space of all possible inputs, and how is whole network "stable" under the group action (|| F(gx) - F(x) || < epsilon) for *arbitrary* input? But of course I do not yet have a better answer.

The networks:

I love MLP and will stick to MLP most of the time! Here we go!

## 1. Quadratic MLP
The idea is stupid

## 2. Synchronous MLP
