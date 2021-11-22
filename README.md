# weirdonetworks
This is a collection of non-conventional neural networks born out of my imagination. Maybe none of them has theoretic value but who knows! Collecting them in a GitHub repo makes notebook experiments much easier (and I do have a lot of experiments written!). For certain networks, the training code is special and will be included as well.

Everything is written to fit Deepmind's Sonnet package, because I like the abstraction and the freedom. The networks are written as a subclass of sonnet.Module

---
# Neural networks for MNIST
So the main theme here is image recognition. Despite MNIST being extremely simple (some say as simple as "hello, world" in C) which I mildly disagree, it is at least a great benchmark, source of inspirations and a perfect testing problem.

I always vent about two things: 1. BatchNorm, 2. Conv2D.
- I won't use BatchNorm because I do not aim to find a network that can recognize a bunch of digits 100% without being able to recognize a single frame reasonably correctly. If there exists an abstraction (neural-network-like model) for human recognition, it has to be able to deal with single frame as well as a bunch of frames.
- I understand that convolutional networks can be good in real-world engineering sense. But I'm not a huge believer of convolution networks, as they are ultimately a special form of linear transformation which bears no theoretic difference than MLP. My personal bias is that it sometimes works well perhaps because of the domain knowledge (inspired by human vision). But... whatever. 

I love MLP and will stick to MLP most of the time! Here we go!

## 1. Quadratic MLP

## 2. Asynchronous MLP
