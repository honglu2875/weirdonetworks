import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp

from random import randrange
from .transform import *

def step(inp, y, model, opt, has_batch_norm=False):
    # Performs one optimizer step on a single mini-batch.
    # inp: a batch of input. [batch size, ...]
    # y: the true value. [batch size]
    # model: the model to be trained
    # opt: the optimizer

    with tf.GradientTape() as tape:
        ###### CAUTION:
        # If one does not flatten: The model will generate a (_,1) "tensor"
        # If one flatten ("tf.reshape(_,[-1])"), it gives a (_) 1-dimensional "tensor"
        #out = tf.reshape(mlp(tf.cast(inp, dtype=tf.float64)),[-1])

        # The output of the model has to be raw logits

        if has_batch_norm:
            out = model(tf.cast(inp, dtype=tf.float32), is_training=True)
        else:
            out = model(tf.cast(inp, dtype=tf.float32))

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=y) # This assumes the output did not go through softmax
        loss = tf.reduce_mean(loss)


    # GET TRAINABLE VARIABLES (Entries in the matrices)
    params = model.trainable_variables
    # GET THE BEST GRADIENT VECTOR THAT "MINIMIZES" THE LOSS (??)
    grads = tape.gradient(loss, params)

    ###DEBUG####
    #print(grads)
    #print(params)

    # UPDATE THE VARIABLES
    opt.apply(grads, params)


    ### get the norm of the gradient
    n = tf.constant(0., dtype=tf.float32)
    for t in grads:
        n = tf.math.add(n, tf.nn.l2_loss(tf.reshape(t,[-1])))

    # Generate the return info
    ret = {}
    ret['loss'] = loss
    ret['grad_norm'] = n
    return ret


def dream(inp, y, model, opt, D=100, ISO_ONLY=True, DEBUG=False, LAYER_WISE=False):
    # Enjoy the sweet dream
    # inp: the FULL input
    # model: the model to fall into sleep
    # opt: the optimizer
    # length: the number of dreams (samples)

    with tf.GradientTape() as tape:

        out = model(tf.cast(inp, dtype=tf.float32))

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=y) # This assumes the output did not go through softmax
        loss = tf.reduce_mean(loss)

    params = model.trainable_variables
    grads = tape.gradient(loss, params) # Get the average gradient (non-persistent because we redo gradient soon)

    # Take one sample, generate an orbit, and suppress the variance in the orbit
    sample = inp[randrange(0, len(inp))]
    with tf.GradientTape() as tape:
        o = tf.cast(
            tf.convert_to_tensor(gen_orbit(sample, D=D, ISO_ONLY=ISO_ONLY)), dtype=tf.float32
            )
        r = tf.nn.softmax(model(o))
        r = tf.math.reduce_std(r, axis=0)
        assert len(r.shape) == 1
        r = tf.math.reduce_sum(r)

    params = model.trainable_variables
    rgrads = tape.gradient(r, params)

    projected_gradient = []
    if LAYER_WISE:
        assert len(grads) == len(rgrads) and len(grads) == len(params)
        for g, rg in zip(grads, rgrads):
            d = tf.math.reduce_sum(tf.math.multiply(rg,g)) / tf.math.reduce_sum(tf.math.multiply(g,g))
            projected_gradient.append(rg-tf.math.multiply(d, g))
        opt.apply(tuple(projected_gradient), params)
    else:
        flattened_g = tf.concat( [tf.reshape(x, [-1]) for x in grads], axis=0 )
        flattened_rg = tf.concat( [tf.reshape(x, [-1]) for x in rgrads], axis=0 )
        d = tf.math.reduce_sum(tf.math.multiply(flattened_rg,flattened_g)) / tf.math.reduce_sum(tf.math.multiply(flattened_g,flattened_g))
        projected_gradient = tuple([rgrads[i]-tf.math.multiply(d,grads[i]) for i in range(len(rgrads))])
        opt.apply(projected_gradient, params)

    return r

    from random import random

@tf.function
def var(inp, model, NUM_SAMPLE):
    r = tf.stack([model(
                tf.cast(
                tf.convert_to_tensor(tfa.image.rotate(inp, random()*360)), dtype=tf.float32
            )) for _ in range(NUM_SAMPLE)], axis=-1)

    r = tf.math.reduce_std(r, axis=-1)
    return r

@tf.function
def dif(inp, model, eps_angle, NUM_SAMPLE):
    o = 0
    for _ in range(NUM_SAMPLE):
        a = random()*360.
        o += abs(
            model(
                tf.cast(
                tf.convert_to_tensor(tfa.image.rotate(inp, a)), dtype=tf.float32
            )) - model(
                tf.cast(
                tf.convert_to_tensor(tfa.image.rotate(inp, a+random()*eps_angle)), dtype=tf.float32
            )))

    return o/NUM_SAMPLE

def dream_v2(inp, y, model, opt, eps_angle=1, NUM_SAMPLE=10, DEBUG=False, USE_VAR=True):
    # Enjoy the sweet dream v2 (rotation only)
    # inp: a batch of input
    # model: the model to fall into sleep
    # opt: the optimizer
    # eps_angle: the maximal range of small angle perturbation
    # NUM_SAMPLE: sample times
    # length: the number of dreams (samples)

    with tf.GradientTape() as tape:

        out = model(tf.cast(inp, dtype=tf.float32))

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=y) # This assumes the output did not go through softmax
        loss = tf.reduce_mean(loss)

    params = model.trainable_variables
    grads = tape.gradient(loss, params) # Get the average gradient (non-persistent because we redo gradient soon)

    # Take one sample, generate an orbit, and suppress the variance in the orbit
    with tf.GradientTape() as tape:
        if USE_VAR:
            r = var(inp, model, NUM_SAMPLE)
        else:
            r = dif(inp, model, eps_angle, NUM_SAMPLE)
        r = tf.math.reduce_mean(r)

    params = model.trainable_variables
    rgrads = tape.gradient(r, params)

    flattened_g = tf.concat( [tf.reshape(x, [-1]) for x in grads], axis=0 )
    flattened_rg = tf.concat( [tf.reshape(x, [-1]) for x in rgrads], axis=0 )
    d = tf.math.reduce_sum(tf.math.multiply(flattened_rg,flattened_g)) / tf.math.reduce_sum(tf.math.multiply(flattened_g,flattened_g))
    projected_gradient = tuple([rgrads[i]-tf.math.multiply(d,grads[i]) for i in range(len(rgrads))])

    opt.apply(projected_gradient, params)

    return r
