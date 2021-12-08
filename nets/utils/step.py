import tensorflow as tf
from random import random
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


def dream(inp, model, opt, D=100, ISO_ONLY=True, DEBUG=False):
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
    sample = inp[random.randrange(0, len(inp))]
    with tf.GradientTape() as tape:
        o = tf.cast(
            tf.convert_to_tensor(gen_orbit(sample, D=D, ISO_ONLY=ISO_ONLY)), dtype=tf.float32
            )
        r = tf.nn.softmax(model(o))
        r = tf.math.reduce_std(r, axis=0)
        assert len(r.shape) == 1
        r = tf.nn.reduce_sum(r)

    params = model.trainable_variables
    rgrads = tape.gradient(r, params)

    d = tf.math.reduce_sum(tf.math.multiply(rgrads,grads))/tf.math.reduce_sum(tf.math.multiply(grads,grads))
    opt.apply(rgrads-tf.math.multiply(d, grads), params) # Apply the projection of the suppression gradient to the orthogonal plane of the loss gradient

    # Debug:
    if DEBUG:
        print(grads)
        print(rgrads)
        print(rgrads-tf.math.multiply(d, grads))

    return r
