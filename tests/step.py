import tensorflow as tf


def step(inp, y, model, opt, has_batch_norm=False, use_softmax=False):
    """Performs one optimizer step on a single mini-batch."""
    with tf.GradientTape() as tape:
        ###### CAUTION:
        # If one does not flatten: The model will generate a (_,1) "tensor"
        # If one flatten ("tf.reshape(_,[-1])"), it gives a (_) 1-dimensional "tensor"
        #out = tf.reshape(mlp(tf.cast(inp, dtype=tf.float64)),[-1])
        if use_softmax:
            logits_or_softmax = "softmax"
        else:
            logits_or_softmax = "logits"
        if has_batch_norm:
            out = model(tf.cast(inp, dtype=tf.float32), is_training=True)[logits_or_softmax]
        else:
            out = model(tf.cast(inp, dtype=tf.float32))[logits_or_softmax]



        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=y) # This assumes the output did not go through softmax

        #loss = tf.math.pow(tf.math.subtract(out, y), tf.cast(tf.constant(2), dtype=tf.float64))
        #loss = tf.math.abs(out - y)
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
    return loss, n
