import tensorflow as tf


def normalize(inputs, epsilon, scope='ln', reuse=None):
    """
    Applies layer normalization.
    :param inputs: a tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    :param epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    :param scope: Optional scope for 'variable_scope'.
    :param reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    :return:
        A tensor with the same shape and data dtype as `inputs`.
    """
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape), dtype=tf.float32)
        gamma = tf.Variable(tf.ones(params_shape), dtype=tf.float32)
        normalized = (inputs - mean) / tf.sqrt((variance + epsilon))
        outputs = gamma * normalized + beta
    return outputs


def multihead_attention(queries, keys, num_units=None, num_heads=8, dropout_rate=0, is_training=True,
                        causality=False, scope='multihead_attention', reuse=None):
    """
    Applies multihead attention
    :param queries: A 3d tensor with shape of [N, T_q, C_q].
    :param keys: A 3d tensor with shape of [N, T_k, C_k].
    :param num_units: A scalar, Attention size.
    :param num_heads: heads of attention.
    :param dropout_rate: A floating point number.
    :param is_training: Boolean. Controller of mechanism for dropout.
    :param causality: Boolean. If true, units that reference the future are masked.
    :param scope: Optional scope for `variable_scope`
    :param reuse: Boolean, whether to reuse the weiths of a previous layer by the same name.
    :return: A 3d tensor with shape of (N, T_q, C)
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)   # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)     # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)     #(N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)      # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))        # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / tf.sqrt(tf.cast(K_.get_shape().as_list()[-1], dtype=tf.float32))

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))       # (N, T_k)
        key_masks = tf.tile(key_masks, [num_units, 1])      # (h*N, T_k)
        # (h*N, T_q, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(outputs)*(-2**32 + 1)

        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])      # (T_q, T_k)
            

