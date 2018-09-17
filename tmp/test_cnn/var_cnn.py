import tensorflow as tf


def get_mask(seq_lens, max_len, dtype=tf.float32):
    mask = tf.cast(tf.sequence_mask(seq_lens, max_len), dtype=dtype)
    mask = tf.expand_dims(tf.expand_dims(mask, -1), -1)
    return mask


def var_cov2d(inputs, w, strides, padding, bias, seq_length, is_clip_output_size=True):
    """
    conv2d process variable length sequence input, without activate function.
    :param inputs: A tensor, [batch_size, max_time, dim, channel].
    :param w: A tensor, [filter_height, filter_width, in_channels, out_channels].
    :param strides: A list of ints. 1-D tensor of length 4.
    :param padding: A string from: "SAME", "VALID".
    :param bias: bias for cnn.
    :param seq_length: A tensor, [batch_size].
    :param is_clip_output_size: clip useless padding value in output
    :return: output, new_seq_len
    """
    h = tf.nn.conv2d(input=inputs, filter=w, strides=strides, padding=padding) + bias
    s = strides[1]
    seq_len1 = seq_length
    if padding == 'VALID':
        k = tf.shape(w)[0]
        seq_len1 = seq_length - k + 1
    new_seq_len = 1 + tf.floordiv((seq_len1 - 1), s)
    mask = get_mask(new_seq_len, tf.shape(h)[1], h.dtype)
    outputs = h * mask
    if is_clip_output_size:
        max_seq_len = tf.reduce_max(new_seq_len)
        outputs = outputs[:, :max_seq_len, :, :]
    return outputs, new_seq_len


def var_max_pool(inputs, ksize, strides, padding, seq_length, is_clip_output_size=True):
    """
    max pool for variable length sequence input,
    if padding == 'SAME', please make sure all the element in inputs >= 0
    :param inputs: A Tensor, [batch_size, max_time, dim, channel].
    :param ksize: A 1-D int Tensor of 4 elements.
    :param strides: A 1-D int Tensor of 4 elements.
    :param padding: A string, either 'VALID' or 'SAME'.
    :param seq_length: A tensor, [batch_size].
    :param is_clip_output_size: clip useless padding value in output
    :return: output, new_seq_len
    """
    h = tf.nn.max_pool(value=inputs, ksize=ksize, strides=strides, padding=padding)
    s = strides[1]
    seq_len1 = seq_length
    if padding == 'VALID':
        k = ksize[1]
        seq_len1 = seq_length - k + 1
    new_seq_len = 1 + tf.floordiv((seq_len1 - 1), s)
    mask = get_mask(new_seq_len, tf.shape(h)[1], h.dtype)
    outputs = h * mask
    if is_clip_output_size:
        max_seq_len = tf.reduce_max(new_seq_len)
        outputs = outputs[:, :max_seq_len, :, :]
    return outputs, new_seq_len
