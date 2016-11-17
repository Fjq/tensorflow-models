import tensorflow as tf
SIZE = 299

def prepare_pos(image):
    # 4 times too big
    print('pos')
    print(image.get_shape())
    image = resize_keep_ratio(image, SIZE*2, SIZE*2, name='pos')
    image = tf.image.resize_images(image, (SIZE, SIZE))
    print(image.get_shape())
    return image

def prepare_neg(image):
    # crop like the original image was cropped
    print('neg')
    print(image.get_shape())
    shape = tf.shape(image)
    min_ = tf.cast(tf.minimum(shape[0], shape[1]), tf.float32)
    ru = tf.cast(tf.random_uniform([], minval=min_ * .05, maxval=min_ * .8), tf.int32)
    # ru = tf.maximum(ru, 20)
    image = tf.random_crop(image, [ru,ru,3])
    image = resize_keep_ratio(image, SIZE*2, SIZE*2, name='neg')
    image = tf.image.resize_images(image, (SIZE, SIZE))
    print(image.get_shape())
    return image

def _round_to_int(x):
    return tf.cast(x + .5, tf.int32)

def resize_keep_ratio(image, height, width, name=''):
    """
    Resizes so that it fits into (height, width), keeping the aspect ratio
    and croping if need be.
    """
    print('resize')
    print(image.get_shape())
    shape = tf.cast(tf.shape(image), tf.float32)
    ratio_h = tf.cast(height, tf.float32) / shape[0]
    ratio_w = tf.cast(width, tf.float32) / shape[1]
    new_shape = tf.cond(
        tf.less_equal(shape[0], shape[1]),
        lambda: (_round_to_int(ratio_h * shape[0]), _round_to_int(ratio_h * shape[1])),
        lambda: (_round_to_int(ratio_w * shape[0]), _round_to_int(ratio_w * shape[1]))
    )
    image = tf.image.resize_images(image, new_shape)

    print(image.get_shape())
    print(image.dtype)
    # tf.image_summary('resized image_' + name, tf.expand_dims(image, 0))

    image = tf.random_crop(image, (height, width, 3))
    print(image.get_shape())
    return image
