import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_CLASSES = 5

class CBAM(layers.Layer):
    def __init__(self, reduction_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = int(input_shape[-1])
        hidden = max(channels // self.reduction_ratio, 8)
        self.mlp1 = layers.Dense(hidden, activation="relu")
        self.mlp2 = layers.Dense(channels)
        self.spatial_conv = layers.Conv2D(1, 7, padding="same", activation="sigmoid")
        super().build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        ca = tf.nn.sigmoid(self.mlp2(self.mlp1(avg_pool)) +
                           self.mlp2(self.mlp1(max_pool)))
        x = inputs * ca
        avg_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_spatial = tf.reduce_max(x, axis=-1, keepdims=True)
        sa = self.spatial_conv(tf.concat([avg_spatial, max_spatial], axis=-1))
        return x * sa

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config

class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name="focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y = tf.one_hot(y_true, NUM_CLASSES)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        pt = tf.reduce_sum(y * y_pred, axis=-1)
        loss = -self.alpha * tf.pow(1 - pt, self.gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss)

    def get_config(self):
        return {"gamma": self.gamma, "alpha": self.alpha}
