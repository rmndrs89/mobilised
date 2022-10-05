import tensorflow as tf
from tensorflow.keras import backend as K

class MyWeightedBinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(self, weight=0.01, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight
    
    def call(self, y_true, y_pred):
        loss = K.mean(K.binary_crossentropy(y_true, y_pred) * (y_true + self.weight), axis=-1)
        return loss
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "weight": self.weight}

class MyWeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, weights=[], **kwargs):
        super().__init__(**kwargs)
        self.weights = weights
    
    def call(self, y_true, y_pred):
        epsilon_ = tf.cast(K.epsilon(), dtype=y_pred.dtype)
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        y_pred = tf.clip_by_value(y_pred, epsilon_, 1 - epsilon_)
        loss = -tf.reduce_sum(y_true * tf.math.log(y_pred) * self.weights, axis=-1)
        return loss

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "weights": self.weights}

class MyWeightedMeanSquaredError(tf.keras.losses.Loss):
    def __init__(self, weights=0.01, **kwargs):
        super().__init__(**kwargs)
        self.weights = weights

    def call(self, y_true, y_pred):
        loss = K.mean(K.mean(K.square(y_pred - y_true)) * (y_true + self.weights), axis=-1)
        return loss

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "weights": self.weights}
