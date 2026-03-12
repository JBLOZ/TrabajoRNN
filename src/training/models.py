from __future__ import annotations

from typing import Iterable, Sequence

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    GRU,
    Input,
    LSTM,
    Layer,
    LayerNormalization,
    Concatenate,
)
from tensorflow.keras.regularizers import l2


class TemporalAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        hidden_dim = int(input_shape[-1])
        self.w = self.add_weight(
            name="att_weight",
            shape=(hidden_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(int(input_shape[1]), 1),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.w) + self.b)
        a = tf.nn.softmax(e, axis=1)
        context = tf.reduce_sum(x * a, axis=1)
        return context


def _rnn_layer(
    units: int,
    rnn_type: str = "gru",
    return_sequences: bool = True,
    bidirectional: bool = False,
    dropout_rate: float = 0.0,
    recurrent_dropout_rate: float = 0.0,
):
    if rnn_type.lower() == "gru":
        layer = GRU(
            units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout_rate,
        )
    elif rnn_type.lower() == "lstm":
        layer = LSTM(
            units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout_rate,
        )
    else:
        raise ValueError(f"Unsupported rnn_type={rnn_type}")
    return Bidirectional(layer) if bidirectional else layer


def build_v1_baseline(
    input_shape: tuple[int, int],
    num_classes: int,
    rnn_type: str = "gru",
    units: int = 48,
    dense_units: int = 32,
    dropout_rate: float = 0.2,
    recurrent_dropout_rate: float = 0.1,
    l2_reg: float = 1e-4,
) -> Model:
    inp = Input(shape=input_shape, name="sequence_input")
    x = _rnn_layer(
        units=units,
        rnn_type=rnn_type,
        return_sequences=False,
        bidirectional=False,
        dropout_rate=dropout_rate,
        recurrent_dropout_rate=recurrent_dropout_rate,
    )(inp)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    out = Dense(num_classes, activation="softmax", name="prediction")(x)
    return Model(inputs=inp, outputs=out, name="v1_baseline_recurrent")


def build_v2_improved(
    input_shape: tuple[int, int],
    num_classes: int,
    rnn_type: str = "gru",
    units: Sequence[int] = (64, 32),
    dropout_rate: float = 0.25,
    recurrent_dropout_rate: float = 0.15,
    l2_reg: float = 1e-4,
    use_attention: bool = True,
) -> Model:
    inp = Input(shape=input_shape, name="sequence_input")
    x = inp
    for i, unit in enumerate(units):
        x = _rnn_layer(
            units=unit,
            rnn_type=rnn_type,
            return_sequences=True,
            bidirectional=True,
            dropout_rate=dropout_rate,
            recurrent_dropout_rate=recurrent_dropout_rate,
        )(x)
        x = LayerNormalization(name=f"ln_{i}")(x)
        x = Dropout(dropout_rate, name=f"do_{i}")(x)

    if use_attention:
        x = TemporalAttention(name="temporal_attention")(x)
    else:
        x = x[:, -1, :]
    x = Dense(64, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation="softmax", name="prediction")(x)
    return Model(inputs=inp, outputs=out, name="v2_improved_recurrent")


def build_v3_dual_branch(
    hrv_shape: tuple[int, int],
    morph_shape: tuple[int, int],
    num_classes: int,
    rnn_type: str = "gru",
    units_hrv: Sequence[int] = (64, 32),
    units_morph: Sequence[int] = (64, 32),
    dropout_rate: float = 0.3,
    recurrent_dropout_rate: float = 0.15,
    l2_reg: float = 1e-4,
    use_attention: bool = True,
) -> Model:
    hrv_in = Input(shape=hrv_shape, name="hrv_input")
    morph_in = Input(shape=morph_shape, name="morph_input")

    def branch(x, units, prefix):
        for i, unit in enumerate(units):
            x = _rnn_layer(
                units=unit,
                rnn_type=rnn_type,
                return_sequences=True,
                bidirectional=True,
                dropout_rate=dropout_rate,
                recurrent_dropout_rate=recurrent_dropout_rate,
            )(x)
            x = LayerNormalization(name=f"{prefix}_ln_{i}")(x)
            x = Dropout(dropout_rate, name=f"{prefix}_do_{i}")(x)
        if use_attention:
            x = TemporalAttention(name=f"{prefix}_attention")(x)
        else:
            x = x[:, -1, :]
        return x

    hrv_branch = branch(hrv_in, units_hrv, "hrv")
    morph_branch = branch(morph_in, units_morph, "morph")
    x = Concatenate(name="fusion")([hrv_branch, morph_branch])
    x = Dense(96, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(48, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    out = Dense(num_classes, activation="softmax", name="prediction")(x)
    return Model(inputs={"hrv_input": hrv_in, "morph_input": morph_in}, outputs=out, name="v3_dual_branch_recurrent")
