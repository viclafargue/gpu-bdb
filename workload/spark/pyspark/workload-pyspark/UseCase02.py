#
# Copyright (C) 2021 Transaction Processing Performance Council (TPC) and/or its contributors.
# This file is part of a software package distributed by the TPC
# The contents of this file have been developed by the TPC, and/or have been licensed to the TPC under one or more contributor
# license agreements.
# This file is subject to the terms and conditions outlined in the End-User
# License Agreement (EULA) which can be found in this distribution (EULA.txt) and is available at the following URL:
# http://www.tpc.org/TPC_Documents_Current_Versions/txt/EULA.txt
# Unless required by applicable law or agreed to in writing, this software is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, and the user bears the entire risk as to quality
# and performance as well as the entire cost of service or repair in case of defect. See the EULA for more details.
#


#
# Copyright 2021 Intel Corporation.
# This software and the related documents are Intel copyrighted materials, and your use of them 
# is governed by the express license under which they were provided to you ("License"). Unless the 
# License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
# transmit this software or the related documents without Intel's prior written permission.
# 
# This software and the related documents are provided as is, with no express or implied warranties, 
# other than those that are expressly stated in the License.
# 
#


import argparse
import io
import math
import os
import re
import string
import subprocess #nosec subprocess is needed here to execute the sox tool.
import sys
import timeit
from pathlib import Path
from typing import Dict, Tuple, Union

import librosa
import numpy as np
import pyarrow
import scipy.io.wavfile as wav
import tensorflow as tf
import builtins as pybtin

# Spark
from horovod.spark.common.estimator import HorovodModel
from horovod.spark.common.store import HDFSStore
# Horovod
from horovod.spark.keras import KerasEstimator, KerasModel
from pyspark import RDD
from pyspark.ml.linalg import Vectors, VectorUDT, Vector, DenseVector
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import udf, lit, max, rand, when
from pyspark.sql.types import StringType, StructType, StructField, IntegerType
# Keras
from tensorflow.keras.backend import ctc_batch_cost, expand_dims, squeeze, ctc_decode
from tensorflow.keras.layers import Input, Masking, TimeDistributed, Dense, ReLU, Dropout, Bidirectional, LSTM, \
    Lambda, ZeroPadding2D, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

# GLOBALE DEFINITIONS
INPUT_SEPARATOR = '|'
OUTPUT_SEPARATOR = '|'
ALPHABET = ' \'' + string.ascii_lowercase
ALPHABET_DICT = {k: v for v, k in enumerate(ALPHABET)}

PAUSE_IN_MS = 20
SAMPLE_RATE = 16000
WINDOW_SIZE = 32
WINDOW_STRIDE = 20
N_MFCC = 26
N_HIDDEN = 1024
DROPOUT_RATE = 0.005
CONTEXT = 9
MAX_RELU = 20
MAX_SEQUENCE_LENGTH = 100  # max. number of symbols in a transcript

# DEFAULTS
BATCH_SIZE_DEFAULT = 32
EPOCHS_DEFAULT = 5

AUDIO_SCHEMA = StructType([
    StructField('rate', IntegerType()),
    StructField('data', VectorUDT())
])

FEATURES_SCHEMA = StructType([
    StructField('length', IntegerType()),
    StructField('data', VectorUDT())
])


def resample_audio(audio, original_sample_rate, desired_sample_rate):
    if original_sample_rate == desired_sample_rate:
        return original_sample_rate, audio
    cmd = f"sox - --type raw --bits 16 --channels 1 --rate {desired_sample_rate} --encoding signed-integer --endian little --compression 0.0 --no-dither - "
    f = io.BytesIO()
    wav.write(f, audio[0], audio[1])
    result = subprocess.run(cmd.split(), input=f.read(), stdout=subprocess.PIPE)
    if result.returncode == 0:
        return desired_sample_rate, np.frombuffer(result.stdout, dtype=np.int16)
    else:
        print(result.stdout, file=sys.stdout)
        print(result.stderr, file=sys.stderr)
        return 0, ""


def decode_sequence(sequence, alphabet: Dict):
    def lookup(k):
        try:
            return alphabet[k]
        except KeyError:
            return '_'

    decoded_sequence = list(map(lookup, sequence))
    return ''.join(decoded_sequence)


def load_data(session: SparkSession, path, audio_seq_file) -> DataFrame:
    meta_data = session.read.csv(path, sep=INPUT_SEPARATOR, inferSchema=True, header=True)
    meta_data = meta_data.repartition(session.sparkContext.defaultParallelism)
    audio_data_seq: RDD = session.sparkContext.sequenceFile(audio_seq_file)
    audio_data = session.createDataFrame(audio_data_seq, ['wav_filename', 'audio_bytes'])

    audio_data = audio_data.repartition(session.sparkContext.defaultParallelism)
    data = meta_data.join(audio_data, 'wav_filename')

    def wav_read(audio_bytes):
        audio = wav.read(io.BytesIO(audio_bytes))
        return audio[0], Vectors.dense(audio[1])

    wav_read_udf = udf(wav_read, AUDIO_SCHEMA)

    data = data.withColumn('audio', wav_read_udf(data.audio_bytes))
    if 'transcript' in data.columns:
        return data.select('wav_filename', 'transcript', 'audio').cache()
    else:
        return data.select('wav_filename', 'audio').cache()


def clean_data(data: DataFrame) -> DataFrame:
    if 'transcript' not in data.columns:
        return data
    # remove samples with no transcript
    pattern = re.compile(f"[^{ALPHABET}]")

    def normalize_transcript(trans):
        try:
            return pattern.sub('', trans.lower())
        except AttributeError:
            return pattern.sub('', str(trans).lower())

    normalize_transcript_udf = udf(normalize_transcript, StringType())

    data = data.withColumn('transcript_norm', normalize_transcript_udf(data.transcript))
    data = data.filter(data.transcript_norm.isNotNull())
    data = data.filter(data.transcript_norm != '')
    return data.cache()


def preprocess_data(data: DataFrame, max_timesteps=None) -> Tuple[DataFrame, int]:
    # resampling
    resample_audio_udf = udf(resample_audio, AUDIO_SCHEMA)
    data = data.withColumn('audio', resample_audio_udf(data.audio.data, data.audio.rate, lit(SAMPLE_RATE)))
    # possibly add silence to each utterance

    # LABELS
    if 'transcript' in data.columns:
        def text_to_seq(text):
            seq = []
            for c in text:
                seq.append(ALPHABET_DICT[c])
            seq = np.asarray(seq)
            return Vectors.dense(seq)

        def vector_length(vector):
            return vector.size

        def ctc_length(vector):
            repeated_chars = 0
            last_char = None
            for elem in vector:
                if last_char == elem:
                    repeated_chars += 1
                last_char = elem
            return vector.size + repeated_chars

        vector_length_udf = udf(vector_length, IntegerType())
        ctc_length_udf = udf(ctc_length, IntegerType())

        text_to_seq_udf = udf(text_to_seq, VectorUDT())
        data = data.withColumn('transcript_seq_raw', text_to_seq_udf(data.transcript_norm))
        data = data.withColumn('labels_len', vector_length_udf(data.transcript_seq_raw))
        data = data.withColumn('ctc_labels_len', ctc_length_udf(data.transcript_seq_raw))
        # filter all training samples where transcript has more than MAX_SEQUENCE_LENGTH symbols
        data = data[(data['labels_len'] <= MAX_SEQUENCE_LENGTH) & (data['labels_len'] > 0)]

    # FEATURES
    # calculate the mel spectograms windows_size 32 * 16, window_stride 20 * 16
    def to_mfcc(audio: Vector, sample_rate, win_size=WINDOW_SIZE, win_stride=WINDOW_STRIDE):
        # convert to float32 (-1.0 to 1.0)
        dtype = np.int16
        min_value = np.iinfo(dtype).min
        max_value = np.iinfo(dtype).max
        factor = 1 / np.max(np.abs([min_value, max_value]))
        y = audio.toArray()
        y = y * factor
        sr = sample_rate
        spectograms = librosa.feature.melspectrogram(y, sr, n_fft=win_size*sr//1000, hop_length=win_stride*sr//1000)
        mfccs = librosa.feature.mfcc(S=spectograms, sr=SAMPLE_RATE, n_mfcc=N_MFCC).transpose()
        # flatten the matrix of shape (timesteps, n_mfcc) to (timesteps * n_mfcc)
        return mfccs.shape[0], Vectors.dense(mfccs.flatten())

    # get the mfcc's (20 coeefficients) for the mel spectograms
    to_melspectograms_udf = udf(to_mfcc, FEATURES_SCHEMA)
    data = data.withColumn('features_raw', to_melspectograms_udf(data.audio.data, data.audio.rate))

    # padding transcripts and features
    data = data.cache()
    if not max_timesteps:
        max_timesteps = data.agg(max(data.features_raw.length)).collect()[0][0]

    def pad_features(features, max_len):
        zeros = np.zeros(max_len * N_MFCC)
        # copy everything
        if features.length < max_len:
            zeros[:features.length * N_MFCC] = features.data
        # copy only a slice
        else:
            zeros = features.data[:max_len * N_MFCC]

        return Vectors.dense(zeros)

    pad_features_udf = udf(pad_features, VectorUDT())
    data = data.withColumn('features', pad_features_udf(data.features_raw, lit(max_timesteps)))

    data = data.withColumn('features_length',
                           when(data.features_raw.length <= max_timesteps, data.features_raw.length).otherwise(max_timesteps))

    if 'transcript' in data.columns:
        def pad_sequence(sequence: Vector, max_len):
            seq_len = len(sequence)
            return Vectors.sparse(max_len, range(seq_len), sequence)

        pad_sequence_udf = udf(pad_sequence, VectorUDT())
        data = data.withColumn('transcript_seq', pad_sequence_udf(data.transcript_seq_raw, lit(MAX_SEQUENCE_LENGTH)))
        data = data[data.features_raw.length > data.ctc_labels_len]
        return data.select(data.wav_filename, data.features, data.features_length, data.transcript_seq, data.labels_len).cache(), max_timesteps
    else:
        return data.select(data.wav_filename, data.features, data.features_length).cache(), max_timesteps


def make_model_func(max_timesteps, n_hidden=N_HIDDEN, with_convolution=True):
    x = Input((max_timesteps, N_MFCC), name="X")
    y_true = Input((MAX_SEQUENCE_LENGTH,), name="y")
    seq_lengths = Input((1,), name="sequence_lengths")
    time_steps = Input((1,), name="time_steps")

    masking = Masking(mask_value=0)(x)

    if with_convolution:
        conv_layer = Lambda(lambda val: expand_dims(val, axis=-1))(masking)
        conv_layer = ZeroPadding2D(padding=(CONTEXT, 0))(conv_layer)
        conv_layer = Conv2D(filters=n_hidden, kernel_size=(2 * CONTEXT + 1, N_MFCC))(conv_layer)
        conv_layer = Lambda(squeeze, arguments=dict(axis=2))(conv_layer)
        conv_layer = ReLU(max_value=20)(conv_layer)
        conv_layer = Dropout(DROPOUT_RATE)(conv_layer)

        layer_1 = TimeDistributed(Dense(n_hidden))(conv_layer)
    else:
        layer_1 = TimeDistributed(Dense(n_hidden))(masking)

    layer_1 = ReLU(max_value=MAX_RELU)(layer_1)
    layer_1 = Dropout(DROPOUT_RATE)(layer_1)

    layer_2 = TimeDistributed(Dense(n_hidden))(layer_1)
    layer_2 = ReLU(max_value=MAX_RELU)(layer_2)
    layer_2 = Dropout(DROPOUT_RATE)(layer_2)

    lstm = Bidirectional(LSTM(n_hidden, return_sequences=True), merge_mode='sum')(layer_2)
    softmax = TimeDistributed(Dense(len(ALPHABET) + 1, activation='softmax'), name='prediction_softmax')(lstm)

    def myloss_layer(args):
        y_true, y_pred, time_steps, label_lengths = args
        return ctc_batch_cost(y_true, y_pred, time_steps, label_lengths)

    ctc_loss_layer = Lambda(myloss_layer, output_shape=(1,), name='ctc')([y_true, softmax, time_steps, seq_lengths])

    model = Model(inputs=[x, y_true, time_steps, seq_lengths], outputs=ctc_loss_layer)

    return model


def make_serving_model(training_model: Union[Model, KerasModel]) -> (KerasModel, int):
    _model = training_model.getModel()
    input_layer = _model.get_layer(name='X')
    max_timesteps = input_layer.input_shape[0][1]
    output_layer = _model.get_layer(name='prediction_softmax')
    predict_model = Model([input_layer.input], output_layer.output)
    meta_data = {
        'features': {
            'spark_data_type': DenseVector,
            'is_sparse_vector_only': False,
            'shape': None,
            'intermediate_format': 'array',
            'max_size': None
        },
        'transcript': {
            'spark_data_type': DenseVector,
            'is_sparse_vector_only': False,
            'shape': None,
            'intermediate_format': 'array',
            'max_size': None
        }
    }
    horovod_model = KerasModel(model=predict_model, feature_columns=['features'], label_columns=['transcript'],
                               _metadata=meta_data, _floatx='float32')
    return horovod_model, max_timesteps


def ctc_dummy(y_true, y_pred):
    mean = tf.reduce_mean(y_pred)
    return mean


def train(data: DataFrame, max_timesteps, batch_size, epochs, work_dir, defaultParallelism, current_user='',
          learning_rate=None) -> HorovodModel:
    model = make_model_func(with_convolution=False, max_timesteps=max_timesteps)
    model.summary(line_length=80)

    store = HDFSStore(work_dir, user=current_user)
    num_samples = data.count()
    num_processes = math.floor(min(defaultParallelism, num_samples * 0.8 / batch_size, num_samples * 0.2 / batch_size))
    num_processes = pybtin.max(num_processes, 1)

    lr = 0.001 / num_processes if not learning_rate else learning_rate
    optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    keras_estimator = KerasEstimator(model=model, num_proc=num_processes, store=store, loss=ctc_dummy, optimizer=optimizer,
                                     validation='validation', batch_size=batch_size, epochs=epochs,
                                     feature_cols=['features', 'transcript_seq', 'features_length', 'labels_len'],
                                     label_cols=['transcript_seq'],
                                     custom_objects={'ctc_batch_cost': ctc_batch_cost, 'ctc_dummy': ctc_dummy})
    data = data.repartition(num_processes)
    model = keras_estimator.fit(data)
    return model


def serve(model: KerasModel, x: DataFrame):
    pred = model.transform(x)
    pred = pred.withColumn('prediction', decode("transcript__output", "features_length"))
    return pred


@udf(returnType=StringType())
def decode(vector, input_length):
    probs = vector.toArray().reshape(-1, len(ALPHABET) + 1)
    decoded_seq = ctc_decode([probs], [input_length], greedy=False)[0][0].numpy()
    inv_alphabet = {v: k for k, v in ALPHABET_DICT.items()}
    return decode_sequence(decoded_seq[0], inv_alphabet)


def main():
    tqdm.pandas()

    model_file_name ="uc02.python.model"

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', metavar='SIZE', type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument('--epochs', metavar='N', type=int, default=EPOCHS_DEFAULT)
    parser.add_argument('--learning_rate', '-lr', required=False, type=float)
    parser.add_argument('--executor_cores_horovod', metavar='N', type=int, default=1)
    parser.add_argument('--task_cpus_horovod', metavar='N', type=int, default=1)

    parser.add_argument('--debug', action='store_true', required=False)
    parser.add_argument('--stage', choices=['training', 'serving'], metavar='stage', required=True)
    parser.add_argument('--workdir', metavar='workdir', required=True)
    parser.add_argument('--output', metavar='output', required=False)
    parser.add_argument("filename")
    parser.add_argument("audio_seq_file")

    args = parser.parse_args()
    batch_size = args.batch
    epochs = args.epochs
    learning_rate = args.learning_rate if args.learning_rate else None
    task_cpus_horovod = args.task_cpus_horovod
    executor_cores_horovod = args.executor_cores_horovod

    path = args.filename
    audio_seq_file = args.audio_seq_file
    stage = args.stage
    work_dir = Path(args.workdir)

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    preprocessed_data_path = f"{work_dir}/tmp/preprocessed.parquet"

    spark = SparkSession.Builder().appName('UC02-spark').getOrCreate()

    current_user = spark.sparkContext.sparkUser()
    if not os.path.isabs(work_dir):
        work_dir = f"hdfs:///user/{current_user}/{work_dir}"

    if not args.output:
        output = work_dir
    else:
        output = args.output

    start = timeit.default_timer()
    raw_data = load_data(spark, path, audio_seq_file)
    end = timeit.default_timer()
    load_time = end - start
    print('load time:\t', load_time)

    if stage == 'training':
        start = timeit.default_timer()
        cleaned_data = clean_data(raw_data)
        preprocessed_data, max_timesteps = preprocess_data(cleaned_data)

        # mark: train / validation split (80/20)
        preprocessed_data = preprocessed_data.withColumn('validation', when(rand(seed=0xC0FFE) <= 0.2, True).otherwise(False))

        end = timeit.default_timer()
        pre_process_time = end - start
        print('pre-process time:\t', pre_process_time)

        start = timeit.default_timer()
        preprocessed_data.write.mode('overwrite').parquet(preprocessed_data_path)
        spark.stop()
        spark_horovod = SparkSession.Builder() \
            .appName(f"UC02-spark-horovod") \
            .config("spark.executor.cores", executor_cores_horovod) \
            .config("spark.task.cpus", task_cpus_horovod) \
            .getOrCreate()
        training_data = spark_horovod.read.parquet(preprocessed_data_path)
        model = train(training_data, max_timesteps, batch_size, epochs, work_dir,
                      spark_horovod.sparkContext.defaultParallelism, current_user, learning_rate)
        end = timeit.default_timer()
        train_time = end - start
        print('train time:\t', train_time)

        model.write().overwrite().save(f"{work_dir}/{model_file_name}")

    elif stage == 'serving':
        model = KerasModel.load(f"{work_dir}/{model_file_name}")
        serving_model, max_timesteps = make_serving_model(model)

        start = timeit.default_timer()
        cleaned_data = clean_data(raw_data)
        preprocessed_data, _ = preprocess_data(cleaned_data, max_timesteps=max_timesteps)
        end = timeit.default_timer()
        pre_process_time = end - start
        print('pre-process time:\t', pre_process_time)

        start = timeit.default_timer()
        prediction = serve(serving_model, preprocessed_data)
        end = timeit.default_timer()
        serve_time = end - start
        print('serve time:\t', serve_time)
        prediction\
            .select('wav_filename', 'prediction')\
            .withColumnRenamed('prediction', 'transcript')\
            .write.mode('overwrite').option('quoteAll', True).csv(f"{output}/predictions.csv", header=True, sep='|')


if __name__ == '__main__':
    main()
