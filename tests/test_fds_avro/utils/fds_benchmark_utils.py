# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""Utility functions for FDS benchmarks."""

import avro.schema
import glob
import json
import os
import tensorflow as tf

from tensorflow_io.core.python.experimental.benchmark.data_source import \
  DataSource
from tensorflow_io.core.python.experimental.benchmark.data_source_registry import \
  SMALL_NUM_RECORDS, get_canonical_name, get_data_source_from_registry
from tensorflow_io.core.python.experimental.benchmark.generator.tensor_generator import \
  IntTensorGenerator, FloatTensorGenerator, WordTensorGenerator, BoolTensorGenerator
from tensorflow_io.core.python.experimental.benchmark.generator.sparse_tensor_generator import \
  IntSparseTensorGenerator, FloatSparseTensorGenerator, WordSparseTensorGenerator, BoolSparseTensorGenerator, \
    get_common_value_dist
from tensorflow_io.core.python.experimental.benchmark.generator.varlen_tensor_generator import \
  IntVarLenTensorGenerator, FloatVarLenTensorGenerator, WordVarLenTensorGenerator, BoolVarLenTensorGenerator
from tensorflow_io.core.python.experimental.fds.dataset import FDSDataset
from tensorflow_io.core.python.experimental.fds.fds_writer import \
  FDSWriter
from tests.test_benchmark.benchmark.utils.benchmark_utils import benchmark_func


FEATURE_COLUMN_KEY = 'com.linkedin.quince.featurizeddataset.FeatureColumnMetadata'

_AVRO_TO_DTYPE = {
  "int": tf.int32, "long": tf.int64, "float": tf.float32, "double": tf.float64, "boolean": tf.bool, "string": tf.string
}

_AVRO_TO_DENSE_TENSOR_GENERATOR = {
  "int": IntTensorGenerator, "long": IntTensorGenerator, "float": FloatTensorGenerator,
  "double": FloatTensorGenerator, "boolean": BoolTensorGenerator, "string": WordTensorGenerator
}

_AVRO_TO_SPARSE_TENSOR_GENERATOR = {
  "int": IntSparseTensorGenerator, "long": IntSparseTensorGenerator, "float": FloatSparseTensorGenerator,
  "double": FloatSparseTensorGenerator, "boolean": BoolSparseTensorGenerator, "string": WordSparseTensorGenerator
}

_AVRO_TO_VARLEN_TENSOR_GENERATOR = {
  "int": IntVarLenTensorGenerator, "long": IntVarLenTensorGenerator, "float": FloatVarLenTensorGenerator,
  "double": FloatVarLenTensorGenerator, "boolean": BoolVarLenTensorGenerator, "string": WordVarLenTensorGenerator
}

def get_features_from_data_source(writer, data_source):
  """Generates a dict of features from data source object

  Args:
    writer: FDSWriter object
    data_source: DataSource object
  """
  scenario = data_source.scenario
  features = {
    feature_name: writer._get_fds_feature(scenario[feature_name]) for feature_name in scenario
  }
  return features

def get_dataset(files, features, batch_size=1, shuffle_buffer_size=0, parallelism=os.cpu_count(),
    interleave_parallelism=0):
  """Generates a tf.data.Dataset from a datasource

  Args:
    files: A list of files
    features: Dict of features
    batch_size: (Optional.) Batch size for FDS dataset
    shuffle_buffer_size: (Optional.) Size of the buffer used for shuffling. See
        tensorflow_io/core/python/experimental/fds/dataset.py for details.
        If unspecified, data is not shuffled.
    parallelism: (Optional.) Number of threads to use while decoding. Defaults
        to all available cores.
  """
  if interleave_parallelism == 0:
    dataset = FDSDataset(
      filenames=files,
      batch_size=batch_size,
      features=features,
      shuffle_buffer_size=shuffle_buffer_size,
      num_parallel_calls=parallelism
    )
  else:
    dataset = tf.data.Dataset.list_files(files)
    dataset = dataset.interleave(
        lambda filename: FDSDataset(filenames=filename,
                                    batch_size=batch_size,
                                    features=features,
                                    shuffle_buffer_size=shuffle_buffer_size,
                                    num_parallel_calls=parallelism),
        cycle_length=interleave_parallelism,
        num_parallel_calls=interleave_parallelism
    )
  return dataset.prefetch(1)

def _is_fully_defined_shape(shape):
  return -1 not in shape

def convert_schema_to_data_source(schema_file_path, num_records=SMALL_NUM_RECORDS):
  schema = avro.schema.Parse(open(schema_file_path, "rb").read())
  scenario = {}
  metadata = json.loads(schema.to_json()['featurizedDatasetMetadata'])
  for feature_name in metadata['columnMetadata']:
    feature_metadata = metadata['columnMetadata'][feature_name]['metadata']
    if FEATURE_COLUMN_KEY in feature_metadata:
      feature_metadata = feature_metadata[FEATURE_COLUMN_KEY]
    else:
      # Column metadata may contain opaque contextual columns not used for training (see go/qt-fds),
      # so we ignore them when converting schema to data source.
      continue
    feature_shape = feature_metadata['tensorShape']
    tensor_type = feature_metadata['tensorCategory'].lower()
    feature_dtype = feature_metadata['valueType'].lower()
    tf_dtype = _AVRO_TO_DTYPE[feature_dtype]
    if tensor_type == 'dense' and _is_fully_defined_shape(feature_shape):
      generator_cls = _AVRO_TO_DENSE_TENSOR_GENERATOR[feature_dtype]
      scenario[feature_name] = generator_cls(tf.TensorSpec(shape=feature_shape, dtype=tf_dtype))
    elif tensor_type == 'sparse':
      generator_cls = _AVRO_TO_SPARSE_TENSOR_GENERATOR[feature_dtype]
      sparse_shape = [dim if dim != -1 else None for dim in feature_shape]
      scenario[feature_name] = generator_cls(tf.SparseTensorSpec(shape=sparse_shape, dtype=tf_dtype),
          get_common_value_dist())
    elif tensor_type == 'dense' or tensor_type == 'ragged':
      generator_cls = _AVRO_TO_VARLEN_TENSOR_GENERATOR[feature_dtype]
      ragged_shape = [dim if dim != -1 else None for dim in feature_shape]
      scenario[feature_name] = generator_cls(tf.SparseTensorSpec(shape=ragged_shape, dtype=tf_dtype))
    else:
      raise ValueError(f"Feature {feature_name} must be either dense, sparse, or ragged; got {tensor_type}.")
  return DataSource(
    scenario=scenario,
    num_records=num_records
  )

def run_fds_benchmark(tensor_type, rank, dtype, num_records, partitions, batch_size, benchmark):
  data_source_name = get_canonical_name(tensor_type, rank, dtype, num_records, partitions)
  data_source = get_data_source_from_registry(data_source_name)
  run_fds_benchmark_from_data_source(data_source, batch_size, benchmark)

def run_fds_benchmark_from_data_source(data_source, batch_size, benchmark,
    parallelism=tf.data.AUTOTUNE, interleave_parallelism=0, codec="null", shuffle_buffer_size=0, rounds=30):
  with FDSWriter(codec=codec) as writer:
    dir_path = writer.write(data_source)
    pattern = os.path.join(dir_path, f"*.{writer.extension}")

    dataset = get_dataset(glob.glob(pattern), get_features_from_data_source(writer, data_source),
        batch_size=batch_size, shuffle_buffer_size=shuffle_buffer_size, parallelism=parallelism,
        interleave_parallelism=interleave_parallelism)
    count = benchmark.pedantic(
        target=benchmark_func,
        args=[dataset],
        iterations=2,
        # pytest-benchmark calculates statistic across rounds. Set it with
        # larger number (N > 30) for test statistic.
        rounds=rounds,
        kwargs={}
    )
    assert count > 0, f"FDS record count: {count} must be greater than 0"
