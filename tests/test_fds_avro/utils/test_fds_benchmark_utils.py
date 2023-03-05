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
"""Test FDS utility functions for benchmarks."""

import os
import pytest
import re
import tempfile
import tensorflow as tf

from tests.test_fds_avro.utils.fds_benchmark_utils import convert_schema_to_data_source
from tensorflow_io.python.experimental.benchmark.generator.tensor_generator import \
  IntTensorGenerator, WordTensorGenerator
from tensorflow_io.python.experimental.benchmark.generator.sparse_tensor_generator import \
  IntSparseTensorGenerator, FloatSparseTensorGenerator, BoolSparseTensorGenerator
from tensorflow_io.python.experimental.benchmark.generator.varlen_tensor_generator import \
  IntVarLenTensorGenerator, FloatVarLenTensorGenerator, BoolVarLenTensorGenerator

def test_convert_schema_to_data_source():
  schema = """{
    "featurizedDatasetMetadata": "{\\"topLevelMetadata\\":{\\"fdsSchemaVersion\\":\\"V0\\"},\\"columnMetadata\\":{\\"dense_int_0d\\":{\\"metadata\\":{\\"com.linkedin.quince.featurizeddataset.FeatureColumnMetadata\\":{\\"tensorShape\\":[],\\"dimensionTypes\\":[],\\"tensorCategory\\":\\"DENSE\\",\\"valueType\\":\\"INT\\"}}},\\"dense_double_1d_unknown_dim\\":{\\"metadata\\":{\\"com.linkedin.quince.featurizeddataset.FeatureColumnMetadata\\":{\\"tensorShape\\":[-1],\\"dimensionTypes\\":[\\"INT\\"],\\"valueType\\":\\"DOUBLE\\",\\"tensorCategory\\":\\"DENSE\\"}}},\\"dense_string_2d\\":{\\"metadata\\":{\\"com.linkedin.quince.featurizeddataset.FeatureColumnMetadata\\":{\\"tensorShape\\":[3, 4],\\"dimensionTypes\\":[],\\"tensorCategory\\":\\"DENSE\\",\\"valueType\\":\\"STRING\\"}}},\\"sparse_long_1d\\":{\\"metadata\\":{\\"com.linkedin.quince.featurizeddataset.FeatureColumnMetadata\\":{\\"tensorShape\\":[3],\\"dimensionTypes\\":[\\"LONG\\"],\\"valueType\\":\\"LONG\\",\\"tensorCategory\\":\\"SPARSE\\"}}},\\"sparse_float_2d\\":{\\"metadata\\":{\\"com.linkedin.quince.featurizeddataset.FeatureColumnMetadata\\":{\\"tensorShape\\":[10, 20],\\"dimensionTypes\\":[\\"LONG\\"],\\"valueType\\":\\"FLOAT\\",\\"tensorCategory\\":\\"SPARSE\\"}}},\\"sparse_bool_1d_unknown_dim\\":{\\"metadata\\":{\\"com.linkedin.quince.featurizeddataset.FeatureColumnMetadata\\":{\\"tensorShape\\":[-1],\\"dimensionTypes\\":[\\"INT\\"],\\"valueType\\":\\"BOOLEAN\\",\\"tensorCategory\\":\\"SPARSE\\"}}},\\"ragged_bool_1d\\":{\\"metadata\\":{\\"com.linkedin.quince.featurizeddataset.FeatureColumnMetadata\\":{\\"tensorShape\\":[-1],\\"dimensionTypes\\":[\\"LONG\\"],\\"valueType\\":\\"BOOLEAN\\",\\"tensorCategory\\":\\"RAGGED\\"}}},\\"ragged_long_2d\\":{\\"metadata\\":{\\"com.linkedin.quince.featurizeddataset.FeatureColumnMetadata\\":{\\"tensorShape\\":[-1, 2],\\"dimensionTypes\\":[\\"INT\\"],\\"valueType\\":\\"LONG\\",\\"tensorCategory\\":\\"RAGGED\\"}}},\\"opaque_column\\":{\\"metadata\\":{\\"com.linkedin.quince.featurizeddataset.OpaqueContextualColumnMetadata\\":{}}}}}",
    "fields": [
        {
            "name": "dense_int_0d",
            "type": [
                "null",
                "int"
            ]
        },
        {
            "name": "dense_double_1d_unknown_dim",
            "type": [
                "null",
                {
                    "items": "double",
                    "type": "array"
                }
            ]
        },
        {
            "name": "dense_string_2d",
            "type": [
                "null",
                {
                  "type": "array",
                  "items": {
                    "type": "array",
                    "items": "string"
                  }
                }
            ]
        },
        {
            "name": "sparse_long_1d",
            "type": [
                "null",
                {
                    "fields": [
                        {
                            "name": "indices0",
                            "type": {
                                "items": "long",
                                "type": "array"
                            }
                        },
                        {
                            "name": "values",
                            "type": {
                                "items": "long",
                                "type": "array"
                            }
                        }
                    ],
                    "name": "sparse_long_1d_feature",
                    "namespace": "com.linkedin.quince.featurizeddataset.FeaturizedDataset",
                    "type": "record"
                }
            ]
        },
        {
            "name": "sparse_float_2d",
            "type": [
                "null",
                {
                    "fields": [
                        {
                            "name": "indices0",
                            "type": {
                                "items": "long",
                                "type": "array"
                            }
                        },
                        {
                            "name": "indices1",
                            "type": {
                                "items": "long",
                                "type": "array"
                            }
                        },
                        {
                            "name": "values",
                            "type": {
                                "items": "float",
                                "type": "array"
                            }
                        }
                    ],
                    "name": "sparse_float_2d_feature",
                    "namespace": "com.linkedin.quince.featurizeddataset.FeaturizedDataset",
                    "type": "record"
                }
            ]
        },
        {
            "name": "sparse_bool_1d_unknown_dim",
            "type": [
                "null",
                {
                    "fields": [
                        {
                            "name": "indices0",
                            "type": {
                                "items": "long",
                                "type": "array"
                            }
                        },
                        {
                            "name": "values",
                            "type": {
                                "items": "boolean",
                                "type": "array"
                            }
                        }
                    ],
                    "name": "sparse_bool_1d_unknown_dim_feature",
                    "namespace": "com.linkedin.quince.featurizeddataset.FeaturizedDataset",
                    "type": "record"
                }
            ]
        },
        {
            "name": "ragged_bool_1d",
            "type": [
                "null",
                {
                    "items": "boolean",
                    "type": "array"
                }
            ]
        },
        {
            "name": "ragged_long_2d",
            "type": [
                "null",
                {
                  "type": "array",
                  "items": {
                    "type": "array",
                    "items": "long"
                  }
                }
            ]
        },
        {
            "name": "opaque_column",
            "type": "string"
        }
    ],
    "name": "FeaturizedDataset",
    "namespace": "com.linkedin.quince.featurizeddataset",
    "type": "record"
  }
  """
  schema_file, file_path = tempfile.mkstemp()
  os.write(schema_file, str.encode(schema))
  os.close(schema_file)
  data_source = convert_schema_to_data_source(file_path)
  scenario = data_source.scenario
  _assert_feature(scenario, 'dense_int_0d', [], tf.int32, IntTensorGenerator)
  _assert_feature(scenario, 'dense_double_1d_unknown_dim', [None], tf.float64, FloatVarLenTensorGenerator)
  _assert_feature(scenario, 'dense_string_2d', [3, 4], tf.string, WordTensorGenerator)
  _assert_feature(scenario, 'sparse_long_1d', [3], tf.int64, IntSparseTensorGenerator)
  _assert_feature(scenario, 'sparse_float_2d', [10, 20], tf.float32, FloatSparseTensorGenerator)
  _assert_feature(scenario, 'sparse_bool_1d_unknown_dim', [None], tf.bool, BoolSparseTensorGenerator)
  _assert_feature(scenario, 'ragged_bool_1d', [None], tf.bool, BoolVarLenTensorGenerator)
  _assert_feature(scenario, 'ragged_long_2d', [None, 2], tf.int64, IntVarLenTensorGenerator)
  assert 'opaque_column' not in scenario

def test_unsupported_feature_type():
  schema = """{
    "featurizedDatasetMetadata": "{\\"topLevelMetadata\\":{\\"fdsSchemaVersion\\":\\"V0\\"},\\"columnMetadata\\":{\\"varlen_feature\\":{\\"metadata\\":{\\"com.linkedin.quince.featurizeddataset.FeatureColumnMetadata\\":{\\"tensorShape\\":[2, -1],\\"dimensionTypes\\":[],\\"tensorCategory\\":\\"VARLEN\\",\\"valueType\\":\\"INT\\"}}}}}",
    "fields": [
        {
            "name": "varlen_feature",
            "type": [
                "null",
                {
                  "type": "array",
                  "items": {
                    "type": "array",
                    "items": "int"
                  }
                }
            ]
        }
    ],
    "name": "FeaturizedDataset",
    "namespace": "com.linkedin.quince.featurizeddataset",
    "type": "record"
  }
  """
  schema_file, file_path = tempfile.mkstemp()
  os.write(schema_file, str.encode(schema))
  os.close(schema_file)
  error_message = "Feature varlen_feature must be either dense, sparse, or ragged; got varlen."
  with pytest.raises(ValueError, match=re.escape(error_message)):
    convert_schema_to_data_source(file_path)

def _assert_feature(scenario, feature_name, expected_shape, expected_dtype, expected_generator_cls):
  generator = scenario[feature_name]
  assert isinstance(generator, expected_generator_cls)
  for generator_dim, expected_dim in zip(generator.spec.shape, expected_shape):
    # tf.TensorShape([None]) and tf.TensorShape([None]) are not equal, so compare
    # dimensions instead of shape.
    assert generator_dim == expected_dim
  assert generator.spec.dtype == expected_dtype
