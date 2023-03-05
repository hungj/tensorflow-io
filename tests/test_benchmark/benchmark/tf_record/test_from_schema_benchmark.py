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
"""TFRecordDataset benchmark from provided schema file."""

import pytest

from tests.test_benchmark.benchmark.utils.benchmark_utils import \
  run_tf_record_benchmark_from_data_source
from tests.test_fds_avro.utils.fds_benchmark_utils import \
  convert_schema_to_data_source


@pytest.mark.benchmark(group="from_schema",)
@pytest.mark.parametrize("batch_size", [(128)])
def test_from_schema_file(schemafile, batch_size, benchmark):
  data_source = convert_schema_to_data_source(schemafile)
  run_tf_record_benchmark_from_data_source(data_source, batch_size, benchmark)
