#include "tensorflow_io/core/kernels/avro/fds/fds_decoder.h"
#include "tensorflow_io/core/kernels/avro/fds/dense_feature_decoder.h"
#include "tensorflow_io/core/kernels/avro/fds/decoder_test_util.h"

#include "tensorflow/core/platform/test.h"

#include "api/Decoder.hh"
#include "api/GenericDatum.hh"
#include "api/Stream.hh"
#include "api/ValidSchema.hh"

namespace tensorflow {
namespace fds {

TEST(FDSDecoder, TestMixedFeatures) {
  std::vector<string> feature_names = {
    "dense_float_1d", "dense_long_2d", "unused_dense",
    "sparse_int_1d", "unsed_sparse", "sparse_string_2d",
    "unused_varlen", "varlen_bool_1d", "varlen_string_2d"};
  std::vector<size_t> feature_pos = {0, 1, 0, 1, 2, 3};
  std::vector<std::initializer_list<int64>> feature_shapes = {
    {3}, {2, 2}, {101}, {6, 10}, {-1}, {-1, -1}};
  std::vector<PartialTensorShape> tensor_shapes;
  for (auto shape : feature_shapes) {
    tensor_shapes.emplace_back(shape);
  }

  FDSSchemaBuilder schema_builder = FDSSchemaBuilder();
  schema_builder.AddDenseFeature(feature_names[0], DT_FLOAT, 1)
                .AddDenseFeature(feature_names[1], DT_INT64, 2)
                .AddDenseFeature(feature_names[2], DT_FLOAT, 2)  // unused
                .AddSparseFeature(feature_names[3], DT_INT32, 1)
                .AddSparseFeature(feature_names[4], DT_DOUBLE, 1)  // unused
                .AddSparseFeature(feature_names[5], DT_STRING, 2)
                .AddDenseFeature(feature_names[6], DT_BOOL, 0)  // unused
                .AddDenseFeature(feature_names[7], DT_BOOL, 1)
                .AddDenseFeature(feature_names[8], DT_STRING, 2);

  string schema = schema_builder.Build();
  avro::ValidSchema writer_schema = schema_builder.BuildVaildSchema();

  avro::GenericDatum fds_datum(writer_schema);
  std::vector<float> dense_float_1d = {1.0, 2.0, 3.0};
  std::vector<std::vector<long>> dense_long_2d = {{1, 3}, {2, 4}};
  std::vector<std::vector<float>> unused_dense = {{1.0, 2.0}};

  std::vector<std::vector<long>> sparse_int_1d_indices = {{100}};
  std::vector<int> sparse_int_1d_values = {100};
  std::vector<std::vector<long>> sparse_string_2d_indices = {{5, 5}, {4, 8}};
  std::vector<string> sparse_string_2d_values = {"TensorFlow", "Linkedin"};
  std::vector<std::vector<long>> unsed_sparse_indices = {{0, 1}};
  std::vector<double> unsed_sparse_values = {1.0, -1.0};

  std::vector<bool> varlen_bool_1d = {true, false, true};
  std::vector<std::vector<string>> varlen_string_2d = {{"ABC"}, {}, {"DEF"}};
  std::vector<string> expected_varlen_string_2d_values = {"ABC", "DEF"};
  bool unused_varlen = true;

  AddDenseValue(fds_datum, feature_names[0], dense_float_1d);
  AddDenseValue(fds_datum, feature_names[1], dense_long_2d);
  AddDenseValue(fds_datum, feature_names[2], unused_dense);
  AddSparseValue(fds_datum, feature_names[3], sparse_int_1d_indices, sparse_int_1d_values);
  AddSparseValue(fds_datum, feature_names[4], unsed_sparse_indices, unsed_sparse_values);
  AddSparseValue(fds_datum, feature_names[5], sparse_string_2d_indices, sparse_string_2d_values);
  AddDenseValue(fds_datum, feature_names[6], unused_varlen);
  AddDenseValue(fds_datum, feature_names[7], varlen_bool_1d);
  AddDenseValue(fds_datum, feature_names[8], varlen_string_2d);

  avro::OutputStreamPtr out_stream = EncodeAvroGenericDatum(fds_datum);
  avro::InputStreamPtr in_stream = avro::memoryInputStream(*out_stream);
  avro::DecoderPtr decoder = avro::binaryDecoder();
  decoder->init(*in_stream);

  std::vector<dense::Metadata> dense_features;
  dense_features.emplace_back(FeatureType::dense, feature_names[0], DT_FLOAT,
                              tensor_shapes[0], feature_pos[0]);
  dense_features.emplace_back(FeatureType::dense, feature_names[1], DT_INT64,
                              tensor_shapes[1], feature_pos[1]);

  size_t values_index = 0;
  std::vector<sparse::Metadata> sparse_features;
  sparse_features.emplace_back(FeatureType::sparse, feature_names[3], DT_INT32,
                               tensor_shapes[2], feature_pos[2], values_index);
  sparse_features.emplace_back(FeatureType::sparse, feature_names[5], DT_STRING,
                               tensor_shapes[3], feature_pos[3], values_index);

  std::vector<varlen::Metadata> varlen_features;
  size_t string_value_index = 1;  // index 0 is used by sparse_string_2d.
  varlen_features.emplace_back(FeatureType::varlen, feature_names[7], DT_BOOL,
                               tensor_shapes[4], feature_pos[4], values_index);
  varlen_features.emplace_back(FeatureType::varlen, feature_names[8], DT_STRING,
                               tensor_shapes[5], feature_pos[5], string_value_index);

  FDSDecoder fds_decoder = FDSDecoder(dense_features, sparse_features, varlen_features);
  Status init_status = fds_decoder.Initialize(writer_schema);
  ASSERT_TRUE(init_status.ok());

  std::vector<Tensor> dense_tensors;
  dense_tensors.emplace_back(DT_FLOAT, TensorShape(feature_shapes[0]));
  dense_tensors.emplace_back(DT_INT64, TensorShape(feature_shapes[1]));

  sparse::ValueBuffer buffer;
  buffer.indices.resize(4);
  buffer.num_of_elements.resize(4);
  buffer.string_values.resize(2);
  buffer.int_values.resize(1);
  buffer.bool_values.resize(1);

  std::vector<avro::GenericDatum> skipped_data = fds_decoder.GetSkippedData();
  long offset = 0;
  Status decode_status = fds_decoder.DecodeFDSDatum(decoder, dense_tensors, buffer,
                                                    skipped_data, static_cast<size_t>(offset));
  ASSERT_TRUE(decode_status.ok());
  AssertTensorValues(dense_tensors[0], dense_float_1d);
  AssertTensorValues(dense_tensors[1], dense_long_2d);
  ValidateBuffer(buffer, sparse_features[0], {offset, 100}, sparse_int_1d_values, {1});
  ValidateBuffer(buffer, sparse_features[1], {offset, 5, 4, offset, 5, 8}, sparse_string_2d_values, {2});
  ValidateBuffer(buffer, varlen_features[0], {offset, 0, offset, 1, offset, 2}, varlen_bool_1d, {3});
  ValidateBuffer(buffer, varlen_features[1], {offset, 0, 0, offset, 2, 0}, expected_varlen_string_2d_values, {2});
}

}  // namespace fds
}  // namespace tensorflow
