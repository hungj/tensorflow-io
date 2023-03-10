#include "tensorflow_io/core/kernels/avro/fds/fds_decoder.h"
#include "tensorflow_io/core/kernels/avro/fds/decoder_test_util.h"

#include "tensorflow/core/platform/test.h"

#include "api/Decoder.hh"
#include "api/Stream.hh"

namespace tensorflow {
namespace fds {
namespace dense {

template<typename T>
void DenseDecoderTest(const T& values, DataType dtype, std::initializer_list<int64> shape, const avro::Type avro_type=avro::AVRO_NULL) {
  string feature_name = "feature";
  FDSSchemaBuilder schema_builder = FDSSchemaBuilder();
  schema_builder.AddDenseFeature(feature_name, dtype, shape.size(), avro_type);

  string schema = schema_builder.Build();
  avro::ValidSchema writer_schema = schema_builder.BuildVaildSchema();
  avro::GenericDatum fds_datum(writer_schema);
  AddDenseValue(fds_datum, feature_name, values);
  avro::OutputStreamPtr out_stream = EncodeAvroGenericDatum(fds_datum);
  avro::InputStreamPtr in_stream = avro::memoryInputStream(*out_stream);
  avro::DecoderPtr decoder = avro::binaryDecoder();
  decoder->init(*in_stream);

  std::vector<dense::Metadata> dense_features;
  std::vector<sparse::Metadata> sparse_features;
  std::vector<varlen::Metadata> varlen_features;
  size_t pos = 0;
  PartialTensorShape tensor_shape(shape);
  dense_features.emplace_back(FeatureType::dense, feature_name, dtype, tensor_shape, pos);

  FDSDecoder fds_decoder = FDSDecoder(dense_features, sparse_features, varlen_features);
  Status init_status = fds_decoder.Initialize(writer_schema);
  ASSERT_TRUE(init_status.ok());

  sparse::ValueBuffer buffer;
  std::vector<avro::GenericDatum> skipped_data = fds_decoder.GetSkippedData();
  std::vector<Tensor> dense_tensors;
  dense_tensors.emplace_back(dtype, TensorShape(shape));
  size_t offset = 0;

  Status decode_status = fds_decoder.DecodeFDSDatum(decoder, dense_tensors, buffer,
                                                    skipped_data, offset);
  ASSERT_TRUE(decode_status.ok());
  const Tensor tensor = dense_tensors[pos];
  AssertTensorValues(tensor, values);
}

TEST(DenseDecoderTest, DT_INT32_scalar) {
  int value = -7;
  DenseDecoderTest(value, DT_INT32, {});
}

TEST(DenseDecoderTest, DT_INT32_1D) {
  std::vector<int> values = {1, 2, 3};
  DenseDecoderTest(values, DT_INT32, {3});
}

TEST(DenseDecoderTest, DT_INT32_2D) {
  std::vector<std::vector<int>> values = {
    {-1, -2, -3}, {4, 5, 6}, {-7, 8, 9}};
  DenseDecoderTest(values, DT_INT32, {3, 3});
}

TEST(DenseDecoderTest, DT_INT64_scalar) {
  long value = 1;
  DenseDecoderTest(value, DT_INT64, {});
}

TEST(DenseDecoderTest, DT_INT64_1D) {
  std::vector<long> values = {1};
  DenseDecoderTest(values, DT_INT64, {1});
}

TEST(DenseDecoderTest, DT_INT64_2D) {
  std::vector<std::vector<long>> values = {{1}};
  DenseDecoderTest(values, DT_INT64, {1, 1});
}

TEST(DenseDecoderTest, DT_FLOAT_scalar) {
  float value = -0.6;
  DenseDecoderTest(value, DT_FLOAT, {});
}

TEST(DenseDecoderTest, DT_FLOAT_1D) {
  std::vector<float> values = {1.5, 0.5, 1.7, 2.6};
  DenseDecoderTest(values, DT_FLOAT, {4});
}

TEST(DenseDecoderTest, DT_FLOAT_2D) {
  std::vector<std::vector<float>> values = {
    {-0.1, -0.2, -0.3}, {-1.4, 5.4, 6.6}};
  DenseDecoderTest(values, DT_FLOAT, {2, 3});
}

TEST(DenseDecoderTest, DT_DOUBLE_scalar) {
  double value = -0.99;
  DenseDecoderTest(value, DT_DOUBLE, {});
}

TEST(DenseDecoderTest, DT_DOUBLE_1D) {
  std::vector<double> values = {1.852, 0.79};
  DenseDecoderTest(values, DT_DOUBLE, {2});
}

TEST(DenseDecoderTest, DT_DOUBLE_2D) {
  std::vector<std::vector<double>> values = {{-3.14, -2.07}};
  DenseDecoderTest(values, DT_DOUBLE, {1, 2});
}

TEST(DenseDecoderTest, DT_STRING_scalar) {
  string value = "abc";
  DenseDecoderTest(value, DT_STRING, {});
}

TEST(DenseDecoderTest, DT_STRING_1D) {
  std::vector<string> values = {"", "", ""};
  DenseDecoderTest(values, DT_STRING, {3});
}

TEST(DenseDecoderTest, DT_STRING_2D) {
  std::vector<std::vector<string>> values = {
    {"abc"}, {"ABC"}, {"LINKEDIN"}};
  DenseDecoderTest(values, DT_STRING, {3, 1});
}

TEST(DenseDecoderTest, DT_BYTES_scalar) {
  byte_array value{0xb4,0xaf,0x98,0x1a};
  DenseDecoderTest(value, DT_STRING, {}, avro::AVRO_BYTES);
}

TEST(DenseDecoderTest, DT_BYTES_1D) {
  byte_array v1{0xb4,0xaf,0x98,0x1a};
  byte_array v2{0xb4,0xaf,0x98};
  byte_array v3{0xb4,0x98,0x1a};
  std::vector<byte_array> values = {v1, v2, v3};
  DenseDecoderTest(values, DT_STRING, {3}, avro::AVRO_BYTES);
}

TEST(DenseDecoderTest, DT_BYTES_2D) {
  byte_array v1{0xb4,0xaf,0x98,0x1a};
  byte_array v2{0xb4,0xaf,0x98};
  byte_array v3{0xb4,0x98,0x1a};
  std::vector<std::vector<byte_array>> values = {
    {v1}, {v2}, {v2}};
  DenseDecoderTest(values, DT_STRING, {3, 1}, avro::AVRO_BYTES);
}

TEST(DenseDecoderTest, DT_BOOL_scalar) {
  bool value = true;
  DenseDecoderTest(value, DT_BOOL, {});
}

TEST(DenseDecoderTest, DT_BOOL_1D) {
  std::vector<bool> values = {true, false, true};
  DenseDecoderTest(values, DT_BOOL, {3});
}

TEST(DenseDecoderTest, DT_BOOL_2D) {
  std::vector<std::vector<bool>> values = {
    {false, false}, {true, true}};
  DenseDecoderTest(values, DT_BOOL, {2, 2});
}

}  // namespace dense
}  // namespace fds
}  // namespace tensorflow
