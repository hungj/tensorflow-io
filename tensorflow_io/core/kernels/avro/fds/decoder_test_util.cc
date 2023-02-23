#include "tensorflow_io/core/kernels/avro/fds/decoder_test_util.h"

#include "api/Compiler.hh"
#include "api/Generic.hh"
#include "api/Specific.hh"
#include "api/ValidSchema.hh"

namespace tensorflow {
namespace fds {

constexpr const char kFDSSchemaPrefix[] =
  "{"
    "\"type\" : \"record\", "
    "\"name\" : \"FeaturizedDataset\", "
    "\"namespace\" : \"com.linkedin.quince.featurizeddataset\", "
    "\"fields\" : [ ";

constexpr const char kFDSSchemaSuffix[] =
    " ] "
  "}";

FDSSchemaBuilder::FDSSchemaBuilder()
  : schema_(kFDSSchemaPrefix), num_of_features_(0) {}

FDSSchemaBuilder& FDSSchemaBuilder::AddDenseFeature(const string& name, DataType dtype,
                                       size_t rank, const avro::Type avro_type) {
  string type = GenerateArrayType(dtype, rank, avro_type);
  string feature_schema = BuildFeatureSchema(name, type);
  AddFeature(feature_schema);
  return *this;
}

FDSSchemaBuilder& FDSSchemaBuilder::AddSparseFeature(const string& name, DataType dtype,
                                                     size_t rank, const avro::Type avro_type) {
  std::vector<size_t> order(rank + 1, 0);
  for (size_t i = 0; i < order.size(); i++) {
    order[i] = i;
  }
  AddSparseFeature(name, dtype, order, avro_type);
  return *this;
}

FDSSchemaBuilder& FDSSchemaBuilder::AddSparseFeature(const string& name, DataType dtype,
                                                     const std::vector<size_t>& order, const avro::Type avro_type) {
  string indices_type = GenerateArrayType(DT_INT64, 1);
  string values_type = GenerateArrayType(dtype, 1, avro_type);
  string fields = "";

  auto values_index = order.size() - 1;
  for (size_t i = 0; i < order.size(); i++) {
    if (i > 0) {
      fields += ", ";
    }
    if (order[i] == values_index) {
      fields += BuildFeatureSchema("values", values_type);
    } else {
      auto indices_name = "indices" + std::to_string(order[i]);
      fields += BuildFeatureSchema(indices_name, indices_type);
    }
  }

  string type = "{"
    "\"type\" : \"record\", "
    "\"name\" : \"" + name + "\", "
    "\"fields\" : [ " + fields + " ] "
  "}";
  string feature_schema = BuildFeatureSchema(name, type);
  AddFeature(feature_schema);
  return *this;
}

FDSSchemaBuilder& FDSSchemaBuilder::AddOpaqueContextualFeature(const string& name,
                                                  const string& type) {
  string feature_schema = BuildFeatureSchema(name, type);
  AddFeature(feature_schema);
  return *this;
}

string FDSSchemaBuilder::Build() {
  return schema_ + kFDSSchemaSuffix;
}

avro::ValidSchema FDSSchemaBuilder::BuildVaildSchema() {
  string schema = Build();

  std::istringstream iss(schema);
  avro::ValidSchema valid_schema;
  avro::compileJsonSchema(iss, valid_schema);
  return valid_schema;
}

void FDSSchemaBuilder::AddFeature(const string& feature_schema) {
  if (num_of_features_ > 0) {
    schema_ += ", ";
  }
  schema_ += feature_schema;
  num_of_features_++;
}

string FDSSchemaBuilder::BuildFeatureSchema(const string& name,
                                            const string& type) {
  return "{"
    "\"name\" : \"" + name + "\", "
    "\"type\" : " + type +
  " }";
}

string FDSSchemaBuilder::BuildNullableFeatureSchema(const string& name,
                                                    const string& type) {
  return "{"
    "\"name\" : \"" + name + "\", "
    "\"type\" : [ \"null\", " + type + " ] "
  "}";
}

string FDSSchemaBuilder::GenerateDataType(DataType dtype, const avro::Type avro_type) {
  switch (dtype) {
    case DT_INT32: {
      return "\"int\"";
    }
    case DT_INT64: {
      return "\"long\"";
    }
    case DT_FLOAT: {
      return "\"float\"";
    }
    case DT_DOUBLE: {
      return "\"double\"";
    }
    case DT_STRING: {
      if (avro_type == avro::AVRO_BYTES) {
        return "\"bytes\"";
      }
      return "\"string\"";
    }
    case DT_BOOL: {
      return "\"boolean\"";
    }
    default: {
      return "";
    }
  }
}

string FDSSchemaBuilder::GenerateArrayType(DataType dtype, size_t rank, const avro::Type avro_type) {
  if (rank == 0) {
    return GenerateDataType(dtype, avro_type);
  }

  string type = GenerateArrayType(dtype, rank - 1, avro_type);
  return  "{"
    "\"type\" : \"array\", "
    "\"items\" : " + type +
  " }";
}

avro::OutputStreamPtr EncodeAvroGenericDatum(avro::GenericDatum& datum) {
  avro::EncoderPtr encoder = avro::binaryEncoder();
  avro::OutputStreamPtr out_stream = avro::memoryOutputStream();
  encoder->init(*out_stream);
  avro::encode(*encoder, datum);
  encoder->flush();
  return std::move(out_stream);
}

avro::OutputStreamPtr EncodeAvroGenericData(std::vector<avro::GenericDatum>& data) {
  avro::EncoderPtr encoder = avro::binaryEncoder();
  avro::OutputStreamPtr out_stream = avro::memoryOutputStream();
  encoder->init(*out_stream);
  for (auto& datum : data) {
    avro::encode(*encoder, datum);
  }
  encoder->flush();
  return std::move(out_stream);
}

}  // namespace fds
}  // namespace tensorflow
