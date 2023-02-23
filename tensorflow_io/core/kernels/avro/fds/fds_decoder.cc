#include "tensorflow_io/core/kernels/avro/fds/fds_decoder.h"

#include "tensorflow_io/core/kernels/avro/fds/dense_feature_decoder.h"
#include "tensorflow_io/core/kernels/avro/fds/sparse_feature_decoder.h"
#include "tensorflow_io/core/kernels/avro/fds/varlen_feature_decoder.h"
#include "tensorflow_io/core/kernels/avro/fds/opaque_contextual_feature_decoder.h"
#include "tensorflow_io/core/kernels/avro/fds/errors.h"

#include "api/Generic.hh"
#include "api/Specific.hh"

namespace tensorflow {
namespace fds {

Status FDSDecoder::Initialize(const avro::ValidSchema& schema) {
  auto& root_node = schema.root();
  if (root_node->type() != avro::AVRO_RECORD) {
    return FDSNotRecordError(avro::toString(root_node->type()), schema.toJson());
  }

  size_t num_of_columns = root_node->leaves();
  feature_names_.resize(num_of_columns, "");
  decoder_types_.resize(num_of_columns, FeatureType::opaque_contextual);
  decoders_.resize(num_of_columns);

  for (size_t i = 0; i < dense_features_.size(); i++) {
    TF_RETURN_IF_ERROR(InitializeFeatureDecoder(schema, root_node, dense_features_[i]));
  }

  for (size_t i = 0; i < sparse_features_.size(); i++) {
    TF_RETURN_IF_ERROR(InitializeFeatureDecoder(schema, root_node, sparse_features_[i]));
  }

  for (size_t i = 0; i < varlen_features_.size(); i++) {
    TF_RETURN_IF_ERROR(InitializeFeatureDecoder(schema, root_node, varlen_features_[i]));
  }

  size_t opaque_contextual_index = 0;
  for (size_t i = 0; i < num_of_columns; i++) {
    if (decoder_types_[i] == FeatureType::opaque_contextual) {
      decoders_[i] = std::unique_ptr<DecoderBase>(
        new opaque_contextual::FeatureDecoder(opaque_contextual_index++));

      auto& opaque_contextual_node = root_node->leafAt(i);
      skipped_data_.emplace_back(opaque_contextual_node);
      if (opaque_contextual_node->hasName()) {
        feature_names_[i] = root_node->leafAt(i)->name();
        LOG(WARNING) << "Column '" << feature_names_[i] << "' from input data"
          << " is not used. Cost of parsing an unused column is prohibitive!! "
          << "Consider dropping it to improve I/O performance.";
      }
    }
  }

  // Decoder requires unvaried schema in all input files.
  // Copy the schema to validate other input files.
  schema_ = schema;

  return Status::OK();
}

}  // namespace fds
}  // namespace tensorflow
