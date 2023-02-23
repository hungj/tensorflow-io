licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "avro_fds",
    hdrs = [
        "decoder_base.h",
        "sparse_value_buffer.h",
        "dense_feature_decoder.h",
        "sparse_feature_decoder.h",
        "sparse_feature_internal_decoder.h",
        "varlen_feature_decoder.h",
        "opaque_contextual_feature_decoder.h",
        "fds_decoder.h",
        "errors.h",
        "shuffle_handler.h",
        "decompression_handler.h",
        "avro_block_reader.h",
        "avro_decoder_template.h",
    ],
    srcs = [
        "errors.cc",
        "fds_decoder.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@avro",
    ],
    alwayslink = 1,
)

cc_test(
    name = "avro_fds_tests",
    srcs = [
        "decoder_test_util.h",
        "decoder_test_util.cc",
        "dense_feature_decoder_test.cc",
        "sparse_feature_decoder_test.cc",
        "varlen_feature_decoder_test.cc",
        "fds_decoder_test.cc",
        "shuffle_handler_test.cc",
        "sparse_value_buffer_test.cc",
        "avro_block_reader_test.cc",
    ],
    deps = [
        ":avro_fds",
        "@com_google_googletest//:gtest_main"
    ],
    copts = tf_io_copts(),
)
