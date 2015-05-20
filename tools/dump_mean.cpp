#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

extern "C" {
#include <stdio.h>
}

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, const char **argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc != 3) {
    LOG(ERROR) << "Usage: "
        << "dump_mean mean_proto_file_in mean_file_out";
    return 1;
  }

  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(argv[1], &blob_proto);

  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);

  int channels = mean_blob.channels();
  int height = mean_blob.height();
  int width = mean_blob.width();
  LOG(ERROR) << "mean shape: "
      << channels << " "
      << height << " "
      << width;
  float *mean = mean_blob.mutable_cpu_data();
  float mean_max = -INFINITY;
  float mean_min = INFINITY;
  for (int i = 0; i < channels * height * width; i++) {
    if (mean[i] > mean_max) {
      mean_max = mean[i];
    }
    if (mean[i] < mean_min) {
      mean_min = mean[i];
    }
  }
  LOG(ERROR) << "mean max/min: " << mean_max << " " << mean_min;

  FILE *fp = fopen(argv[2], "wb");
  size_t write_count = fwrite(mean, sizeof(float), channels * height * width, fp);
  if (write_count != channels * height * width) {
    LOG(ERROR) << "FATAL: failed to write to file!";
    return -1;
  }
  fclose(fp);

  LOG(ERROR) << "Wrote mean to file: " << argv[2];

  return 0;
}
