// This is a script to upgrade "V0" network prototxts to the new format.
// Usage:
//    upgrade_net_proto_binary v0_net_proto_file_in net_proto_file_out

#include <google/protobuf/repeated_field.h>

#include <cmath>
#include <cstring>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <sstream>

extern "C" {
#include <stdio.h>
}

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

using std::ofstream;

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc != 3) {
    LOG(ERROR) << "Usage: "
        << "dump_weights net_proto_file_in weights_file_out";
    return 1;
  }

  NetParameter net_param;
  if (!ReadProtoFromBinaryFile(argv[1], &net_param)) {
    LOG(ERROR) << "Failed to parse input binary file as NetParameter: "
               << argv[1];
    return 2;
  }
  bool success = true;

  std::string prefix(argv[2]);

  LOG(ERROR) << "Net name: " << net_param.name();
  int num_layers = net_param.layer_size();
  LOG(ERROR) << "Num layers: " << num_layers;

  int num_conv_layers = 0;

  for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
    LayerParameter layer = net_param.layer(layer_idx);
    int num_blobs = layer.blobs_size();
    LOG(ERROR) << "Layer " << layer_idx << " blobs: " << num_blobs;
    if (num_blobs == 0) {
      continue;
    }
    num_conv_layers += 1;

    for (int blob_idx = 0; blob_idx < num_blobs; blob_idx++) {
      BlobProto blob = layer.blobs(blob_idx);
      Blob<float> real_blob;
      real_blob.FromProto(blob);

      /*int blob_shape_dims = blob.shape().dim_size();
      int blob_shape[blob_shape_dims];
      for (int dim = 0; dim < blob_shape_dims; dim++) {
        blob_shape[dim] = blob.shape().dim(dim);
      }
      LOG(ERROR) << "Blob dims: " << blob_shape_dims;
      if (blob_shape_dims == 2) {
        LOG(ERROR) << "Blob shape (2): " << blob_shape[0] << " " << blob_shape[1];
      } else if (blob_shape_dims == 4) {
        LOG(ERROR) << "Blob shape (4): " << blob_shape[0] << " " << blob_shape[1] << " " << blob_shape[2] << " " << blob_shape[3];
      }*/

      int num = blob.num();
      int channels = blob.channels();
      int height = blob.height();
      int width = blob.width();
      LOG(ERROR) << "Blob shape: " << num << " " << channels << " " << height << " " << width;

      std::ostringstream s;
      if (height > 1) {
        // Weights blob.
        s << prefix << "_layer_" << num_conv_layers << "_weights.mem";
      } else {
        // Bias blob.
        s << prefix << "_layer_" << num_conv_layers << "_bias.mem";
      }
      std::string name(s.str());
      LOG(ERROR) << "Output filename: " << name;

      FILE *fp = fopen(name.c_str(), "wb");
      int n = num * channels * height * width;
      for (int i = 0; i < n; i++) {
        //float x = blob.data(i);
        float x = real_blob.cpu_data()[i];
        size_t write_count = fwrite(&x, sizeof(float), 1, fp);
        if (write_count != 1) {
          LOG(ERROR) << "failed to write!";
          return -1;
        }
      }
      fclose(fp);

      /*FILE *fp = fopen(name.c_str(), "wb");
      int n = num * channels * height * width;
      for (int i = 0; i < n; i++) {
        //float x = blob.data(i);
        float x = real_blob.cpu_data()[i];
        size_t write_count = fwrite(&x, sizeof(float), 1, fp);
        if (write_count != 1) {
          LOG(ERROR) << "failed to write!";
          return -1;
        }
      }
      fclose(fp);*/
    }
  }

  //WriteProtoToBinaryFile(net_param, argv[2]);

  LOG(ERROR) << "Wrote weights to " << argv[2];
  return !success;
}
