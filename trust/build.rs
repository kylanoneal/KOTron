fn main() {
    // In this example, we're assuming your proto file is located in the "proto" directory.
    prost_build::compile_protos(&["proto/tron_pb.proto"], &["proto/"]).expect("Failed to compile proto files");
}
