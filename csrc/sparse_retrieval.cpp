#include <torch/extension.h>

void sparse_accumulate_cuda(
    const torch::Tensor& query_indices,
    const torch::Tensor& query_values,
    const torch::Tensor& feature_offsets,
    const torch::Tensor& doc_indices,
    const torch::Tensor& posting_values,
    torch::Tensor& scores);

void sparse_accumulate(
    const torch::Tensor& query_indices,
    const torch::Tensor& query_values,
    const torch::Tensor& feature_offsets,
    const torch::Tensor& doc_indices,
    const torch::Tensor& posting_values,
    torch::Tensor scores) {
  TORCH_CHECK(query_indices.is_cuda(), "query indices must be CUDA tensors");
  const auto device = query_indices.device();
  TORCH_CHECK(query_values.device() == device && feature_offsets.device() == device &&
                  doc_indices.device() == device && posting_values.device() == device &&
                  scores.device() == device,
              "all sparse retrieval tensors must reside on the same CUDA device");
  TORCH_CHECK(query_indices.scalar_type() == torch::kInt32 &&
                  doc_indices.scalar_type() == torch::kInt32,
              "feature and local document indices must be int32");
  TORCH_CHECK(feature_offsets.scalar_type() == torch::kInt64,
              "posting-list offsets must be int64");
  TORCH_CHECK(query_values.scalar_type() == torch::kFloat32 &&
                  posting_values.scalar_type() == torch::kFloat32 &&
                  scores.scalar_type() == torch::kFloat32,
              "query values, posting values, and scores must be float32");
  TORCH_CHECK(query_indices.dim() == 2 && query_values.sizes() == query_indices.sizes(),
              "queries must have matching [batch, K] shapes");
  TORCH_CHECK(feature_offsets.dim() == 1 && feature_offsets.numel() >= 2 &&
                  doc_indices.dim() == 1 && posting_values.dim() == 1 &&
                  doc_indices.numel() == posting_values.numel(),
              "invalid posting-list tensor shapes");
  TORCH_CHECK(scores.dim() == 2 && scores.size(0) == query_indices.size(0) &&
                  scores.size(1) > 0 && scores.stride(1) == 1 &&
                  scores.stride(0) >= scores.size(1),
              "scores must have matching batch size and contiguous rows");
  TORCH_CHECK(query_indices.is_contiguous() && query_values.is_contiguous() &&
                  feature_offsets.is_contiguous() && doc_indices.is_contiguous() &&
                  posting_values.is_contiguous(),
              "compact query and posting tensors must be contiguous");
  sparse_accumulate_cuda(query_indices, query_values, feature_offsets, doc_indices,
                         posting_values, scores);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("accumulate", &sparse_accumulate,
             "Accumulate exact sparse query/posting products into CUDA float32 scores");
}
