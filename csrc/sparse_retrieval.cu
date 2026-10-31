#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/macros/Macros.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace {

// Each warp visits one partition of an active (query, feature) posting list.
// Small query batches can have fewer active lists than the GPU can execute
// concurrently. Splitting sufficiently long lists across warps increases
// available parallelism without allocating product/coordinate intermediates.
// There is no dense dictionary-width query/gallery multiplication. Different
// feature lists can contribute to the same (query, gallery-row) score, hence
// atomic float32 accumulation. The scores are zeroed by the caller beforehand.
__global__ void accumulate_postings_kernel(
    const int32_t* __restrict__ query_indices,
    const float* __restrict__ query_values,
    const int64_t* __restrict__ feature_offsets,
    const int32_t* __restrict__ doc_indices,
    const float* __restrict__ posting_values,
    float* __restrict__ scores,
    const int64_t query_width,
    const int64_t query_entries,
    const int64_t hidden_dim,
    const int64_t score_row_stride,
    const int64_t posting_partitions) {
  constexpr int warp_width = 32;
  const int lane = threadIdx.x % warp_width;
  const int warps_per_block = blockDim.x / warp_width;
  const int64_t first_task =
      static_cast<int64_t>(blockIdx.x) * warps_per_block + threadIdx.x / warp_width;
  const int64_t task_stride = static_cast<int64_t>(gridDim.x) * warps_per_block;
  const int64_t partitioned_tasks = query_entries * posting_partitions;
  for (int64_t task = first_task; task < partitioned_tasks; task += task_stride) {
    const int64_t query_entry = task / posting_partitions;
    const int64_t partition = task % posting_partitions;
    const float query_value = query_values[query_entry];
    if (query_value == 0.0f) {
      continue;
    }
    const int32_t feature = query_indices[query_entry];
    CUDA_KERNEL_ASSERT(feature >= 0 && feature < hidden_dim);
    const int64_t begin = feature_offsets[feature];
    const int64_t end = feature_offsets[feature + 1];
    const int64_t query_row = query_entry / query_width;
    float* query_scores = scores + query_row * score_row_stride;
    for (int64_t posting = begin + partition * warp_width + lane;
         posting < end; posting += warp_width * posting_partitions) {
      const int32_t gallery_row = doc_indices[posting];
      atomicAdd(query_scores + gallery_row, query_value * posting_values[posting]);
    }
  }
}

}  // namespace

void sparse_accumulate_cuda(
    const torch::Tensor& query_indices,
    const torch::Tensor& query_values,
    const torch::Tensor& feature_offsets,
    const torch::Tensor& doc_indices,
    const torch::Tensor& posting_values,
    torch::Tensor& scores) {
  const c10::cuda::CUDAGuard device_guard(query_indices.device());
  const int64_t query_entries = query_indices.numel();
  if (query_entries == 0 || posting_values.numel() == 0) {
    return;
  }
  constexpr int threads = 128;
  constexpr int warps_per_block = threads / 32;
  // Metadata and cached device properties only; this performs no tensor read
  // or host/device synchronization in the timed search path. Bound partitioning
  // by average posting length to avoid launching mostly empty warps on short
  // lists. The heuristic changes work assignment, never the products computed.
  const auto* properties = at::cuda::getDeviceProperties(query_indices.get_device());
  const int64_t target_warps = static_cast<int64_t>(properties->multiProcessorCount) * 8;
  const int64_t occupancy_partitions = (target_warps + query_entries - 1) / query_entries;
  const int64_t hidden_dim = feature_offsets.numel() - 1;
  const int64_t average_postings = (posting_values.numel() + hidden_dim - 1) / hidden_dim;
  const int64_t length_partitions = (average_postings + 31) / 32;
  const int64_t posting_partitions = std::max<int64_t>(
      1, std::min<int64_t>(8, std::min(occupancy_partitions, length_partitions)));
  const int64_t tasks = query_entries * posting_partitions;
  const int64_t requested_blocks = (tasks + warps_per_block - 1) / warps_per_block;
  const int blocks = static_cast<int>(std::min<int64_t>(requested_blocks, 65535));
  const auto stream = at::cuda::getCurrentCUDAStream(query_indices.get_device());
  accumulate_postings_kernel<<<blocks, threads, 0, stream>>>(
      query_indices.data_ptr<int32_t>(), query_values.data_ptr<float>(),
      feature_offsets.data_ptr<int64_t>(), doc_indices.data_ptr<int32_t>(),
      posting_values.data_ptr<float>(), scores.data_ptr<float>(), query_indices.size(1),
      query_entries, hidden_dim, scores.stride(0), posting_partitions);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
