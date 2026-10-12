// Copyright 2026 The ML Drift Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ml_drift/samples/llm/qwen3_model_builder.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/kernels/reduce.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/samples/llm/llm_config.h"
#include "ml_drift/samples/llm/llm_tensor_loader.h"
#include "ml_drift/samples/llm/llm_weights_loader.h"

namespace ml_drift {
namespace {

// QWEN 3 TRANSFORMER BUILDER

// This file demonstrates how to build a Qwen 3 Transformer model using ML Drift
// (GpuModelBuilder). For code readability and clarity, the implementation
// is organized into four logical sections:
//
//   Section 1: Build Context & Tensor Helpers
//              - Uses GpuModelBuilder::TensorHandle directly without
//              indirection
//              - Bundles shared model building state (GPU builder, loader,
//              config)
//              - Provides clean wrapper methods for common graph operations
//
//   Section 2: Custom GPU Kernels & Low-Level Hardware Operations
//              - Position IDs, Attention Masking, RoPE, KV Cache Operations
//              - Fused SiLU-Up activation
//
//   Section 3: Transformer Layers & Neural Network Blocks
//              - RMSNorm, Linear, Feed-Forward (MLP), Multi-Head Attention
//              - TransformerLayer (combines attention, MLP, and residual links)
//
//   Section 4: Top-Level Model Assembly
//              - EmbedTokens, LmHead, and
//              Qwen3ModelBuilder::BuildModelInternal

using TensorHandle = GpuModelBuilder::TensorHandle;

// SECTION 1: BUILD CONTEXT & TENSOR HELPERS

// Context object bundling shared dependencies for model building,
// reducing parameter boilerplate across layer functions.
struct BuildContext {
  GpuModelBuilder& builder;
  CreateGpuModelInfo& create_info;
  LlmTensorLoader* loader;
  const LlmConfig& config;
  int sequence_size;
  TensorHandle params_i32;

  DataType data_type() const {
    return DeduceDataTypeFromPrecision(create_info.precision);
  }

  TensorHandle AddTensor(const TensorDescriptor& desc) {
    return builder.AddTensor(desc);
  }

  TensorHandle AddTensor(const BHWC& shape, DataType dt) {
    return builder.AddTensor(shape, dt);
  }

  TensorHandle AddTensor(int b, int h, int w, int c, DataType dt) {
    return builder.AddTensor(b, h, w, c, dt);
  }

  TensorHandle AddConstantTensor(
      const ml_drift::Tensor<ml_drift::Linear, DataType::kFloat32>& t,
      DataType dt) {
    return builder.AddConstantTensor(t, dt);
  }

  TensorHandle AddExternalTensor(GpuSpatialTensor* tensor) {
    TensorHandle th = builder.AddTensor(tensor->GetDescriptor());
    create_info.external_immutable_tensors[th.id] = tensor;
    return th;
  }

  TensorHandle FullyConnected(
      const TensorHandle& src, const GpuModelBuilder::Weights& weights,
      const TensorHandle* biases = nullptr,
      const TensorHandle* src_exp = nullptr,
      const ConvRuntimeCheckDesc& runtime_check = {},
      const TensorHandle* runtime_check_tensor = nullptr) {
    const TensorHandle* check_tensor = runtime_check_tensor;
    if (check_tensor == nullptr && (runtime_check.src_end_ch_index >= 0 ||
                                    runtime_check.dst_end_ch_index >= 0)) {
      check_tensor = &params_i32;
    }
    auto status_or = builder.FullyConnectedExternalWeights(
        src, weights, biases, src_exp, runtime_check, check_tensor);
    ABSL_CHECK_OK(status_or);
    return status_or.value();
  }

  TensorHandle FullyConnectedInt4(const TensorHandle& src,
                                  const GpuModelBuilder::Weights& weights,
                                  const TensorHandle* biases = nullptr) {
    return builder.FullyConnectedInt4ExternalWeights(src, weights, biases);
  }

  TensorHandle FullyConnectedInt8(const TensorHandle& src,
                                  const GpuModelBuilder::Weights& weights,
                                  const TensorHandle* biases = nullptr) {
    return builder.FullyConnectedInt8ExternalWeights(src, weights, biases);
  }

  TensorHandle SoftmaxReduce(const TensorHandle& src,
                             const SoftmaxRuntimeCheckDesc& runtime_params,
                             const TensorHandle* params_tensor = nullptr) {
    return builder.SoftmaxReduce(src, runtime_params,
                                 params_tensor ? params_tensor : &params_i32);
  }

  TensorHandle SplitRoPEConcat(const TensorHandle& src,
                               const TensorHandle& position_ids,
                               const RoPEAttributes& rope_attr) {
    return builder.SplitRoPEConcat(src, position_ids, rope_attr);
  }

  TensorHandle EmbeddingLookup(const TensorHandle& input_ids,
                               const GpuModelBuilder::Weights& weights,
                               DataType dt, Axis axis) {
    return builder.EmbeddingLookup(input_ids, weights, dt, axis);
  }

  TensorHandle RMSNormalization(
      const TensorHandle& src, float epsilon,
      ml_drift::Tensor<ml_drift::Linear, DataType::kFloat32>* gamma) {
    return builder.RMSNormalization(src, epsilon, gamma);
  }

  TensorHandle Reduce(const TensorHandle& src, Reduce::Type type,
                      const std::set<Axis>& axes) {
    return builder.Reduce(src, type, axes);
  }

  TensorHandle Reshape(const TensorHandle& src, const BHWC& new_shape) {
    return builder.Reshape(src, new_shape);
  }

  TensorHandle Transpose(const TensorHandle& src, const BHWC& perm) {
    return builder.Transpose(src, perm);
  }

  TensorHandle Slice(const TensorHandle& src, const BHWC& offset,
                     const BHWC& slice_shape) {
    return builder.SubTensor(src, offset, slice_shape);
  }

  TensorHandle Add(const TensorHandle& a, const TensorHandle& b) {
    return builder.Add(a, b);
  }

  TensorHandle Mul(const TensorHandle& a, const TensorHandle& b) {
    return builder.Multiplication(a, b);
  }

  TensorHandle Mul(const TensorHandle& a, float scalar) {
    return builder.Multiplication(a, scalar);
  }

  void AddGpuOperation(const std::vector<TensorHandle>& inputs,
                       const std::vector<TensorHandle>& outputs,
                       std::unique_ptr<GPUOperation> operation,
                       const std::string& name = "") {
    builder.AddGpuOperation(inputs, outputs, std::move(operation), name);
  }
};

// SECTION 2: CUSTOM GPU KERNELS & LOW-LEVEL HARDWARE OPERATIONS

// Generates token position IDs based on current runtime token offsets.
TensorHandle PositionIds(BuildContext& ctx) {
  TensorHandle position_ids =
      ctx.AddTensor(1, 1, ctx.sequence_size, 1, DataType::kInt32);
  GPUOperation op;
  BufferDescriptor buffer_desc;
  buffer_desc.element_type = DataType::kInt32;
  buffer_desc.element_size = 1;
  buffer_desc.memory_type = MemoryType::kGlobal;
  op.AddSrcBuffer("src", buffer_desc);
  op.AddDstTensor("dst", position_ids.tensor_desc);
  op.args_.AddInt("param_index", LlmRuntimeParams::kTokenOffsetIndex);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  op.code_ = R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst.Width() || Y >= args.dst.Height() || S >= args.dst.Slices()) {
    return;
  }
  args.dst::type res_value;
  int offset = args.src.Read(args.param_index);
  res_value.x = ucl::Convert<args.dst::scalar_type>(offset + X);
  res_value.y = ucl::Convert<args.dst::scalar_type>(offset + X);
  res_value.z = ucl::Convert<args.dst::scalar_type>(offset + X);
  res_value.w = ucl::Convert<args.dst::scalar_type>(offset + X);
  args.dst.Write(res_value, X, Y, S);
}
)";
  ctx.AddGpuOperation({ctx.params_i32}, {position_ids},
                      std::make_unique<GPUOperation>(std::move(op)),
                      "fill_position_ids");
  return position_ids;
}

// Applies attention masking by setting masked positions to -30000.0f.
TensorHandle SelectMask(BuildContext& ctx, const TensorHandle& src,
                        const TensorHandle& mask) {
  TensorHandle dst = ctx.AddTensor(src.tensor_desc);
  ElementwiseDescriptor op_desc;
  // exp(-30000) underflows cleanly to exactly 0.0f.
  if (src.tensor_desc.GetDataType() == DataType::kFloat32) {
    op_desc.args.AddFloat("min_float", -30000.0f);
  } else {
    op_desc.args.AddHalf("min_float", static_cast<half>(-30000.0f));
  }
  op_desc.code = R"(
      out_value.x = in2_value.x ? args.min_float : in_value.x;
      out_value.y = in2_value.y ? args.min_float : in_value.y;
      out_value.z = in2_value.z ? args.min_float : in_value.z;
      out_value.w = in2_value.w ? args.min_float : in_value.w;
  )";
  OperationDef definition;
  definition.src_tensors.push_back(src.tensor_desc);
  definition.src_tensors.push_back(mask.tensor_desc);
  definition.dst_tensors.push_back(dst.tensor_desc);
  auto select_mask_op = std::make_unique<GPUOperation>(CreateGpuOperation(
      definition, std::move(op_desc), mask.tensor_desc.GetBHWCShape(),
      dst.tensor_desc.GetBHWCShape()));
  ctx.AddGpuOperation({src, mask}, {dst}, std::move(select_mask_op),
                      "select_mask");
  return dst;
}

// GPU kernel generating causal attention masks.
GPUOperation AttentionMaskOp(const TensorDescriptor& params_desc,
                             const TensorDescriptor& out_desc, int mask_width) {
  GPUOperation op;
  op.args_.AddInt("mask_width", mask_width);
  op.args_.AddInt("window_size", 0);
  op.args_.AddInt("sliding_local_cache", 0);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  op.code_ = absl::Substitute(R"(
  MAIN_FUNCTION($$0) {
  int linear_xb = ucl::GetGlobalId<0>();
  int X = linear_xb / args.dst.Batch();
  int B = linear_xb % args.dst.Batch();
  args.dst.SetBatchRef(B);
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst.Width() || Y >= args.dst.Height() || S >= args.dst.Slices()) {
    return;
  }
  int token_offset;
  args.params_tensor.ReadPerChannel(token_offset, 0, 0, $0);

  args.dst::type mask_value;
  int k;
  bool visible;

  // x
  k = S * 4 + 0;
  int token_index = k - token_offset;
  visible = (token_index <= X);
  mask_value.x = !visible;

  // y
  k = S * 4 + 1;
  token_index = k - token_offset;
  visible = (token_index <= X);
  mask_value.y = !visible;

  // z
  k = S * 4 + 2;
  token_index = k - token_offset;
  visible = (token_index <= X);
  mask_value.z = !visible;

  // w
  k = S * 4 + 3;
  token_index = k - token_offset;
  visible = (token_index <= X);
  mask_value.w = !visible;

  args.dst.Write(mask_value, X, Y, S);
})",
                              LlmRuntimeParams::kTokenOffsetIndex);
  op.AddSrcTensor("params_tensor", params_desc);
  op.AddDstTensor("dst", out_desc);
  return op;
}

// Builds global causal attention mask.
TensorHandle AttentionMask(BuildContext& ctx) {
  const int cache_size = ctx.config.cache_size;
  const BHWC mask_shape = BHWC(1, 1, ctx.sequence_size, cache_size);
  TensorHandle attention_mask = ctx.AddTensor(mask_shape, DataType::kBool);
  ctx.AddGpuOperation(
      {ctx.params_i32}, {attention_mask},
      std::make_unique<GPUOperation>(AttentionMaskOp(
          ctx.params_i32.tensor_desc, attention_mask.tensor_desc, cache_size)),
      "apply_attention_mask");
  return attention_mask;
}

// --- KV Cache Tensor Layout and Shape Helpers ---

WeightsDescription GetKWeightsDescription(DataType data_type, int cache_size,
                                          int kv_heads_count,
                                          int head_dimension) {
  WeightsDescription weights_desc;
  weights_desc.type = data_type;
  weights_desc.layout = WeightsLayout::kOSpatialIOGroupO4I4;
  weights_desc.output_group_size = (cache_size + 3) / 4;
  return weights_desc;
}

OHWI GetKWeightsShape(int cache_size, int kv_heads_count, int head_dimension) {
  return OHWI(cache_size, kv_heads_count, 1, head_dimension);
}

WeightsDescription GetVWeightsDescription(DataType data_type, int cache_size,
                                          int kv_heads_count,
                                          int head_dimension) {
  WeightsDescription weights_desc;
  weights_desc.type = data_type;
  weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  weights_desc.output_group_size = (head_dimension + 3) / 4;
  return weights_desc;
}

OHWI GetVWeightsShape(int cache_size, int kv_heads_count, int head_dimension) {
  return OHWI(head_dimension, kv_heads_count, 1, cache_size);
}

TensorDescriptor GetKCacheTensorDescriptor(DataType data_type, int cache_size,
                                           int kv_heads_count,
                                           int head_dimension) {
  WeightsDescription weights_desc = GetKWeightsDescription(
      data_type, cache_size, kv_heads_count, head_dimension);
  TensorDescriptor weights_td = TensorDescriptor(
      weights_desc.type, TensorStorageType::kBuffer, Layout::kLinear);
  weights_td.SetBHWCShape(
      BHWC(1, 1, 1,
           GetTotalElementsCountForLayout(
               weights_desc,
               GetKWeightsShape(cache_size, kv_heads_count, head_dimension))));
  return weights_td;
}

TensorDescriptor GetVCacheTensorDescriptor(DataType data_type, int cache_size,
                                           int kv_heads_count,
                                           int head_dimension) {
  WeightsDescription weights_desc = GetVWeightsDescription(
      data_type, cache_size, kv_heads_count, head_dimension);
  TensorDescriptor weights_td = TensorDescriptor(
      weights_desc.type, TensorStorageType::kBuffer, Layout::kLinear);
  weights_td.SetBHWCShape(
      BHWC(1, 1, 1,
           GetTotalElementsCountForLayout(
               weights_desc,
               GetVWeightsShape(cache_size, kv_heads_count, head_dimension))));
  return weights_td;
}

// Updates shared K and V cache tensors with newly computed key/value tokens.
// k:       {1, batch_size * number_of_kv_heads, sequence_size, head_dimension}
// v:       {1, batch_size * number_of_kv_heads, sequence_size, head_dimension}
// cache_k: {1, 1, 1, linear_elements_o4i4}
// cache_v: {1, 1, 1, linear_elements_i4o4}
void UpdateKVCache(BuildContext& ctx, const TensorHandle& k,
                   const TensorHandle& v, const TensorHandle& cache_k,
                   const TensorHandle& cache_v) {
  std::string code = absl::Substitute(R"(
MAIN_FUNCTION($$0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.src_k.Width() || Y >= args.batch_size || S >= args.src_k.Slices()) {
    return;
  }
  int token_index_offset = args.params.Read($0);
  int active_tokens = args.params.Read($1);
  int token_index = token_index_offset + X;
  if (token_index >= args.cache_size || token_index >= active_tokens) {
    return;
  }

  args.src_k::type value_k = args.src_k.Read(X, Y, S);
  args.src_v::type value_v = args.src_v.Read(X, Y, S);
  {
    // cache_k - kOSpatialIOGroupO4I4 layout, O - cache_size, I - head_size
    int k_o = token_index;
    int k_o_slice = k_o / 4;
    int k_o4_index = k_o % 4;
    int k_i = S * 4;
    int k_i_slice = k_i / 4;
    int k_sp = Y;
    int i_slices = (args.head_size + 3) / 4;
    int o_slices = (args.cache_size + 3) / 4;
    int k_index = ((k_sp * i_slices + k_i_slice) * o_slices + k_o_slice) * 4 + k_o4_index;
    args.cache_k.WriteLinear(value_k, k_index);
  }
  {
    // cache_v - kOSpatialIOGroupI4O4 layout, O - head_size, I - cache_size
    int v_o = S * 4;
    int v_o_slice = v_o / 4;
    int v_i = token_index;
    int v_i_slice = v_i / 4;
    int v_i4_index = v_i % 4;
    int v_sp = Y;
    int i_slices = (args.cache_size + 3) / 4;
    int o_slices = (args.head_size + 3) / 4;
    int v_index = ((v_sp * i_slices + v_i_slice) * o_slices + v_o_slice) * 4 + v_i4_index;
    args.cache_v.WriteLinear(value_v, v_index);
  }
})",
                                      LlmRuntimeParams::kTokenOffsetIndex,
                                      LlmRuntimeParams::kActiveTokensIndex);

  class AddValuesToCacheOp : public GPUOperation {
   public:
    AddValuesToCacheOp() = default;
    int3 GetGridSize() const override {
      return int3(src_[0]->Width(), batch_size_, src_[0]->Slices());
    }
    int batch_size_ = 1;
  };

  const int kv_heads_count = k.tensor_desc.GetBHWCShape().h;
  AddValuesToCacheOp custom_op;
  custom_op.batch_size_ = kv_heads_count;
  custom_op.args_.AddInt("cache_size", ctx.config.cache_size);
  custom_op.args_.AddInt("batch_size", kv_heads_count);
  custom_op.args_.AddInt("head_size", ctx.config.head_dimension);
  custom_op.AddSrcTensor("src_k", k.tensor_desc);
  custom_op.AddSrcTensor("src_v", v.tensor_desc);
  BufferDescriptor params_buffer;
  params_buffer.element_type = DataType::kInt32;
  params_buffer.element_size = 1;
  custom_op.AddSrcBuffer("params", params_buffer);
  custom_op.AddDstTensor("cache_k", cache_k.tensor_desc);
  custom_op.AddDstTensor("cache_v", cache_v.tensor_desc);
  custom_op.code_ = code;

  ctx.AddGpuOperation(
      {k, v, ctx.params_i32}, {cache_k, cache_v},
      std::make_unique<AddValuesToCacheOp>(std::move(custom_op)),
      "cache_update");
}

// Standard Q/K/V layout transformation: reshapes and transposes heads/tokens.
void TransformQKV(BuildContext& ctx, int sequence_size, TensorHandle* q,
                  TensorHandle* k, TensorHandle* v) {
  const int batch_size = ctx.config.batch_size;
  const int head_size = ctx.config.head_dimension;
  const int heads_count = ctx.config.number_of_heads;
  const int kv_heads_count = ctx.config.number_of_kv_heads;
  const int kv_group_size = heads_count / kv_heads_count;
  if (q != nullptr) {
    *q = ctx.Reshape(*q,
                     BHWC{batch_size, sequence_size, heads_count, head_size});
    *q = ctx.Transpose(*q, BHWC(0, 2, 1, 3));
    *q = ctx.Reshape(*q, BHWC{1, batch_size * kv_heads_count,
                              sequence_size * kv_group_size, head_size});
  }
  if (k != nullptr) {
    *k = ctx.Reshape(
        *k, BHWC{batch_size, sequence_size, kv_heads_count, head_size});
    *k = ctx.Transpose(*k, BHWC(0, 2, 1, 3));
    *k = ctx.Reshape(
        *k, BHWC{1, batch_size * kv_heads_count, sequence_size, head_size});
  }
  if (v != nullptr) {
    *v = ctx.Reshape(
        *v, BHWC{batch_size, sequence_size, kv_heads_count, head_size});
    *v = ctx.Transpose(*v, BHWC(0, 2, 1, 3));
    *v = ctx.Reshape(
        *v, BHWC{1, batch_size * kv_heads_count, sequence_size, head_size});
  }
}

// Fused GPU Kernel: Reshapes Q/K/V, applies per-head RMSNorm, applies
// Rotary Position Embedding (RoPE), and writes K/V directly into the KV cache
// in a single hardware-optimized kernel.
void TransformQKVRmsNormRoPE(BuildContext& ctx, const TensorHandle& position,
                             TensorHandle* q, const TensorHandle& k,
                             const TensorHandle& v, const TensorHandle& cache_k,
                             const TensorHandle& cache_v, float min_timescale,
                             float max_timescale, const std::string& name) {
  const bool k_eq_v = (k.id == v.id);
  const int heads_count = ctx.config.number_of_heads;
  const int head_size = ctx.config.head_dimension;
  const int batch_size = ctx.config.batch_size;
  const int kv_heads_count = ctx.config.number_of_kv_heads;
  const int kv_head_group_size = heads_count / kv_heads_count;
  DataType data_type = ctx.data_type();

  TensorHandle q_dst =
      ctx.AddTensor(BHWC{1, batch_size * kv_heads_count,
                         kv_head_group_size * ctx.sequence_size, head_size},
                    q->tensor_desc.GetDataType());

  // Fuse Q scaling into Q RMSNorm.
  ml_drift::Tensor<ml_drift::Linear, DataType::kFloat32> q_gamma;
  q_gamma.shape = ml_drift::Linear(head_size);
  q_gamma.data = ctx.loader->LoadFloat32(
      absl::StrCat(name, ".q_norm", kWeightSuffix), head_size);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  for (int i = 0; i < q_gamma.data.size(); ++i) {
    q_gamma.data[i] =
        (q_gamma.data[i] + (ctx.config.qwen35 ? 1.0f : 0.0f)) * scale;
  }
  TensorHandle q_gamma_tensor = ctx.AddConstantTensor(q_gamma, data_type);

  ml_drift::Tensor<ml_drift::Linear, DataType::kFloat32> k_gamma;
  k_gamma.shape = ml_drift::Linear(head_size);
  k_gamma.data = ctx.loader->LoadFloat32(
      absl::StrCat(name, ".k_norm", kWeightSuffix), head_size);
  if (ctx.config.qwen35) {
    for (float& gamma : k_gamma.data) {
      gamma += 1.0f;
    }
  }
  TensorHandle k_gamma_tensor = ctx.AddConstantTensor(k_gamma, data_type);

  std::string pow_func_name = "pow";
  std::string sin_func_name = "sin";
  std::string cos_func_name = "cos";
  if (ctx.builder.gpu_info().IsAdreno() &&
      ctx.builder.gpu_info().IsApiOpenCl()) {
    pow_func_name = "native_powr";
    sin_func_name = "native_sin";
    cos_func_name = "native_cos";
  }

  std::string code = R"(MAIN_FUNCTION($0) {
  int DST_X = ucl::GetGlobalId<0>();
  int DST_Y = ucl::GetGlobalId<1>();
  int DST_S_GLOBAL = ucl::GetGlobalId<2>();

  int LOCAL_S = ucl::GetLocalId<2>();

  int half_slices_count = args.q_dst.Slices() / 2;
  bool is_active = (DST_X < args.q_dst.Width() && DST_Y < args.q_dst.Height() && DST_S_GLOBAL < half_slices_count);

    int rotary_half_slices = args.rotary_dim / 8;
    int DST_S0 = DST_S_GLOBAL < rotary_half_slices
      ? DST_S_GLOBAL : DST_S_GLOBAL + rotary_half_slices;
    int DST_S1 = DST_S_GLOBAL < rotary_half_slices
      ? DST_S_GLOBAL + rotary_half_slices : DST_S_GLOBAL + half_slices_count;

  int batch_id = DST_Y / args.kv_heads_count;
  int kv_head_id = DST_Y % args.kv_heads_count;
  int kv_head_group_id = DST_X / args.sequence_size;
  int seq_id = DST_X % args.sequence_size;

  args.q_src.SetBatchRef(batch_id);
  args.k_src.SetBatchRef(batch_id);
)";

  if (!k_eq_v) {
    code += R"(  args.v_src.SetBatchRef(batch_id);
)";
  }

  code += absl::Substitute(R"(
  __local float shared_q_sum[$0];
  __local float shared_k_sum[$0];
  __local float shared_inv_std_q;
  __local float shared_inv_std_k;
)",
                           head_size / 8);

  code += R"(
  // 1. Cooperative Load & Sum of Squares for Q
  float4 q_val0 = ucl::Init<float4>(0.0f);
  float4 q_val1 = ucl::Init<float4>(0.0f);
  float local_sum_q = 0.0f;
  if (is_active) {
    int q_head_id = kv_head_id * args.kv_head_group_size + kv_head_group_id;
    int SRC_S0 = q_head_id * args.q_dst.Slices() + DST_S0;
    int SRC_S1 = q_head_id * args.q_dst.Slices() + DST_S1;
    q_val0 = ucl::Convert<float4>(args.q_src.Read(seq_id, 0, SRC_S0));
    q_val1 = ucl::Convert<float4>(args.q_src.Read(seq_id, 0, SRC_S1));
    local_sum_q = dot(q_val0, q_val0) + dot(q_val1, q_val1);
  }
  shared_q_sum[LOCAL_S] = local_sum_q;

  // Cooperative Load & Sum of Squares for K
  float4 raw_k_val0 = ucl::Init<float4>(0.0f);
  float4 raw_k_val1 = ucl::Init<float4>(0.0f);
  float4 k_val0 = ucl::Init<float4>(0.0f);
  float4 k_val1 = ucl::Init<float4>(0.0f);
  float local_sum_k = 0.0f;
  if (is_active && kv_head_group_id == 0) {
    int K_SRC_S0 = kv_head_id * args.q_dst.Slices() + DST_S0;
    int K_SRC_S1 = kv_head_id * args.q_dst.Slices() + DST_S1;
    raw_k_val0 = ucl::Convert<float4>(args.k_src.Read(seq_id, 0, K_SRC_S0));
    raw_k_val1 = ucl::Convert<float4>(args.k_src.Read(seq_id, 0, K_SRC_S1));
    k_val0 = raw_k_val0;
    k_val1 = raw_k_val1;
    local_sum_k = dot(k_val0, k_val0) + dot(k_val1, k_val1);
  }
  shared_k_sum[LOCAL_S] = local_sum_k;

  ucl::SyncThreads<WorkGroup, Local>();
)";

  code += absl::Substitute(R"(
  int reduction_size = $0;
  while (reduction_size > 1) {
    int active_thread_limit = reduction_size / 2;
    int offset = (reduction_size + 1) / 2;
    if (LOCAL_S < active_thread_limit) {
      shared_q_sum[LOCAL_S] += shared_q_sum[LOCAL_S + offset];
      shared_k_sum[LOCAL_S] += shared_k_sum[LOCAL_S + offset];
    }
    ucl::SyncThreads<WorkGroup, Local>();
    reduction_size = offset;
  }
)",
                           head_size / 8);

  code += R"(
  if (LOCAL_S == 0) {
    float inv_ch = 1.0f / ucl::Convert<float>(args.q_dst.Channels());
    shared_inv_std_q = rsqrt(shared_q_sum[0] * inv_ch + 1e-6f);
    shared_inv_std_k = rsqrt(shared_k_sum[0] * inv_ch + 1e-6f);
  }
  ucl::SyncThreads<WorkGroup, Local>();

  if (is_active) {
    float stddev_inv_q = shared_inv_std_q;
    float4 val0 = q_val0 * stddev_inv_q;
    float4 val1 = q_val1 * stddev_inv_q;

    float4 gamma0 = args.q_gamma.Read<float>(DST_S0);
    float4 gamma1 = args.q_gamma.Read<float>(DST_S1);
    val0 = val0 * gamma0;
    val1 = val1 * gamma1;

    // 2. RoPE Constants
    float4 fraction;
    float inv_q_dst_ch = 1.0f / ucl::Convert<float>(args.rotary_dim);
    fraction.x = 2.0f * ucl::Convert<float>(DST_S_GLOBAL * 4 + 0) * inv_q_dst_ch;
    fraction.y = 2.0f * ucl::Convert<float>(DST_S_GLOBAL * 4 + 1) * inv_q_dst_ch;
    fraction.z = 2.0f * ucl::Convert<float>(DST_S_GLOBAL * 4 + 2) * inv_q_dst_ch;
    fraction.w = 2.0f * ucl::Convert<float>(DST_S_GLOBAL * 4 + 3) * inv_q_dst_ch;
    float4 min_timescale = ucl::Init<float4>(args.min_timescale);
    float4 max_timescale = ucl::Init<float4>(args.max_timescale);
)";
  code += "    float4 timescale = min_timescale * " + pow_func_name +
          "(max_timescale / min_timescale, fraction);\n";
  code +=
      "    float4 pos_val = "
      "ucl::Init<float4>(args.position.Read<float>(seq_id, 0, 0).x);\n";
  code += "    float4 sinusoid_inp = pos_val / timescale;\n";
  code += "    Type sin_val = ucl::Convert<Type>(" + sin_func_name +
          "(sinusoid_inp));\n";
  code += "    Type cos_val = ucl::Convert<Type>(" + cos_func_name +
          "(sinusoid_inp));\n";

  code += R"(
    if (DST_S_GLOBAL >= args.rotary_dim / 8) {
      sin_val = ucl::Init<Type>(0.0f);
      cos_val = ucl::Init<Type>(1.0f);
    }
    Type val0_t = ucl::Convert<Type>(val0);
    Type val1_t = ucl::Convert<Type>(val1);
    Type q_out0 = val0_t * cos_val - val1_t * sin_val;
    Type q_out1 = val1_t * cos_val + val0_t * sin_val;
    args.q_dst.Write(q_out0, DST_X, DST_Y, DST_S0);
    args.q_dst.Write(q_out1, DST_X, DST_Y, DST_S1);

    if (kv_head_group_id == 0) {
      float stddev_inv_k = shared_inv_std_k;
      float4 k_v0 = k_val0 * stddev_inv_k;
      float4 k_v1 = k_val1 * stddev_inv_k;
      float4 k_g0 = args.k_gamma.Read<float>(DST_S0);
      float4 k_g1 = args.k_gamma.Read<float>(DST_S1);
      k_v0 = k_v0 * k_g0;
      k_v1 = k_v1 * k_g1;
      Type k_val0_t = ucl::Convert<Type>(k_v0);
      Type k_val1_t = ucl::Convert<Type>(k_v1);
      Type out0 = k_val0_t * cos_val - k_val1_t * sin_val;
      Type out1 = k_val1_t * cos_val + k_val0_t * sin_val;
)";

  if (k_eq_v) {
    code += R"(
      Type v_out0 = ucl::Convert<Type>(raw_k_val0);
      Type v_out1 = ucl::Convert<Type>(raw_k_val1);
)";
  } else {
    code += R"(
      int V_SRC_S0 = kv_head_id * args.q_dst.Slices() + DST_S0;
      int V_SRC_S1 = kv_head_id * args.q_dst.Slices() + DST_S1;
      Type v_out0 = args.v_src.Read(seq_id, 0, V_SRC_S0);
      Type v_out1 = args.v_src.Read(seq_id, 0, V_SRC_S1);
)";
  }

  code += absl::Substitute(R"(
      int token_index_offset = args.params.Read($0);
      int active_tokens = args.params.Read($1);
      int token_index = token_index_offset + seq_id;
      if (token_index < args.cache_size && token_index < active_tokens) {
        int head_slices = args.q_dst.Slices();
        int token_chunks = (args.cache_size + 3) / 4;
        int t_chunk = token_index / 4;
        int t_rem = token_index % 4;

        int k_idx0 = ((DST_Y * head_slices + DST_S0) * token_chunks + t_chunk) * 4 + t_rem;
        int k_idx1 = ((DST_Y * head_slices + DST_S1) * token_chunks + t_chunk) * 4 + t_rem;
        args.cache_k.WriteLinear(ucl::Convert<args.cache_k::type>(out0), k_idx0);
        args.cache_k.WriteLinear(ucl::Convert<args.cache_k::type>(out1), k_idx1);

        int v_idx0 = ((DST_Y * token_chunks + t_chunk) * head_slices + DST_S0) * 4 + t_rem;
        int v_idx1 = ((DST_Y * token_chunks + t_chunk) * head_slices + DST_S1) * 4 + t_rem;
        args.cache_v.WriteLinear(ucl::Convert<args.cache_v::type>(v_out0), v_idx0);
        args.cache_v.WriteLinear(ucl::Convert<args.cache_v::type>(v_out1), v_idx1);
      }
    }
  }
})",
                           LlmRuntimeParams::kTokenOffsetIndex,
                           LlmRuntimeParams::kActiveTokensIndex);

  code = absl::StrReplaceAll(
      code, {{"Type", ToUclDataType(q_dst.tensor_desc.GetDataType(), 4)}});

  class TransformQKVRmsNormApplyRoPEOp : public GPUOperation {
   public:
    int3 GetGridSize() const override {
      const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
      const int grid_y = dst_[0]->Height() * dst_[0]->Depth();
      const int grid_z = dst_[0]->Slices() / 2;
      return int3(grid_x, grid_y, grid_z);
    }
    std::vector<int3> GetPossibleKernelWorkGroups(
        TuningType tuning_type, const GpuInfo& gpu_info,
        const KernelInfo& kernel_info) const override {
      return {work_group_size_};
    }
  };

  TransformQKVRmsNormApplyRoPEOp custom_op;
  custom_op.args_.AddInt("kv_heads_count", kv_heads_count);
  custom_op.args_.AddInt("kv_head_group_size", kv_head_group_size);
  custom_op.args_.AddInt("sequence_size", ctx.sequence_size);
  custom_op.args_.AddInt("cache_size", ctx.config.cache_size);
  custom_op.args_.AddInt(
      "rotary_dim", ctx.config.qwen35
                        ? static_cast<int>(head_size *
                                           ctx.config.qwen35->partial_rotary_factor)
                        : head_size);
  custom_op.args_.AddFloat("min_timescale", min_timescale);
  custom_op.args_.AddFloat("max_timescale", max_timescale);
  custom_op.work_group_size_ = int3(1, 1, head_size / 8);

  custom_op.AddSrcTensor("q_src", q->tensor_desc);
  custom_op.AddSrcTensor("k_src", k.tensor_desc);
  if (!k_eq_v) {
    custom_op.AddSrcTensor("v_src", v.tensor_desc);
  }
  custom_op.AddSrcTensor("position", position.tensor_desc);
  custom_op.AddSrcTensor("q_gamma", q_gamma_tensor.tensor_desc);
  custom_op.AddSrcTensor("k_gamma", k_gamma_tensor.tensor_desc);

  BufferDescriptor params_buffer;
  params_buffer.element_type = DataType::kInt32;
  params_buffer.element_size = 1;
  custom_op.AddSrcBuffer("params", params_buffer);

  custom_op.AddDstTensor("q_dst", q_dst.tensor_desc);
  custom_op.AddDstTensor("cache_k", cache_k.tensor_desc);
  custom_op.AddDstTensor("cache_v", cache_v.tensor_desc);

  custom_op.code_ = code;
  custom_op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  std::vector<TensorHandle> inputs = {*q, k};
  if (!k_eq_v) {
    inputs.push_back(v);
  }
  inputs.push_back(position);
  inputs.push_back(q_gamma_tensor);
  inputs.push_back(k_gamma_tensor);
  inputs.push_back(ctx.params_i32);

  std::vector<TensorHandle> outputs = {q_dst, cache_k, cache_v};

  ctx.AddGpuOperation(
      inputs, outputs,
      std::make_unique<TransformQKVRmsNormApplyRoPEOp>(std::move(custom_op)),
      "QKVRmsNormRoPE");

  *q = q_dst;
}

// Fused GPU Kernel: Computes SiLU activation and multiplies by the Up
// projection in the Feed-Forward (MLP) block.
GPUOperation CreateSiluMulOp(const TensorDescriptor& gate_desc,
                             const TensorDescriptor& up_desc,
                             const TensorDescriptor& dst_desc) {
  ElementwiseDescriptor op_desc;
  op_desc.code = absl::Substitute(R"(
      float4 gate_f32 = ucl::Convert<float4>(in_value);
      float4 up_f32 = ucl::Convert<float4>(in2_value);
      float4 sig_val = 1.0f / (1.0f + exp(-gate_f32));
      float4 silu_val = gate_f32 * sig_val;
      float4 res_f32 = clamp(silu_val * up_f32, ucl::Init<float4>(-65504.0f), ucl::Init<float4>(65504.0f));
      out_value = ucl::Convert<$0>(res_f32);
  )", ToUclDataType(dst_desc.GetDataType(), 4));
  OperationDef definition;
  definition.src_tensors.push_back(gate_desc);
  definition.src_tensors.push_back(up_desc);
  definition.dst_tensors.push_back(dst_desc);
  return CreateGpuOperation(definition, std::move(op_desc),
                            dst_desc.GetBHWCShape(), dst_desc.GetBHWCShape());
}

TensorHandle SiluMul(BuildContext& ctx, const TensorHandle& gate,
                     const TensorHandle& up) {
  TensorHandle dst = ctx.AddTensor(gate.tensor_desc);
  auto silu_mul_op = std::make_unique<GPUOperation>(
      CreateSiluMulOp(gate.tensor_desc, up.tensor_desc, dst.tensor_desc));
  ctx.AddGpuOperation({gate, up}, {dst}, std::move(silu_mul_op), "silu_mul");
  return dst;
}

// SECTION 3: TRANSFORMER LAYERS & NEURAL NETWORK BLOCKS

// Root Mean Square Normalization (RMSNorm).
// Normalizes input activations across channels and scales by learned gamma
// weights. Qwen 3 uses scale_shift = 0.0f by default.
TensorHandle RMSNorm(BuildContext& ctx, const TensorHandle& src,
                     const std::string& name, float scale_shift = 0.0f,
                     float multiplier = 1.0f) {
  ml_drift::Tensor<ml_drift::Linear, DataType::kFloat32> gamma;
  gamma.shape = ml_drift::Linear(src.tensor_desc.GetBHWCShape().c);
  gamma.data = ctx.loader->LoadFloat32(absl::StrCat(name, kWeightSuffix),
                                       src.tensor_desc.GetBHWCShape().c);
  for (int i = 0; i < gamma.data.size(); ++i) {
    gamma.data[i] = (scale_shift + gamma.data[i]) * multiplier;
  }
  return ctx.RMSNormalization(src, /*epsilon=*/1e-6f, &gamma);
}

// Linear (Fully Connected) projection layer with automatic precision detection.
// Automatically detects whether weights are INT4, INT8, or floating-point,
// loads the weights accordingly, and invokes the appropriate fully connected
// operation.
TensorHandle Linear(BuildContext& ctx, const TensorHandle& src,
                    const std::string& name, int out_channels,
                    const TensorHandle* biases = nullptr) {
  const int in_channels = src.tensor_desc.GetBHWCShape().c;
  const OHWI weights_shape = OHWI(out_channels, 1, 1, in_channels);
  auto [weights, weight_type] = GetWeights(ctx, name, weights_shape);

  if (weight_type == DataType::kInt4) {
    return ctx.FullyConnectedInt4(src, weights, biases);
  } else if (weight_type == DataType::kInt8) {
    return ctx.FullyConnectedInt8(src, weights, biases);
  } else {
    return ctx.FullyConnected(src, weights, biases);
  }
}

// Feed-Forward (MLP) Block for Qwen 3:
//   1. Parallel Gate and Up projections (.mlp.gate_proj, .mlp.up_proj)
//   2. Fused SiLU activation with gating elementwise multiplication (SiluMul)
//   3. Down projection (.mlp.down_proj)
TensorHandle FFLayer(BuildContext& ctx, const TensorHandle& src,
                     const std::string& name) {
  TensorHandle gate = Linear(ctx, src, absl::StrCat(name, ".mlp.gate_proj"),
                             ctx.config.hidden_dimension);
  TensorHandle up = Linear(ctx, src, absl::StrCat(name, ".mlp.up_proj"),
                           ctx.config.hidden_dimension);

  TensorHandle ff_hidden = SiluMul(ctx, gate, up);

  return Linear(ctx, ff_hidden, absl::StrCat(name, ".mlp.down_proj"),
                ctx.config.model_dimension);
}

TensorHandle CausalConv(BuildContext& ctx, const TensorHandle& src,
                        const TensorHandle& history, const std::string& name) {
  const int kernel_size = ctx.config.qwen35->linear_conv_kernel_dim;
  const int channels = src.tensor_desc.GetBHWCShape().c;
  const auto data = ctx.loader->LoadFloat32(
      absl::StrCat(name, ".conv1d", kWeightSuffix), channels * kernel_size);
  ABSL_CHECK_EQ(data.size(), channels * kernel_size);
  ml_drift::Tensor<ml_drift::Linear, DataType::kFloat32> weights;
  weights.shape = ml_drift::Linear(channels * kernel_size);
  weights.data.resize(channels * kernel_size);
  for (int channel = 0; channel < channels; ++channel) {
    for (int tap = 0; tap < kernel_size; ++tap) {
      weights.data[tap * channels + channel] = data[channel * kernel_size + tap];
    }
  }
  TensorHandle weights_handle = ctx.AddConstantTensor(weights, DataType::kFloat32);
  TensorHandle dst = ctx.AddTensor(src.tensor_desc);
  class CausalConvOp : public GPUOperation {
   public:
    int3 GetGridSize() const override {
      return int3(dst_[0]->Slices(), dst_[0]->Batch(), 1);
    }
  };
  CausalConvOp op;
  op.AddSrcTensor("src", src.tensor_desc);
  op.AddSrcTensor("history", history.tensor_desc);
  op.AddSrcTensor("weights", weights_handle.tensor_desc);
  BufferDescriptor params_desc;
  params_desc.element_type = DataType::kInt32;
  params_desc.element_size = 1;
  op.AddSrcBuffer("params", params_desc);
  op.AddDstTensor("dst", dst.tensor_desc);
  op.AddDstTensor("history_dst", history.tensor_desc);
  op.args_.AddInt("kernel_size", kernel_size);
  op.code_ = absl::Substitute(R"(
MAIN_FUNCTION($$0) {
  int slice = ucl::GetGlobalId<0>();
  int batch = ucl::GetGlobalId<1>();
  if (slice >= args.dst.Slices() || batch >= args.dst.Batch()) {
    return;
  }
  args.src.SetBatchRef(batch);
  args.history.SetBatchRef(batch);
  args.dst.SetBatchRef(batch);
  args.history_dst.SetBatchRef(batch);
  int offset = args.params.Read($1);
  int active_tokens = args.params.Read($2);
  float4 history_values[$0];
  for (int tap = 0; tap < args.kernel_size - 1; ++tap) {
    history_values[tap] = ucl::Init<float4>(0.0f);
    if (offset > 0) {
      history_values[tap] = args.history.Read<float>(tap, 0, slice);
    }
  }
  for (int token = 0; token < args.src.Width(); ++token) {
    if (offset + token >= active_tokens) {
      break;
    }
    float4 value = args.src.Read<float>(token, 0, slice);
    float4 result = value * args.weights.Read<float>(
        (args.kernel_size - 1) * args.dst.Slices() + slice);
    for (int tap = 0; tap < args.kernel_size - 1; ++tap) {
      result += history_values[tap] * args.weights.Read<float>(
          tap * args.dst.Slices() + slice);
    }
    for (int tap = 0; tap < args.kernel_size - 2; ++tap) {
      history_values[tap] = history_values[tap + 1];
    }
    history_values[args.kernel_size - 2] = value;
    result = result / (1.0f + exp(-result));
    args.dst.Write(ucl::Convert<args.dst::type>(result), token, 0, slice);
  }
  for (int tap = 0; tap < args.kernel_size - 1; ++tap) {
    args.history_dst.Write(
        ucl::Convert<args.history_dst::type>(history_values[tap]), tap, 0, slice);
  }
}
)", kernel_size - 1, LlmRuntimeParams::kTokenOffsetIndex,
       LlmRuntimeParams::kActiveTokensIndex);
  ctx.AddGpuOperation({src, history, weights_handle, ctx.params_i32},
                      {dst, history},
                      std::make_unique<CausalConvOp>(std::move(op)),
                      "qwen35_causal_conv");
  return dst;
}

TensorHandle GatedDeltaRule(BuildContext& ctx, const TensorHandle& qkv,
                            const TensorHandle& gate_a,
                            const TensorHandle& gate_b,
                            const TensorHandle& state,
                            const std::string& name) {
  const auto& config = *ctx.config.qwen35;
  const int key_dim = config.linear_key_head_dim;
  const int value_dim = config.linear_value_head_dim;
  const int value_heads = config.linear_num_value_heads;
  const int key_heads = config.linear_num_key_heads;
  ml_drift::Tensor<ml_drift::Linear, DataType::kFloat32> a_log, dt_bias;
  a_log.shape = ml_drift::Linear(value_heads);
  a_log.data = ctx.loader->LoadFloat32(absl::StrCat(name, ".A_log.bin"),
                                      value_heads);
  dt_bias.shape = ml_drift::Linear(value_heads);
  dt_bias.data = ctx.loader->LoadFloat32(absl::StrCat(name, ".dt_bias.bin"),
                                        value_heads);
  TensorHandle a_log_handle = ctx.AddConstantTensor(a_log, DataType::kFloat32);
  TensorHandle dt_bias_handle = ctx.AddConstantTensor(dt_bias, DataType::kFloat32);
  TensorHandle dst = ctx.AddTensor(ctx.config.batch_size, value_heads,
                                   ctx.sequence_size, value_dim, ctx.data_type());
  class GatedDeltaRuleOp : public GPUOperation {
   public:
    int3 GetGridSize() const override {
      return int3(dst_[0]->Slices(), dst_[0]->Height() * dst_[0]->Batch(), 1);
    }
    std::vector<int3> GetPossibleKernelWorkGroups(
        TuningType tuning_type, const GpuInfo& gpu_info,
        const KernelInfo& kernel_info) const override {
      return {work_group_size_};
    }
  };
  GatedDeltaRuleOp op;
  op.work_group_size_ = int3(value_dim / 4, 1, 1);
  op.AddSrcTensor("qkv", qkv.tensor_desc);
  op.AddSrcTensor("gate_a", gate_a.tensor_desc);
  op.AddSrcTensor("gate_b", gate_b.tensor_desc);
  op.AddSrcTensor("a_log", a_log_handle.tensor_desc);
  op.AddSrcTensor("dt_bias", dt_bias_handle.tensor_desc);
  op.AddSrcTensor("state", state.tensor_desc);
  BufferDescriptor params_desc;
  params_desc.element_type = DataType::kInt32;
  params_desc.element_size = 1;
  op.AddSrcBuffer("params", params_desc);
  op.AddDstTensor("dst", dst.tensor_desc);
  op.AddDstTensor("state_dst", state.tensor_desc);
  op.args_.AddInt("key_dim", key_dim);
  op.args_.AddInt("key_heads", key_heads);
  op.args_.AddInt("value_heads", value_heads);
  op.args_.AddFloat("query_scale", 1.0f / std::sqrt(static_cast<float>(key_dim)));
  op.code_ = absl::Substitute(R"(
MAIN_FUNCTION($$0) {
  int slice = ucl::GetLocalId<0>();
  int head_index = ucl::GetGlobalId<1>();
  int head = head_index % args.value_heads;
  int batch = head_index / args.value_heads;
  int key_head = head / (args.value_heads / args.key_heads);
  args.qkv.SetBatchRef(batch);
  args.gate_a.SetBatchRef(batch);
  args.gate_b.SetBatchRef(batch);
  args.dst.SetBatchRef(batch);
  int offset = args.params.Read($3);
  int active_tokens = args.params.Read($4);
  float4 state_values[$0];
  for (int row = 0; row < args.key_dim; ++row) {
    state_values[row] = ucl::Init<float4>(0.0f);
    if (offset > 0) {
      state_values[row] = args.state.Read<float>(row, head_index, slice);
    }
  }
  __local float4 queries[$1];
  __local float4 keys[$1];
  __local float query_sums[$2];
  __local float key_sums[$2];
  float a_log;
  float dt_bias;
  args.a_log.ReadPerChannel(a_log, head);
  args.dt_bias.ReadPerChannel(dt_bias, head);
  for (int token = 0; token < args.dst.Width(); ++token) {
    if (offset + token >= active_tokens) {
      break;
    }
    float query_sum = 0.0f;
    float key_sum = 0.0f;
    for (int key_slice = slice; key_slice < $1; key_slice += $2) {
      int query_slice = key_head * $1 + key_slice;
      float4 query = args.qkv.Read<float>(token, 0, query_slice);
      float4 key = args.qkv.Read<float>(token, 0, args.key_heads * $1 + query_slice);
      queries[key_slice] = query;
      keys[key_slice] = key;
      query_sum += dot(query, query);
      key_sum += dot(key, key);
    }
    query_sums[slice] = query_sum;
    key_sums[slice] = key_sum;
    ucl::SyncThreads<WorkGroup, Local>();
    int reduction_size = $2;
    while (reduction_size > 1) {
      int limit = reduction_size / 2;
      int reduction_offset = (reduction_size + 1) / 2;
      if (slice < limit) {
        query_sums[slice] += query_sums[slice + reduction_offset];
        key_sums[slice] += key_sums[slice + reduction_offset];
      }
      ucl::SyncThreads<WorkGroup, Local>();
      reduction_size = reduction_offset;
    }
    float query_scale = rsqrt(query_sums[0] + 1e-6f) * args.query_scale;
    float key_scale = rsqrt(key_sums[0] + 1e-6f);
    float gate_a;
    float gate_b;
    args.gate_a.ReadPerChannel<float>(gate_a, token, 0, head);
    args.gate_b.ReadPerChannel<float>(gate_b, token, 0, head);
    float step = gate_a + dt_bias;
    float softplus = max(step, 0.0f) + log(1.0f + exp(-fabs(step)));
    float decay = exp(-exp(a_log) * softplus);
    float beta = 1.0f / (1.0f + exp(-gate_b));
    float4 predicted = ucl::Init<float4>(0.0f);
    for (int row = 0; row < args.key_dim; ++row) {
      state_values[row] *= decay;
      float4 key_vector = keys[row / 4];
      int component = row % 4;
      float key = (component == 0 ? key_vector.x :
           component == 1 ? key_vector.y :
           component == 2 ? key_vector.z : key_vector.w) * key_scale;
      predicted += state_values[row] * key;
    }
    float4 value = args.qkv.Read<float>(
        token, 0, 2 * args.key_heads * $1 + head * $2 + slice);
    float4 delta = (value - predicted) * beta;
    float4 output_value = ucl::Init<float4>(0.0f);
    for (int row = 0; row < args.key_dim; ++row) {
      float4 key_vector = keys[row / 4];
      float4 query_vector = queries[row / 4];
      int component = row % 4;
      float key = (component == 0 ? key_vector.x :
           component == 1 ? key_vector.y :
           component == 2 ? key_vector.z : key_vector.w) * key_scale;
      float query = (component == 0 ? query_vector.x :
             component == 1 ? query_vector.y :
             component == 2 ? query_vector.z : query_vector.w) * query_scale;
      state_values[row] += key * delta;
      output_value += state_values[row] * query;
    }
    args.dst.Write(ucl::Convert<args.dst::type>(output_value), token, head, slice);
    ucl::SyncThreads<WorkGroup, Local>();
  }
  for (int row = 0; row < args.key_dim; ++row) {
    args.state_dst.Write(state_values[row], row, head_index, slice);
  }
}
)", key_dim, key_dim / 4, value_dim / 4,
       LlmRuntimeParams::kTokenOffsetIndex, LlmRuntimeParams::kActiveTokensIndex);
  ctx.AddGpuOperation(
      {qkv, gate_a, gate_b, a_log_handle, dt_bias_handle, state, ctx.params_i32},
      {dst, state}, std::make_unique<GatedDeltaRuleOp>(std::move(op)),
      "qwen35_gated_delta_rule");
  return dst;
}

TensorHandle LinearAttention(BuildContext& ctx, const TensorHandle& src,
                             const std::string& name,
                             const TensorHandle& conv_state,
                             const TensorHandle& recurrent_state) {
  const auto& config = *ctx.config.qwen35;
  const int key_channels = config.linear_num_key_heads * config.linear_key_head_dim;
  const int value_channels =
      config.linear_num_value_heads * config.linear_value_head_dim;
  TensorHandle qkv = Linear(ctx, src, absl::StrCat(name, ".in_proj_qkv"),
                             2 * key_channels + value_channels);
  TensorHandle gate_a = Linear(ctx, src, absl::StrCat(name, ".in_proj_a"),
                                config.linear_num_value_heads);
  TensorHandle gate_b = Linear(ctx, src, absl::StrCat(name, ".in_proj_b"),
                                config.linear_num_value_heads);
  TensorHandle gate_z = Linear(ctx, src, absl::StrCat(name, ".in_proj_z"),
                                value_channels);
  qkv = CausalConv(ctx, qkv, conv_state, name);
  TensorHandle attn = GatedDeltaRule(ctx, qkv, gate_a, gate_b, recurrent_state, name);
  ml_drift::Tensor<ml_drift::Linear, DataType::kFloat32> gamma;
  gamma.shape = ml_drift::Linear(config.linear_value_head_dim);
  gamma.data = ctx.loader->LoadFloat32(absl::StrCat(name, ".norm", kWeightSuffix),
                                       config.linear_value_head_dim);
  attn = ctx.RMSNormalization(attn, 1e-6f, &gamma);
  gate_z = ctx.Reshape(
      gate_z, BHWC(ctx.config.batch_size, ctx.sequence_size,
                   config.linear_num_value_heads, config.linear_value_head_dim));
  gate_z = ctx.Transpose(gate_z, BHWC(0, 2, 1, 3));
  attn = SiluMul(ctx, gate_z, attn);
  attn = ctx.Transpose(attn, BHWC(0, 2, 1, 3));
  attn = ctx.Reshape(attn, BHWC(ctx.config.batch_size, 1, ctx.sequence_size,
                               value_channels));
  return Linear(ctx, attn, absl::StrCat(name, ".out_proj"),
                ctx.config.model_dimension);
}

// Multi-Head Attention (MHA / GQA) Block:
//   1. Project input to Q, K, and V (.q_proj, .k_proj, .v_proj)
//   2. Apply Rotary Position Embedding (RoPE) and per-head RMSNorms
//   3. Update KV Caches with new K and V tokens
//   4. Compute scaled dot-product attention scores with causal masking
//   5. Apply softmax and weighted sum over V
//   6. Output linear projection (.o_proj)
TensorHandle SelfAttention(BuildContext& ctx, const TensorHandle& src,
                           const TensorHandle& attention_mask,
                           const TensorHandle& position_ids,
                           const std::string& name, int layer_i,
                           const TensorHandle& k_cache_in,
                           const TensorHandle& v_cache_in) {
  const int q_channels = ctx.config.head_dimension * ctx.config.number_of_heads;
  const int kv_channels =
      ctx.config.head_dimension * ctx.config.number_of_kv_heads;

  TensorHandle q = Linear(ctx, src, absl::StrCat(name, ".q_proj"),
                          q_channels * (ctx.config.qwen35 ? 2 : 1));
  TensorHandle gate;
  if (ctx.config.qwen35) {
    const int head_size = ctx.config.head_dimension;
    const BHWC split_shape(ctx.config.batch_size, ctx.sequence_size,
                           ctx.config.number_of_heads, head_size);
    q = ctx.Reshape(q, BHWC(ctx.config.batch_size, ctx.sequence_size,
                           ctx.config.number_of_heads, head_size * 2));
    gate = ctx.Slice(q, BHWC(0, 0, 0, head_size), split_shape);
    q = ctx.Slice(q, BHWC(0, 0, 0, 0), split_shape);
    const BHWC projection_shape(ctx.config.batch_size, 1, ctx.sequence_size,
                                q_channels);
    q = ctx.Reshape(q, projection_shape);
    gate = ctx.Reshape(gate, projection_shape);
  }
  TensorHandle k = Linear(ctx, src, absl::StrCat(name, ".k_proj"), kv_channels);
  TensorHandle v = Linear(ctx, src, absl::StrCat(name, ".v_proj"), kv_channels);

  float min_timescale = 1.0f;
  float max_timescale =
      ctx.config.qwen35 ? ctx.config.qwen35->rope_theta : 1000000.0f;

  if (ctx.config.head_dimension % 8 == 0) {
    // Also updates the KV cache.
    TransformQKVRmsNormRoPE(ctx, position_ids, &q, k, v, k_cache_in, v_cache_in,
                            min_timescale, max_timescale, name);
  } else {
    TransformQKV(ctx, ctx.sequence_size, &q, &k, &v);

    const float scale =
        1.0f / std::sqrt(static_cast<float>(ctx.config.head_dimension));
    q = RMSNorm(ctx, q, absl::StrCat(name, ".q_norm"),
                /*scale_shift=*/0.0f, /*multiplier=*/scale);
    k = RMSNorm(ctx, k, absl::StrCat(name, ".k_norm"));

    RoPEAttributes rope_attr;
    rope_attr.max_timescale = max_timescale;
    q = ctx.SplitRoPEConcat(q, position_ids, rope_attr);
    k = ctx.SplitRoPEConcat(k, position_ids, rope_attr);

    UpdateKVCache(ctx, k, v, k_cache_in, v_cache_in);
  }

  const int cache_size = ctx.config.cache_size;

  // Attention dot-product with partially fused softmax
  const int kv_heads_count =
      ctx.config.batch_size * ctx.config.number_of_kv_heads;
  DataType cache_dt = ctx.data_type();
  WeightsDescription k_weights_desc = GetKWeightsDescription(
      cache_dt, cache_size, kv_heads_count, ctx.config.head_dimension);
  OHWI k_weights_shape =
      GetKWeightsShape(cache_size, kv_heads_count, ctx.config.head_dimension);
  ConvRuntimeCheckDesc qk_runtime_check;
  qk_runtime_check.dst_end_ch_index =
      LlmRuntimeParams::kActiveTokensAlignedIndex;

  const GpuModelBuilder::Weights k_external_weights =
      CreateExternalWeights(k_cache_in, k_weights_desc, k_weights_shape);
  TensorHandle attn =
      ctx.FullyConnected(q, k_external_weights, /*biases=*/nullptr,
                         /*src_exp=*/nullptr, qk_runtime_check);
  attn = SelectMask(ctx, attn, attention_mask);

  SoftmaxRuntimeCheckDesc softmax_runtime_params;
  softmax_runtime_params.end_ch_index =
      LlmRuntimeParams::kActiveTokensAlignedIndex;
  TensorHandle att_exp = ctx.SoftmaxReduce(attn, softmax_runtime_params);

  WeightsDescription v_weights_desc = GetVWeightsDescription(
      cache_dt, cache_size, kv_heads_count, ctx.config.head_dimension);
  OHWI v_weights_shape =
      GetVWeightsShape(cache_size, kv_heads_count, ctx.config.head_dimension);
  ConvRuntimeCheckDesc v_runtime_check;
  v_runtime_check.src_end_ch_index =
      LlmRuntimeParams::kActiveTokensAlignedIndex;

  const GpuModelBuilder::Weights v_external_weights =
      CreateExternalWeights(v_cache_in, v_weights_desc, v_weights_shape);
  attn = ctx.FullyConnected(attn, v_external_weights, /*biases=*/nullptr,
                            &att_exp, v_runtime_check);

  // Reshape attention output for final linear projection
  const int batch_size = ctx.config.batch_size;
  const int head_size = ctx.config.head_dimension;
  const int heads_count = ctx.config.number_of_heads;
  attn = ctx.Reshape(
      attn, BHWC{batch_size, heads_count, ctx.sequence_size, head_size});
  attn = ctx.Transpose(attn, BHWC(0, 2, 1, 3));
  attn = ctx.Reshape(
      attn, BHWC{batch_size, 1, ctx.sequence_size, heads_count * head_size});

  if (ctx.config.qwen35) {
    TensorHandle gated = ctx.AddTensor(attn.tensor_desc);
    ElementwiseDescriptor gate_desc;
    gate_desc.code = absl::Substitute(R"(
      float4 gate_f32 = ucl::Convert<float4>(in2_value);
      float4 value_f32 = ucl::Convert<float4>(in_value);
      out_value = ucl::Convert<$0>(
          value_f32 / (1.0f + exp(-gate_f32)));
    )", ToUclDataType(gated.tensor_desc.GetDataType(), 4));
    OperationDef definition;
    definition.src_tensors = {attn.tensor_desc, gate.tensor_desc};
    definition.dst_tensors = {gated.tensor_desc};
    ctx.AddGpuOperation(
        {attn, gate}, {gated},
        std::make_unique<GPUOperation>(CreateGpuOperation(
            definition, std::move(gate_desc), gate.tensor_desc.GetBHWCShape(),
            gated.tensor_desc.GetBHWCShape())),
        "attention_output_gate");
    attn = gated;
  }

  return Linear(ctx, attn, absl::StrCat(name, ".o_proj"),
                ctx.config.model_dimension);
}

// A single Transformer layer in Qwen 3:
//   1. Attention block with RMSNorm before attention, with residual link.
//   2. MLP (FeedForward) block with RMSNorm before MLP, with residual link.
TensorHandle TransformerLayer(BuildContext& ctx, const TensorHandle& src,
                              const TensorHandle& attention_mask,
                              const TensorHandle& position_ids,
                              const std::string& name, int layer_i,
                              const TensorHandle& k_cache_in,
                              const TensorHandle& v_cache_in) {
  TensorHandle residual = src;

  // 1. Attention Block
  const float scale_shift = ctx.config.qwen35 ? 1.0f : 0.0f;
  TensorHandle x = RMSNorm(ctx, src, absl::StrCat(name, ".input_layernorm"),
                            scale_shift);
  if (ctx.config.qwen35 &&
      (layer_i + 1) % ctx.config.qwen35->full_attention_interval != 0) {
    x = LinearAttention(ctx, x, absl::StrCat(name, ".linear_attn"), k_cache_in,
                        v_cache_in);
  } else {
    x = SelfAttention(ctx, x, attention_mask, position_ids,
                      absl::StrCat(name, ".self_attn"), layer_i, k_cache_in,
                      v_cache_in);
  }
  x = ctx.Add(residual, x);

  // 2. Feed-Forward / MLP Block
  residual = x;
  x = RMSNorm(ctx, x, absl::StrCat(name, ".post_attention_layernorm"),
               scale_shift);
  TensorHandle mlp_output = FFLayer(ctx, x, name);
  return ctx.Add(residual, mlp_output);
}

// SECTION 4: TOP-LEVEL MODEL ASSEMBLY

struct EmbeddingState {
  GpuModelBuilder::Weights weights;
  DataType weight_type = DataType::kUnknown;
};

// Loads token embeddings (without sqrt(d_model) scaling).
TensorHandle EmbedTokens(BuildContext& ctx, const TensorHandle& input_ids,
                         EmbeddingState* state) {
  const std::string embed_name = "model.embed_tokens";
  const OHWI weights_shape =
      OHWI(ctx.config.vocabulary_size, 1, 1, ctx.config.model_dimension);

  auto [weights, weight_type] = GetWeights(ctx, embed_name, weights_shape);
  state->weights = weights;
  state->weight_type = weight_type;

  DataType emb_type = ctx.data_type();
  return ctx.EmbeddingLookup(input_ids, state->weights, emb_type, Axis::kWidth);
}

// Projects final hidden states to vocabulary logits using LM head weights.
// Reuses token embedding weights if separate lm_head weights are not present.
absl::StatusOr<TensorHandle> LmHead(BuildContext& ctx,
                                    const TensorHandle& hidden_state,
                                    const EmbeddingState& emb_state) {
  if (ctx.loader->HasTensor(absl::StrCat("lm_head", kWeightSuffix))) {
    return Linear(ctx, hidden_state, "lm_head", ctx.config.vocabulary_size);
  } else {
    // Reuse loaded embedding weights for LM head.
    if (emb_state.weight_type == DataType::kInt4) {
      return ctx.FullyConnectedInt4(hidden_state, emb_state.weights,
                                    /*biases=*/nullptr);
    } else if (emb_state.weight_type == DataType::kInt8) {
      return ctx.FullyConnectedInt8(hidden_state, emb_state.weights,
                                    /*biases=*/nullptr);
    } else {
      return ctx.FullyConnected(hidden_state, emb_state.weights,
                                /*biases=*/nullptr);
    }
  }
}

}  // namespace

Qwen3ModelBuilder::Qwen3ModelBuilder(
    const LlmConfig& config, const GpuInfo& gpu_info,
    CreateGpuModelInfo& create_info,
    std::unique_ptr<LlmTensorLoader> tensor_loader)
    : config_(config),
      gpu_info_(gpu_info),
      create_info_(create_info),
      tensor_loader_(std::move(tensor_loader)),
      model_builder_(gpu_info, create_info.hints, create_info.precision,
                     create_info.storage_type) {
  if (tensor_loader_) {
    tensor_loader_->SetGpuInfo(gpu_info);
    tensor_loader_->SetDataType(
        DeduceDataTypeFromPrecision(create_info.precision));
  }
}

void Qwen3ModelBuilder::CreateKVCache(
    std::vector<GpuModelBuilder::TensorHandle>* k_caches_in_handles,
    std::vector<GpuModelBuilder::TensorHandle>* v_caches_in_handles) {
  DataType cache_dt = DeduceDataTypeFromPrecision(create_info_.precision);
  const int kv_heads_count = config_.batch_size * config_.number_of_kv_heads;
  auto k_desc = GetKCacheTensorDescriptor(
      cache_dt, config_.cache_size, kv_heads_count, config_.head_dimension);
  auto v_desc = GetVCacheTensorDescriptor(
      cache_dt, config_.cache_size, kv_heads_count, config_.head_dimension);
  k_caches_in_handles->resize(config_.stack_size);
  v_caches_in_handles->resize(config_.stack_size);

  for (int i = 0; i < config_.stack_size; ++i) {
    if (config_.qwen35 &&
        (i + 1) % config_.qwen35->full_attention_interval != 0) {
      const auto& config = *config_.qwen35;
      const int channels = 2 * config.linear_num_key_heads * config.linear_key_head_dim +
                           config.linear_num_value_heads * config.linear_value_head_dim;
      (*k_caches_in_handles)[i] = model_builder_.AddTensor(
          BHWC(config_.batch_size, 1, config.linear_conv_kernel_dim - 1, channels),
          cache_dt);
      (*v_caches_in_handles)[i] = model_builder_.AddTensor(
          BHWC(1, config_.batch_size * config.linear_num_value_heads,
               config.linear_key_head_dim, config.linear_value_head_dim),
          DataType::kFloat32);
    } else {
      (*k_caches_in_handles)[i] = model_builder_.AddTensor(k_desc);
      (*v_caches_in_handles)[i] = model_builder_.AddTensor(v_desc);
    }
  }
}

absl::Status Qwen3ModelBuilder::BuildModelInternal(
    int sequence_size, GpuModel* gpu_model,
    const GpuModelBuilder::TensorHandle& params_i32_handle,
    const std::vector<GpuModelBuilder::TensorHandle>& k_caches_in_handles,
    const std::vector<GpuModelBuilder::TensorHandle>& v_caches_in_handles,
    GpuModelBuilder::TensorHandle* input_handle,
    GpuModelBuilder::TensorHandle* output_handle) {
  BuildContext ctx{model_builder_, create_info_,  tensor_loader_.get(),
                   config_,        sequence_size, params_i32_handle};

  TensorHandle input_ids =
      ctx.AddTensor(config_.batch_size, 1, sequence_size, 1, DataType::kInt32);
  *input_handle = input_ids;

  // 1. Token Embeddings
  EmbeddingState emb_state;
  TensorHandle x = EmbedTokens(ctx, input_ids, &emb_state);

  // 2. Positional and Attention Masks
  TensorHandle position_ids = PositionIds(ctx);
  TensorHandle attention_mask = AttentionMask(ctx);

  // 3. Transformer Layers Stack
  for (int i = 0; i < config_.stack_size; ++i) {
    x = TransformerLayer(ctx, x, attention_mask, position_ids,
                         absl::StrCat("model.layers.", i), i,
                         k_caches_in_handles[i], v_caches_in_handles[i]);
  }

  // 4. Final Normalization
  x = RMSNorm(ctx, x, "model.norm", config_.qwen35 ? 1.0f : 0.0f);

  // Slice out the last token hidden state before the LM head projection during
  // prefill to avoid projecting all prompt tokens to vocabulary logits.
  if (sequence_size > 1) {
    x = ctx.Slice(x, BHWC(0, 0, sequence_size - 1, 0),
                  BHWC(config_.batch_size, 1, 1, config_.model_dimension));
  }

  // 5. LM Head Output Projection
  auto status_or_logits = LmHead(ctx, x, emb_state);
  ABSL_RETURN_IF_ERROR(status_or_logits.status());
  *output_handle = status_or_logits.value();

  // 6. Compile GPU Model
  std::vector<ValueId> input_ids_vec = {input_handle->id, params_i32_handle.id};
  std::vector<ValueId> output_ids_vec = {output_handle->id};
  for (int i = 0; i < config_.stack_size; ++i) {
    input_ids_vec.push_back(k_caches_in_handles[i].id);
    output_ids_vec.push_back(k_caches_in_handles[i].id);
    input_ids_vec.push_back(v_caches_in_handles[i].id);
    output_ids_vec.push_back(v_caches_in_handles[i].id);
  }

  ABSL_RETURN_IF_ERROR(
      model_builder_.GetGpuModel(input_ids_vec, output_ids_vec, gpu_model));

  // Ensure descriptors for shared external inputs (runtime params and
  // KV caches) are present in the produced GpuModel's tensors map.
  gpu_model->tensors[params_i32_handle.id] = params_i32_handle.tensor_desc;
  for (int i = 0; i < config_.stack_size; ++i) {
    gpu_model->tensors[k_caches_in_handles[i].id] =
        k_caches_in_handles[i].tensor_desc;
    gpu_model->tensors[v_caches_in_handles[i].id] =
        v_caches_in_handles[i].tensor_desc;
  }

  return absl::OkStatus();
}

absl::Status Qwen3ModelBuilder::Build(
    GpuModel* prefill_model, GpuModel* decode_model,
    GpuModelBuilder::TensorHandle* input_handle,
    GpuModelBuilder::TensorHandle* decode_input_handle,
    GpuModelBuilder::TensorHandle* params_i32_handle,
    GpuModelBuilder::TensorHandle* output_handle,
    GpuModelBuilder::TensorHandle* decode_output_handle,
    std::vector<GpuModelBuilder::TensorHandle>* k_caches_in_handles,
    std::vector<GpuModelBuilder::TensorHandle>* v_caches_in_handles) {
  if (config_.qwen35) {
    const auto& config = *config_.qwen35;
    const int rotary_dim =
        static_cast<int>(config_.head_dimension * config.partial_rotary_factor);
    if (config.full_attention_interval < 1 || config.linear_conv_kernel_dim < 2 ||
        config.linear_key_head_dim < 4 || config.linear_key_head_dim % 4 != 0 ||
        config.linear_value_head_dim < 4 || config.linear_value_head_dim % 4 != 0 ||
        config.linear_num_key_heads < 1 || config.linear_num_value_heads < 1 ||
        config.linear_num_value_heads % config.linear_num_key_heads != 0 ||
        config_.head_dimension % 8 != 0 || rotary_dim < 8 || rotary_dim % 8 != 0 ||
        rotary_dim > config_.head_dimension) {
      return absl::InvalidArgumentError("Invalid Qwen3.5 attention dimensions");
    }
  }
  // Create shared KV cache tensors
  CreateKVCache(k_caches_in_handles, v_caches_in_handles);

  TensorDescriptor params_td = TensorDescriptor{
      DataType::kInt32, TensorStorageType::kBuffer, Layout::kHWC};
  params_td.SetBHWCShape(BHWC(1, 1, 1, LlmRuntimeParams::kTotalParamsCount));
  *params_i32_handle = model_builder_.AddTensor(params_td);

  // Build prefill model
  ABSL_RETURN_IF_ERROR(BuildModelInternal(
      config_.sequence_size, prefill_model, *params_i32_handle,
      *k_caches_in_handles, *v_caches_in_handles, input_handle, output_handle));

  // Build decode model
  ABSL_RETURN_IF_ERROR(BuildModelInternal(
      1, decode_model, *params_i32_handle, *k_caches_in_handles,
      *v_caches_in_handles, decode_input_handle, decode_output_handle));

  return absl::OkStatus();
}

absl::Status Qwen3ModelBuilder::BuildPostProcessGreedy(
    GpuModel* gpu_model, GpuModelBuilder::TensorHandle* input_logits_handle,
    GpuModelBuilder::TensorHandle* output_token_handle) {
  BuildContext ctx{
      model_builder_,      create_info_, tensor_loader_.get(), config_, 1,
      *input_logits_handle};
  DataType input_type = ctx.data_type();

  // Input is expected to be (B, 1, 1, Vocab)
  TensorHandle logits = ctx.AddTensor(
      BHWC{config_.batch_size, 1, 1, config_.vocabulary_size}, input_type);
  *input_logits_handle = logits;

  // ArgMax via Reduce across vocabulary dimension
  TensorHandle output_token =
      ctx.Reduce(logits, Reduce::Type::kMaximumIndex, {Axis::kChannels});
  *output_token_handle = output_token;

  return model_builder_.GetGpuModel(
      std::vector<ValueId>{input_logits_handle->id},
      std::vector<ValueId>{output_token_handle->id}, gpu_model);
}

}  // namespace ml_drift
