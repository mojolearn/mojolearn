// SPDX-License-Identifier: Apache-2.0
// Optional native PyTorch dispatcher registration. No Python callbacks.
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <torch/library.h>
#include "abi.h"
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
constexpr int64_t kRank = 8, kWords = 11;
using Descriptor = std::array<int64_t, kWords>;

void check_shape(at::IntArrayRef shape) {
  TORCH_CHECK(shape.size() <= kRank, "mojolearn: tensor rank exceeds eight");
  int64_t count = 1;
  for (auto dim : shape) {
    TORCH_CHECK(dim > 0, "mojolearn: empty and negative dimensions are unsupported");
    TORCH_CHECK(count <= (int64_t(1) << 58) / dim, "mojolearn: tensor shape overflows native storage");
    count *= dim;
  }
}

void check_tensor(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.layout() == at::kStrided && tensor.is_contiguous(),
              "mojolearn: pass a contiguous strided tensor");
  TORCH_CHECK(!tensor.is_conj() && !tensor.is_neg(),
              "mojolearn: resolve conjugate/negative views before calling");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kInt,
              "mojolearn: tensor storage must be float32 or int32");
  check_shape(tensor.sizes());
}

Descriptor describe(const at::Tensor& tensor) {
  check_tensor(tensor);
  TORCH_CHECK(!tensor.is_mps(), "mojolearn: MPS storage must cross the native host-buffer boundary");
  Descriptor d{};
  d[0] = reinterpret_cast<int64_t>(tensor.const_data_ptr());
  d[1] = tensor.scalar_type() == at::kInt;
  d[2] = tensor.dim();
  for (int64_t j = 0; j < tensor.dim(); ++j) d[3 + j] = tensor.size(j);
  return d;
}

at::Tensor operation(
    at::TensorList inputs, int64_t code, at::IntArrayRef output_shape,
    at::IntArrayRef options, at::ArrayRef<double> scalars, bool integer_output,
    std::optional<at::Device> requested_device) {
  TORCH_CHECK(mojolearn_adapter_abi() == 1, "mojolearn: incompatible adapter ABI");
  TORCH_CHECK(code >= 1 && code <= 80 && code != 47, "mojolearn: unknown operation code");
  check_shape(output_shape);
  auto device = requested_device.value_or(inputs.empty() ? at::Device(at::kCPU) : inputs[0].device());
  if (device.is_mps() && !device.has_index()) device = at::Device(at::kMPS, 0);
  TORCH_CHECK(device.is_cpu() || ((device.is_cuda() || device.is_mps()) && device.index() == 0),
              "mojolearn: this adapter build supports CPU and visible CUDA/HIP/MPS GPU zero");
  const c10::OptionalDeviceGuard guard(device);
  bool host_staging = device.is_mps();
#ifdef MOJOLEARN_ADAPTER_TEST_HOST_STAGING
  // Qualification artifact only: exercise the same transfer path on CUDA/HIP.
  // Never set this define when building a distributable extension.
  host_staging = !device.is_cpu();
#endif
  // Validate every input before copying any storage. In particular, obtaining
  // data_ptr() from an MPS tensor does not yield a Mojo device address.
  for (const auto& input : inputs) {
    TORCH_CHECK(input.device() == device, "mojolearn: every tensor must share the output device");
    TORCH_CHECK(!input.requires_grad(), "mojolearn: this inference entry has no registered backward; detach explicitly");
    check_tensor(input);
  }
  std::vector<at::Tensor> staged_inputs;
  staged_inputs.reserve(inputs.size());
  std::vector<int64_t> descriptors;
  descriptors.reserve(inputs.size() * kWords);
  for (const auto& input : inputs) {
    if (host_staging) {
      // Ordinary native ATen copy, unchanged dtype, blocking completion. No
      // foreign Metal pointers, Python calls, casts, or CPU numerical fallback.
      staged_inputs.push_back(input.to(at::Device(at::kCPU), false, true));
    }
    auto descriptor = describe(host_staging ? staged_inputs.back() : input);
    descriptors.insert(descriptors.end(), descriptor.begin(), descriptor.end());
  }
  std::vector<float> args;
  args.reserve(scalars.size());
  for (auto value : scalars) args.push_back(static_cast<float>(value));
  auto output = at::empty(output_shape, at::TensorOptions().device(device).dtype(integer_output ? at::kInt : at::kFloat));
  auto destination = host_staging ? at::empty(output_shape, output.options().device(at::kCPU)) : output;
  auto out_descriptor = describe(destination);
  // Public native stream API. The Mojo executor uses its own context/queue;
  // wait for producer work before handing it addresses, and drain the Mojo
  // queue before returning. This baseline deliberately makes no async claim.
  if (device.is_cuda()) {
    c10::impl::VirtualGuardImpl impl(device.type());
    impl.getStream(device).synchronize();
  }
  std::array<char, 2048> error{};
  // MPS-to-host copies above already wait for their producers. The new ABI
  // uploads those bytes into Mojo-owned Metal buffers and keeps all math on GPU.
  const auto entry = host_staging ? mojolearn_adapter_operation_host_buffers : mojolearn_adapter_operation;
  const auto status = entry(
      device.is_cpu() ? -1 : 0, code,
      reinterpret_cast<int64_t>(descriptors.data()), inputs.size(),
      reinterpret_cast<int64_t>(out_descriptor.data()),
      reinterpret_cast<int64_t>(options.data()), options.size(),
      reinterpret_cast<int64_t>(args.data()), args.size(),
      reinterpret_cast<int64_t>(error.data()), error.size());
  TORCH_CHECK(status == 0, "mojolearn: ", error.data());
  if (host_staging) {
    output.copy_(destination, false);
    // Keep staging owners alive through final copy completion. MPS supports
    // device synchronization; its generic stream-synchronize API is incomplete.
    c10::impl::VirtualGuardImpl impl(device.type());
    impl.synchronizeDevice(device.index());
  }
  return output;
}

at::Tensor operation_meta(
    at::TensorList inputs, int64_t code, at::IntArrayRef output_shape,
    at::IntArrayRef options, at::ArrayRef<double> scalars, bool integer_output,
    std::optional<at::Device> requested_device) {
  TORCH_CHECK(code >= 1 && code <= 80 && code != 47, "mojolearn: unknown operation code");
  check_shape(output_shape);
  return at::empty(output_shape, at::TensorOptions().device(at::kMeta).dtype(integer_output ? at::kInt : at::kFloat));
}

std::vector<at::Tensor> training_outputs(at::TensorList inputs, at::IntArrayRef options,
                                         at::Device device) {
  TORCH_CHECK(inputs.size() == 3 || inputs.size() == 7,
              "mojolearn: training takes three fresh or seven resume tensors");
  TORCH_CHECK(options.size() == 13 && options[7] > 0, "mojolearn: invalid training options");
  TORCH_CHECK(inputs[2].dim() == 1 && inputs[2].numel() > 0,
              "mojolearn: parameters must be a nonempty vector");
  int64_t previous = inputs.size() == 7 ? inputs[6].numel() : 0;
  TORCH_CHECK(previous <= (int64_t(1) << 58) - options[7], "mojolearn: curve capacity overflow");
  auto spec = at::TensorOptions().device(device).dtype(at::kFloat);
  std::vector<at::Tensor> result;
  for (int j = 0; j < 3; ++j) result.push_back(at::empty({inputs[2].numel()}, spec));
  result.push_back(at::empty({24}, spec.dtype(at::kInt)));
  result.push_back(at::empty({previous + options[7]}, spec));
  return result;
}

std::vector<at::Tensor> train(at::TensorList inputs, at::IntArrayRef model,
                              at::IntArrayRef options, at::ArrayRef<double> scalars) {
  TORCH_CHECK(mojolearn_adapter_abi() == 1, "mojolearn: incompatible adapter ABI");
  TORCH_CHECK(inputs.size() == 3 || inputs.size() == 7,
              "mojolearn: training takes three fresh or seven resume tensors");
  auto device = inputs[0].device();
  if (device.is_mps() && !device.has_index()) device = at::Device(at::kMPS, 0);
  TORCH_CHECK(device.is_cpu() || ((device.is_cuda() || device.is_mps()) && device.index() == 0),
              "mojolearn: training supports CPU and visible CUDA/HIP/MPS GPU zero");
  const c10::OptionalDeviceGuard guard(device);
  bool host_staging = device.is_mps();
#ifdef MOJOLEARN_ADAPTER_TEST_HOST_STAGING
  host_staging = !device.is_cpu();
#endif
  for (const auto& input : inputs) {
    check_tensor(input);
    TORCH_CHECK(input.device() == device, "mojolearn: training tensors must share one device");
    TORCH_CHECK(!input.requires_grad(),
                "mojolearn: whole training owns its tape and optimizer; detach input tensors explicitly");
  }
  auto outputs = training_outputs(inputs, options, device);
  std::vector<at::Tensor> staged_inputs, staged_outputs;
  std::vector<int64_t> descriptors, destinations;
  for (const auto& input : inputs) {
    if (host_staging) staged_inputs.push_back(input.to(at::Device(at::kCPU), false, true));
    auto d = describe(host_staging ? staged_inputs.back() : input);
    descriptors.insert(descriptors.end(), d.begin(), d.end());
  }
  for (const auto& output : outputs) {
    if (host_staging) staged_outputs.push_back(at::empty(output.sizes(), output.options().device(at::kCPU)));
    auto d = describe(host_staging ? staged_outputs.back() : output);
    destinations.insert(destinations.end(), d.begin(), d.end());
  }
  if (device.is_cuda()) {
    c10::impl::VirtualGuardImpl impl(device.type());
    impl.getStream(device).synchronize();
  }
  std::array<char, 2048> error{};
  const auto train_status = mojolearn_adapter_train(
      device.is_cpu() ? -1 : 0, host_staging,
      reinterpret_cast<int64_t>(descriptors.data()), inputs.size(),
      reinterpret_cast<int64_t>(destinations.data()),
      reinterpret_cast<int64_t>(model.data()), model.size(),
      reinterpret_cast<int64_t>(options.data()), options.size(),
      reinterpret_cast<int64_t>(scalars.data()), scalars.size(),
      reinterpret_cast<int64_t>(error.data()), error.size());
  TORCH_CHECK(train_status == 0, "mojolearn: ", error.data());
  if (host_staging) {
    for (size_t j = 0; j < outputs.size(); ++j) outputs[j].copy_(staged_outputs[j], false);
    c10::impl::VirtualGuardImpl impl(device.type());
    impl.synchronizeDevice(device.index());
  }
  return outputs;
}

std::vector<at::Tensor> train_meta(at::TensorList inputs, at::IntArrayRef model,
                                   at::IntArrayRef options, at::ArrayRef<double> scalars) {
  return training_outputs(inputs, options, at::Device(at::kMeta));
}
}  // namespace

TORCH_LIBRARY(mojolearn, m) {
  m.def("operation(Tensor[] inputs, int code, int[] output_shape, int[] options=[], float[] scalars=[], bool integer_output=False, Device? device=None) -> Tensor");
  m.def("train(Tensor[] inputs, int[] model, int[] options, float[] scalars) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(mojolearn, CompositeExplicitAutograd, m) {
  m.impl("operation", TORCH_FN(operation));
  m.impl("train", TORCH_FN(train));
}
TORCH_LIBRARY_IMPL(mojolearn, Meta, m) {
  m.impl("operation", TORCH_FN(operation_meta));
  m.impl("train", TORCH_FN(train_meta));
}
