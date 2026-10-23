// SPDX-License-Identifier: Apache-2.0
// Optional native PyTorch dispatcher registration. No Python callbacks.
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <torch/library.h>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" int64_t mojolearn_adapter_abi();
extern "C" int64_t mojolearn_adapter_operation(
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t);

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

Descriptor describe(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.layout() == at::kStrided && tensor.is_contiguous(),
              "mojolearn: pass a contiguous strided tensor");
  TORCH_CHECK(!tensor.is_conj() && !tensor.is_neg(),
              "mojolearn: resolve conjugate/negative views before calling");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kInt,
              "mojolearn: tensor storage must be float32 or int32");
  check_shape(tensor.sizes());
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
  const auto device = requested_device.value_or(inputs.empty() ? at::Device(at::kCPU) : inputs[0].device());
  TORCH_CHECK(device.is_cpu() || (device.is_cuda() && device.index() == 0),
              "mojolearn: this adapter build supports CPU and visible GPU zero");
  const c10::OptionalDeviceGuard guard(device);
  std::vector<int64_t> descriptors;
  descriptors.reserve(inputs.size() * kWords);
  for (const auto& input : inputs) {
    TORCH_CHECK(input.device() == device, "mojolearn: every tensor must share the output device");
    TORCH_CHECK(!input.requires_grad(), "mojolearn: this inference entry has no registered backward; detach explicitly");
    auto descriptor = describe(input);
    descriptors.insert(descriptors.end(), descriptor.begin(), descriptor.end());
  }
  std::vector<float> args;
  args.reserve(scalars.size());
  for (auto value : scalars) args.push_back(static_cast<float>(value));
  auto output = at::empty(output_shape, at::TensorOptions().device(device).dtype(integer_output ? at::kInt : at::kFloat));
  auto out_descriptor = describe(output);
  // Public native stream API. The Mojo executor uses its own context/queue;
  // wait for producer work before handing it addresses, and drain the Mojo
  // queue before returning. This baseline deliberately makes no async claim.
  if (!device.is_cpu()) {
    c10::impl::VirtualGuardImpl impl(device.type());
    impl.getStream(device).synchronize();
  }
  std::array<char, 2048> error{};
  const auto status = mojolearn_adapter_operation(
      device.is_cpu() ? -1 : 0, code,
      reinterpret_cast<int64_t>(descriptors.data()), inputs.size(),
      reinterpret_cast<int64_t>(out_descriptor.data()),
      reinterpret_cast<int64_t>(options.data()), options.size(),
      reinterpret_cast<int64_t>(args.data()), args.size(),
      reinterpret_cast<int64_t>(error.data()), error.size());
  TORCH_CHECK(status == 0, "mojolearn: ", error.data());
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
}  // namespace

TORCH_LIBRARY(mojolearn, m) {
  m.def("operation(Tensor[] inputs, int code, int[] output_shape, int[] options=[], float[] scalars=[], bool integer_output=False, Device? device=None) -> Tensor");
}
TORCH_LIBRARY_IMPL(mojolearn, CompositeExplicitAutograd, m) {
  m.impl("operation", TORCH_FN(operation));
}
TORCH_LIBRARY_IMPL(mojolearn, Meta, m) {
  m.impl("operation", TORCH_FN(operation_meta));
}
