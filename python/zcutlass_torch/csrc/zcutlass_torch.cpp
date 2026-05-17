#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <optional>
#include <sstream>

#include "zcutlass/gemm.hpp"

namespace {

zcutlass::DType dtype_from_tensor(const at::Tensor& tensor) {
  if (tensor.scalar_type() == at::kHalf) {
    return zcutlass::DType::F16;
  }
  if (tensor.scalar_type() == at::kBFloat16) {
    return zcutlass::DType::BF16;
  }
  TORCH_CHECK(false, "zcutlass_torch only supports float16 and bfloat16 tensors");
}

void check_2d_cuda_contiguous(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.dim() == 2, name, " must be rank-2");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be row-major contiguous");
  (void)dtype_from_tensor(tensor);
}

const void* optional_const_ptr(const std::optional<at::Tensor>& tensor) {
  return tensor.has_value() ? tensor->data_ptr() : nullptr;
}

at::Tensor zcutlass_gemm(torch::Tensor a,
                         torch::Tensor b,
                         std::optional<torch::Tensor> c,
                         std::optional<torch::Tensor> bias,
                         double alpha,
                         double beta) {
  check_2d_cuda_contiguous(a, "A");
  check_2d_cuda_contiguous(b, "B");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "A and B dtype must match");
  TORCH_CHECK(a.size(1) == b.size(0), "A.shape[1] must equal B.shape[0]");
  TORCH_CHECK(a.device() == b.device(), "A and B must be on the same CUDA device");

  if (c.has_value()) {
    check_2d_cuda_contiguous(*c, "C");
    TORCH_CHECK(c->scalar_type() == a.scalar_type(), "C dtype must match A/B");
    TORCH_CHECK(c->device() == a.device(), "C must be on the same CUDA device");
    TORCH_CHECK(c->size(0) == a.size(0) && c->size(1) == b.size(1),
                "C shape must be [A.shape[0], B.shape[1]]");
  }
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias->dim() == 1, "bias must be rank-1");
    TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias->scalar_type() == a.scalar_type(), "bias dtype must match A/B");
    TORCH_CHECK(bias->device() == a.device(), "bias must be on the same CUDA device");
    TORCH_CHECK(bias->size(0) == b.size(1), "bias length must equal B.shape[1]");
  }

  const c10::cuda::CUDAGuard device_guard(a.device());
  at::Tensor d = at::empty({a.size(0), b.size(1)}, a.options());

  zcutlass::GemmDesc desc{a.size(0),
                          b.size(1),
                          a.size(1),
                          a.stride(0),
                          b.stride(0),
                          c.has_value() ? c->stride(0) : b.size(1),
                          d.stride(0),
                          dtype_from_tensor(a),
                          dtype_from_tensor(b),
                          dtype_from_tensor(a),
                          dtype_from_tensor(d),
                          static_cast<float>(alpha),
                          static_cast<float>(beta),
                          optional_const_ptr(bias),
                          at::cuda::getCurrentCUDAStream().stream()};

  const zcutlass::Status status =
      zcutlass::gemm(desc, a.data_ptr(), b.data_ptr(), optional_const_ptr(c), d.data_ptr());
  TORCH_CHECK(status == zcutlass::Status::Success,
              "zcutlass::gemm failed: ",
              zcutlass::status_to_string(status));
  return d;
}

}  // namespace

TORCH_LIBRARY(zcutlass_torch, m) {
  m.def("gemm(Tensor A, Tensor B, Tensor? C=None, Tensor? bias=None, float alpha=1.0, float beta=0.0) -> Tensor");
  m.impl("gemm", torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(zcutlass_gemm)));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
