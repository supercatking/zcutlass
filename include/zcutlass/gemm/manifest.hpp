#pragma once

#include <cstddef>

#include "zcutlass/gemm/operation.hpp"

namespace zcutlass::gemm_api {

class Manifest {
 public:
  static constexpr int kMaxOperations = 64;

  void append(const GemmOperation* operation);
  const GemmOperation* find_best(const GemmArguments& args,
                                 const GemmPreference& pref) const;
  int size() const;
  const GemmOperation* at(int index) const;

 private:
  const GemmOperation* operations_[kMaxOperations] = {};
  int size_ = 0;
};

Manifest& global_manifest();
void initialize_builtin_operations();

}  // namespace zcutlass::gemm_api

