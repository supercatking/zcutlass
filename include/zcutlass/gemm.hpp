#pragma once

#include "zcutlass/arch/arch.hpp"
#include "zcutlass/gemm/gemm.hpp"
#include "zcutlass/gemm/manifest.hpp"
#include "zcutlass/gemm/operation.hpp"
#include "zcutlass/layout/layout.hpp"
#include "zcutlass/numeric_types.hpp"
#include "zcutlass/status.hpp"

namespace zcutlass {

int version_major();
int version_minor();
int version_patch();

}  // namespace zcutlass
