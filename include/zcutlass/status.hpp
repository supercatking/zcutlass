#pragma once

namespace zcutlass {

enum class Status {
  Success,
  InvalidArgument,
  NotSupported,
  RuntimeError,
};

const char* status_to_string(Status status);

}  // namespace zcutlass

