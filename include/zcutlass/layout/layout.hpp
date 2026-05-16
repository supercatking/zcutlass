#pragma once

#include <cstdint>

namespace zcutlass::layout {

struct RowMajor {};
struct ColumnMajor {};

enum class LayoutKind {
  RowMajor,
  ColumnMajor,
};

struct MatrixLayout {
  LayoutKind kind = LayoutKind::RowMajor;
  int64_t stride = 0;
};

inline int64_t offset(MatrixLayout layout, int64_t row, int64_t col) {
  if (layout.kind == LayoutKind::RowMajor) {
    return row * layout.stride + col;
  }
  return col * layout.stride + row;
}

inline const char* layout_name(LayoutKind kind) {
  return kind == LayoutKind::RowMajor ? "row" : "column";
}

}  // namespace zcutlass::layout

