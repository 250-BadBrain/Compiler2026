#pragma once

namespace sysyc::config {

// Structural replacements are enabled only through IR/CFG/dataflow matchers.
// Keep both switches together so tests exercise the same optimized path that is
// intended for submission.
inline constexpr bool kEnableStructuralSpecializations = true;
inline constexpr bool kEnableGenericKernelLowering = true;

} // namespace sysyc::config
