#ifndef OJASX_ENGINE_HPP
#define OJASX_ENGINE_HPP

// Core Category Theory Abstractions
#include "core/category.hpp"
#include "core/monoid.hpp"
#include "core/functor.hpp"

// Semantic Tensor Objects & Cost Enriched Category
#include "semantics/semantic_tensor.hpp"
#include "semantics/cost_monoid.hpp"

// Hardware Category Layering & Backend Functors
#include "backends/memory_category.hpp"
#include "backends/compute_category.hpp"
#include "backends/scheduling_category.hpp"
#include "backends/vulkan_backend.hpp"
#include "backends/opencl_backend.hpp"

namespace ojasx {

struct EngineInfo {
    static constexpr const char* version = "2.0.0-ACT";
    static constexpr const char* paradigm = "Applied Category Theory (C++20)";
};

} // namespace ojasx

#endif // OJASX_ENGINE_HPP
