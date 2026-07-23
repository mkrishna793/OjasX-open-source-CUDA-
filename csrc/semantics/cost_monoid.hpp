#ifndef OJASX_SEMANTICS_COST_MONOID_HPP
#define OJASX_SEMANTICS_COST_MONOID_HPP

#include <cstddef>
#include <algorithm>
#include "../core/monoid.hpp"

namespace ojasx::semantics {

// ============================================================================
// MORPHISM COST MONOID STRUCT
// ============================================================================

struct MorphismCost {
    std::size_t memory_bytes = 0;
    std::size_t flops        = 0;
    double energy_joules     = 0.0;
    double bandwidth_gbps    = 0.0;
    double latency_us        = 0.0;

    // Monoidal Identity Element e
    static constexpr MorphismCost identity() noexcept {
        return MorphismCost{0, 0, 0.0, 0.0, 0.0};
    }

    // Monoidal Binary Combination Operator (+)
    constexpr MorphismCost operator+(const MorphismCost& rhs) const noexcept {
        return MorphismCost{
            memory_bytes + rhs.memory_bytes,
            flops + rhs.flops,
            energy_joules + rhs.energy_joules,
            std::max(bandwidth_gbps, rhs.bandwidth_gbps),
            latency_us + rhs.latency_us
        };
    }

    // Cost Optimization Scalar Scoring (Weight Vector W)
    constexpr double score(double w_latency = 1.0, double w_energy = 0.5, double w_mem = 0.001) const noexcept {
        return (latency_us * w_latency) + (energy_joules * w_energy) + (static_cast<double>(memory_bytes) * w_mem);
    }
};

} // namespace ojasx::semantics

#endif // OJASX_SEMANTICS_COST_MONOID_HPP
