#ifndef OJASX_SEMANTICS_COST_MONOID_HPP
#define OJASX_SEMANTICS_COST_MONOID_HPP

#include <cstddef>
#include <algorithm>
#include "../core/monoid.hpp"

namespace ojasx::semantics {

// ============================================================================
// MORPHISM COST MONOID (THERMODYNAMIC & HARDWARE METRICS)
// ============================================================================

struct MorphismCost {
    std::size_t memory_bytes = 0;
    std::size_t flops        = 0;
    double energy_joules     = 0.0;
    double bandwidth_gbps    = 0.0;
    double latency_us        = 0.0;
    double thermal_resistance_k_per_w = 0.25; // Default silicon packaging thermal resistance

    // Monoidal Identity Element e
    static constexpr MorphismCost identity() noexcept {
        return MorphismCost{0, 0, 0.0, 0.0, 0.0, 0.25};
    }

    // Monoidal Binary Combination Operator (+)
    constexpr MorphismCost operator+(const MorphismCost& rhs) const noexcept {
        return MorphismCost{
            memory_bytes + rhs.memory_bytes,
            flops + rhs.flops,
            energy_joules + rhs.energy_joules,
            std::max(bandwidth_gbps, rhs.bandwidth_gbps),
            latency_us + rhs.latency_us,
            std::max(thermal_resistance_k_per_w, rhs.thermal_resistance_k_per_w)
        };
    }

    // Average Operating Power Dissipation (Watts)
    constexpr double power_watts() const noexcept {
        if (latency_us <= 0.0) return 0.0;
        return (energy_joules / (latency_us * 1e-6));
    }

    // Peak Estimated Temperature Delta (Kelvin above ambient)
    constexpr double delta_temperature_c() const noexcept {
        return power_watts() * thermal_resistance_k_per_w;
    }

    // Pareto Dominance: returns true if this cost is strictly better than or equal in all dimensions
    constexpr bool is_pareto_dominant_over(const MorphismCost& other) const noexcept {
        bool better_or_equal = (energy_joules <= other.energy_joules) &&
                               (latency_us <= other.latency_us) &&
                               (memory_bytes <= other.memory_bytes);
        bool strictly_better = (energy_joules < other.energy_joules) ||
                               (latency_us < other.latency_us) ||
                               (memory_bytes < other.memory_bytes);
        return better_or_equal && strictly_better;
    }

    // Cost Optimization Scalar Scoring (Weight Vector W)
    constexpr double score(double w_latency = 1.0, double w_energy = 0.5, double w_mem = 0.001) const noexcept {
        return (latency_us * w_latency) + (energy_joules * w_energy) + (static_cast<double>(memory_bytes) * w_mem);
    }

    // Monoidal Transfer Cost for PCIe / Memory Bus Transits
    static constexpr MorphismCost transfer_cost(std::size_t bytes, double bus_bw_gbps = 32.0, double energy_per_byte_nj = 5.0) noexcept {
        double bytes_d = static_cast<double>(bytes);
        double latency_s = bytes_d / (bus_bw_gbps * 1e9);
        double energy_j = bytes_d * (energy_per_byte_nj * 1e-9);
        return MorphismCost{
            .memory_bytes = bytes,
            .flops = 0,
            .energy_joules = energy_j,
            .bandwidth_gbps = bus_bw_gbps,
            .latency_us = latency_s * 1e6,
            .thermal_resistance_k_per_w = 0.1
        };
    }
};

} // namespace ojasx::semantics

#endif // OJASX_SEMANTICS_COST_MONOID_HPP
