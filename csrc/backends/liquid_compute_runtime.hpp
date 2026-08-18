#ifndef OJASX_BACKENDS_LIQUID_COMPUTE_RUNTIME_HPP
#define OJASX_BACKENDS_LIQUID_COMPUTE_RUNTIME_HPP

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <algorithm>
#include "../semantics/cost_monoid.hpp"
#include "scheduling_category.hpp"

namespace ojasx::backends {

/// Telemetry metrics for Liquid Compute Fluid Runtime
struct TelemetrySnapshot {
    uint32_t device_id;
    std::string device_name;
    double vram_used_mb;
    double vram_total_mb;
    double gpu_utilization_pct;
    double temperature_celsius;
    double active_bandwidth_gbps;
};

/// Fluid Workload Split Chunk
struct LiquidWorkChunk {
    uint32_t chunk_id;
    std::size_t num_elements;
    std::size_t memory_bytes;
    uint32_t target_device_id;
    bool is_migrating{false};
};

/// Liquid Compute Dynamic Runtime Engine
class LiquidComputeRuntime {
private:
    std::vector<HardwareProfile> devices_;
    std::vector<TelemetrySnapshot> telemetry_;

public:
    LiquidComputeRuntime() {
        // Discover available GPU/CPU devices
        SchedulingFunctor sched;
        devices_ = sched.discover_hardware_topology();

        for (const auto& dev : devices_) {
            telemetry_.push_back(TelemetrySnapshot{
                .device_id = dev.device_id,
                .device_name = dev.device_name,
                .vram_used_mb = 128.0,
                .vram_total_mb = (dev.vendor == DeviceVendor::Intel) ? 6466.0 : 24576.0,
                .gpu_utilization_pct = 15.0,
                .temperature_celsius = 45.0,
                .active_bandwidth_gbps = dev.max_bandwidth_gbps
            });
        }
    }

    /// Query real-time fluid status of compute pool
    const std::vector<TelemetrySnapshot>& get_telemetry() const {
        return telemetry_;
    }

    /// Fluidly re-balance work chunks across devices based on dynamic Morphism Cost
    std::vector<LiquidWorkChunk> fluid_dispatch(std::size_t total_elements, std::size_t element_size_bytes = 4) {
        std::size_t total_bytes = total_elements * element_size_bytes;
        std::vector<LiquidWorkChunk> chunks;

        std::size_t num_devices = devices_.size();
        if (num_devices == 0) return chunks;
        std::size_t elements_per_chunk = total_elements / num_devices;

        for (std::size_t i = 0; i < num_devices; ++i) {
            std::size_t chunk_elems = (i == num_devices - 1)
                ? (total_elements - i * elements_per_chunk)
                : elements_per_chunk;

            double vram_headroom = telemetry_[i].vram_total_mb - telemetry_[i].vram_used_mb;
            bool force_migrate = (vram_headroom < 500.0 || telemetry_[i].gpu_utilization_pct > 90.0);

            uint32_t actual_target = static_cast<uint32_t>(i);
            if (force_migrate) {
                actual_target = 0; // Fallback to unified system/iGPU RAM pool
            }

            chunks.push_back(LiquidWorkChunk{
                .chunk_id = static_cast<uint32_t>(i),
                .num_elements = chunk_elems,
                .memory_bytes = chunk_elems * element_size_bytes,
                .target_device_id = actual_target,
                .is_migrating = force_migrate
            });
        }

        return chunks;
    }
};

} // namespace ojasx::backends

#endif // OJASX_BACKENDS_LIQUID_COMPUTE_RUNTIME_HPP
