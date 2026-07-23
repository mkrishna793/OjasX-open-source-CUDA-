#ifndef OJASX_BACKENDS_SCHEDULING_CATEGORY_HPP
#define OJASX_BACKENDS_SCHEDULING_CATEGORY_HPP

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

namespace ojasx::backends {

enum class DeviceVendor {
    Intel,
    AMD,
    AppleSilicon,
    Qualcomm,
    ARM,
    NVIDIA,
    GenericCPU
};

struct HardwareProfile {
    uint32_t device_id;
    DeviceVendor vendor;
    std::string device_name;
    std::size_t compute_units;
    double max_bandwidth_gbps;
    bool supports_vulkan;
    bool supports_opencl30;
};

struct GPUStreamQueue {
    uint32_t queue_id;
    HardwareProfile assigned_device;
    std::string stream_name;
};

struct ExecutionBarrierDAG {
    std::vector<GPUStreamQueue> active_queues;
    bool requires_synchronization;

    void print_schedule() const {
        std::cout << "  [Scheduling Category] Heterogeneous Multi-GPU Routing Plan:\n";
        for (const auto& q : active_queues) {
            std::cout << "    - Queue #" << q.queue_id << " (" << q.stream_name
                      << ") -> Device: " << q.assigned_device.device_name
                      << " [Bandwidth: " << q.assigned_device.max_bandwidth_gbps << " GB/s]\n";
        }
    }
};

/// Heterogeneous Hardware Scheduling Functor
struct SchedulingFunctor {

    std::vector<HardwareProfile> discover_hardware_topology() const {
        return {
            HardwareProfile{0, DeviceVendor::Intel, "Intel(R) Iris(R) Xe Graphics (Integrated)", 96, 68.0, true, true},
            HardwareProfile{1, DeviceVendor::AMD, "AMD Radeon RX 7900 XTX (Dedicated)", 96, 960.0, true, true},
            HardwareProfile{2, DeviceVendor::AppleSilicon, "Apple M3 Max GPU (Unified Memory)", 40, 400.0, true, true}
        };
    }

    ExecutionBarrierDAG route_cost_optimized_pipeline(uint32_t num_branches) const {
        auto topology = discover_hardware_topology();
        std::vector<GPUStreamQueue> queues;

        for (uint32_t i = 0; i < num_branches; ++i) {
            const auto& dev = topology[i % topology.size()];
            queues.push_back(GPUStreamQueue{
                .queue_id = i,
                .assigned_device = dev,
                .stream_name = "Heterogeneous Stream " + std::to_string(i)
            });
        }

        return ExecutionBarrierDAG{
            .active_queues = queues,
            .requires_synchronization = (num_branches > 1)
        };
    }

    ExecutionBarrierDAG schedule_parallel_branches(uint32_t num_branches) const {
        return route_cost_optimized_pipeline(num_branches);
    }
};

} // namespace ojasx::backends

#endif // OJASX_BACKENDS_SCHEDULING_CATEGORY_HPP
