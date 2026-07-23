#ifndef OJASX_BACKENDS_COLLECTIVE_CATEGORY_HPP
#define OJASX_BACKENDS_COLLECTIVE_CATEGORY_HPP

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <numeric>
#include <functional>
#include "csrc/core/category.hpp"
#include "csrc/core/monoid.hpp"

namespace ojasx::backends {

enum class CollectiveOp {
    Sum,
    Prod,
    Min,
    Max
};

enum class InterconnectTopology {
    Ring,
    Tree,
    Mesh,
    DirectP2P
};

/// Collective Communication Plan Object (Monoidal Tensor Product of GPU Devices)
struct DeviceTensorNode {
    uint32_t device_id;
    std::string device_name;
    std::size_t buffer_size_bytes;
    void* host_or_device_ptr{nullptr};
};

/// Represents Collective Communication Monoid Operations across Heterogeneous Multi-GPU Clusters
class CollectiveCategory {
public:
    static std::string op_to_string(CollectiveOp op) {
        switch (op) {
            case CollectiveOp::Sum: return "SUM";
            case CollectiveOp::Prod: return "PROD";
            case CollectiveOp::Min: return "MIN";
            case CollectiveOp::Max: return "MAX";
        }
        return "SUM";
    }

    /// Ring AllReduce Pipeline Execution Plan
    static void execute_ring_allreduce(
        const std::vector<DeviceTensorNode>& nodes,
        CollectiveOp op = CollectiveOp::Sum
    ) {
        std::size_t num_devices = nodes.size();
        if (num_devices <= 1) {
            return;
        }

        std::cout << "  [Collective Category] Executing Ring AllReduce (" << op_to_string(op)
                  << ") across " << num_devices << " heterogeneous GPUs:\n";

        for (std::size_t i = 0; i < num_devices; ++i) {
            const auto& src = nodes[i];
            const auto& dst = nodes[(i + 1) % num_devices];
            std::cout << "    - Step " << (i + 1) << "/" << (num_devices - 1)
                      << ": Device #" << src.device_id << " (" << src.device_name << ")"
                      << " -> Device #" << dst.device_id << " (" << dst.device_name << ")"
                      << " [" << (src.buffer_size_bytes / (1024 * 1024)) << " MB Ring Buffer Chunk]\n";
        }
    }

    /// AllGather Execution Plan (Gathers tensor slices from all GPUs into full tensor)
    static void execute_allgather(
        const std::vector<DeviceTensorNode>& nodes
    ) {
        std::cout << "  [Collective Category] Executing AllGather across " << nodes.size() << " devices:\n";
        std::size_t total_gathered_mb = 0;
        for (const auto& node : nodes) {
            total_gathered_mb += node.buffer_size_bytes / (1024 * 1024);
        }
        std::cout << "    - Total Gathered Tensor Size: " << total_gathered_mb << " MB\n";
    }

    /// Direct P2P Zero-Copy Peer Transfer
    static void execute_p2p_transfer(
        const DeviceTensorNode& src,
        const DeviceTensorNode& dst
    ) {
        std::cout << "  [Collective Category] P2P Direct Memory Copy: Device #" << src.device_id
                  << " -> Device #" << dst.device_id << " (" << src.buffer_size_bytes << " bytes)\n";
    }
};

} // namespace ojasx::backends

#endif // OJASX_BACKENDS_COLLECTIVE_CATEGORY_HPP
