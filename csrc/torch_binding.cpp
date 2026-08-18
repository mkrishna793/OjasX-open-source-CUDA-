/**
 * OjasX C++20 Applied Category Theory (ACT) Native PyTorch Bindings.
 * Bridges native Category Morphisms, Cost Monoids, and Hardware Kernel Synthesizers.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ojasx_engine.hpp"

namespace py = pybind11;
using namespace ojasx;
using namespace ojasx::core;
using namespace ojasx::semantics;
using namespace ojasx::backends;

PYBIND11_MODULE(ojasx_native, m) {
    m.doc() = "OjasX Applied Category Theory (C++20) Native Compute Engine";

    // 1. Engine Information
    m.attr("__version__") = EngineInfo::version;
    m.attr("__paradigm__") = EngineInfo::paradigm;

    // 2. Morphism Cost Monoid
    py::class_<MorphismCost>(m, "MorphismCost")
        .def(py::init<>())
        .def_readwrite("memory_bytes", &MorphismCost::memory_bytes)
        .def_readwrite("flops", &MorphismCost::flops)
        .def_readwrite("energy_joules", &MorphismCost::energy_joules)
        .def_readwrite("bandwidth_gbps", &MorphismCost::bandwidth_gbps)
        .def_readwrite("latency_us", &MorphismCost::latency_us)
        .def_readwrite("thermal_resistance_k_per_w", &MorphismCost::thermal_resistance_k_per_w)
        .def("power_watts", &MorphismCost::power_watts)
        .def("delta_temperature_c", &MorphismCost::delta_temperature_c)
        .def("score", &MorphismCost::score, py::arg("w_latency") = 1.0, py::arg("w_energy") = 0.5, py::arg("w_mem") = 0.001)
        .def("is_pareto_dominant_over", &MorphismCost::is_pareto_dominant_over)
        .def("__add__", [](const MorphismCost& a, const MorphismCost& b) { return a + b; })
        .def_static("identity", &MorphismCost::identity)
        .def_static("transfer_cost", &MorphismCost::transfer_cost, py::arg("bytes"), py::arg("bus_bw_gbps") = 32.0, py::arg("energy_per_byte_nj") = 5.0);

    // 3. Hardware Matrix Acceleration Architecture & MicroKernelConfig
    py::enum_<MatrixHardwareAcceleration>(m, "MatrixHardwareAcceleration")
        .value("StandardSimd", MatrixHardwareAcceleration::StandardSimd)
        .value("IntelDPAS", MatrixHardwareAcceleration::IntelDPAS)
        .value("AMDWMMA", MatrixHardwareAcceleration::AMDWMMA)
        .value("AppleAMX", MatrixHardwareAcceleration::AppleAMX)
        .value("VulkanCoopMatrix", MatrixHardwareAcceleration::VulkanCoopMatrix)
        .export_values();

    py::class_<MicroKernelConfig>(m, "MicroKernelConfig")
        .def(py::init<>())
        .def_readwrite("tile_m", &MicroKernelConfig::tile_m)
        .def_readwrite("tile_n", &MicroKernelConfig::tile_n)
        .def_readwrite("tile_k", &MicroKernelConfig::tile_k)
        .def_readwrite("vector_width", &MicroKernelConfig::vector_width)
        .def_readwrite("subgroup_size", &MicroKernelConfig::subgroup_size)
        .def_readwrite("accel_type", &MicroKernelConfig::accel_type)
        .def_readwrite("enable_fused_activation", &MicroKernelConfig::enable_fused_activation)
        .def_readwrite("activation_type", &MicroKernelConfig::activation_type)
        .def("__repr__", &MicroKernelConfig::to_string);

    // 4. Kernel Synthesizer
    py::class_<KernelSynthesizer>(m, "KernelSynthesizer")
        .def_static("synthesize_gemm", &KernelSynthesizer::synthesize_gemm,
                    py::arg("M"), py::arg("N"), py::arg("K"), py::arg("vendor"), py::arg("is_fp16") = false)
        .def_static("emit_fused_gemm_kernel", &KernelSynthesizer::emit_fused_gemm_kernel)
        .def_static("emit_fused_backward_gemm_kernel", &KernelSynthesizer::emit_fused_backward_gemm_kernel);

    // 5. Liquid Compute Runtime & Telemetry
    py::class_<TelemetrySnapshot>(m, "TelemetrySnapshot")
        .def_readonly("device_id", &TelemetrySnapshot::device_id)
        .def_readonly("device_name", &TelemetrySnapshot::device_name)
        .def_readonly("vram_used_mb", &TelemetrySnapshot::vram_used_mb)
        .def_readonly("vram_total_mb", &TelemetrySnapshot::vram_total_mb)
        .def_readonly("gpu_utilization_pct", &TelemetrySnapshot::gpu_utilization_pct)
        .def_readonly("temperature_celsius", &TelemetrySnapshot::temperature_celsius)
        .def_readonly("active_bandwidth_gbps", &TelemetrySnapshot::active_bandwidth_gbps);

    py::class_<LiquidWorkChunk>(m, "LiquidWorkChunk")
        .def_readonly("chunk_id", &LiquidWorkChunk::chunk_id)
        .def_readonly("num_elements", &LiquidWorkChunk::num_elements)
        .def_readonly("memory_bytes", &LiquidWorkChunk::memory_bytes)
        .def_readonly("target_device_id", &LiquidWorkChunk::target_device_id)
        .def_readonly("is_migrating", &LiquidWorkChunk::is_migrating);

    py::class_<LiquidComputeRuntime>(m, "LiquidComputeRuntime")
        .def(py::init<>())
        .def("get_telemetry", &LiquidComputeRuntime::get_telemetry)
        .def("fluid_dispatch", &LiquidComputeRuntime::fluid_dispatch, py::arg("total_elements"), py::arg("element_size_bytes") = 4);
}
