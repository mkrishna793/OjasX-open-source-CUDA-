#ifndef OJASX_SEMANTICS_SEMANTIC_TENSOR_HPP
#define OJASX_SEMANTICS_SEMANTIC_TENSOR_HPP

#include <cstddef>
#include <type_traits>
#include <tuple>
#include <string_view>
#include <vector>
#include <stdexcept>

namespace ojasx::semantics {

// ============================================================================
// 1. DYNAMIC & STATIC SHAPE ABSTRACTIONS
// ============================================================================

/// Dynamic Dimension Tag for runtime scaling (e.g. Dynamic Batch Size or Sequence Length)
template <std::string_view Tag>
struct DynamicDimension {
    static constexpr bool is_dynamic = true;
    static constexpr std::string_view name = Tag;
};

template <std::size_t N>
struct FixedDimension {
    static constexpr bool is_dynamic = false;
    static constexpr std::size_t value = N;
};

// Shape class supporting both static compile-time checking and dynamic runtime sizes
struct RuntimeShape {
    std::vector<std::size_t> dims;

    std::size_t total_elements() const noexcept {
        std::size_t total = 1;
        for (auto d : dims) total *= d;
        return total;
    }

    bool operator==(const RuntimeShape& other) const noexcept {
        return dims == other.dims;
    }
};

// ============================================================================
// 2. SEMANTIC TAG DEFINITIONS
// ============================================================================

namespace Geometry {
    struct Scalar {};
    struct Euclidean1D {};
    struct Euclidean2D {};
    struct Euclidean3D {};
    struct ManifoldPoint {};
    struct SO3Rotation {};
}

namespace Meaning {
    struct RawArray {};
    struct RGBImage {};
    struct FeatureMap {};
    struct Logits {};
    struct ProbabilityDistribution {};
    struct VelocityVector {};
    struct Embedding {};
    struct AttentionWeights {};
}

namespace Metric {
    struct None {};
    struct L2Euclidean {};
    struct L1Manhattan {};
    struct CosineDistance {};
}

namespace Probability {
    struct Unnormalized {};
    struct NormalizedProbability {};
    struct LogProbability {};
}

namespace Device {
    struct HostRAM {};
    struct IntegratedGPU {};
    struct DedicatedGPU {};
    struct DistributedP2P {};
}

namespace Layout {
    struct NCHW_Contiguous {};
    struct NHWC_Contiguous {};
    struct Tiled32x32 {};
}

// ============================================================================
// 3. THE 7-DIMENSIONAL SEMANTIC TENSOR OBJECT (DYNAMIC & STATIC)
// ============================================================================

template <
    typename ShapeTag,
    typename GeometryTag    = Geometry::Euclidean2D,
    typename MeaningTag     = Meaning::FeatureMap,
    typename MetricTag      = Metric::None,
    typename ProbabilityTag = Probability::Unnormalized,
    typename DeviceTag      = Device::DedicatedGPU,
    typename LayoutTag      = Layout::NCHW_Contiguous
>
struct SemanticTensor {
    using shape       = ShapeTag;
    using geometry    = GeometryTag;
    using meaning     = MeaningTag;
    using metric      = MetricTag;
    using probability = ProbabilityTag;
    using device      = DeviceTag;
    using layout      = LayoutTag;

    RuntimeShape dynamic_shape;

    constexpr SemanticTensor() : dynamic_shape{} {}
    explicit SemanticTensor(RuntimeShape s) : dynamic_shape(std::move(s)) {}

    std::size_t size() const noexcept {
        if (dynamic_shape.dims.empty()) {
            return 1;
        }
        return dynamic_shape.total_elements();
    }
};

} // namespace ojasx::semantics

#endif // OJASX_SEMANTICS_SEMANTIC_TENSOR_HPP
