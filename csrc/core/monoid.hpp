#ifndef OJASX_CORE_MONOID_HPP
#define OJASX_CORE_MONOID_HPP

#include <concepts>
#include <utility>

namespace ojasx::core {

/// Monoid Concept (M, binary_op, identity_element)
template <typename M>
concept Monoid = std::copyable<M> && requires(M a, M b) {
    { M::identity() } -> std::same_as<M>;
    { combine(a, b) } -> std::same_as<M>;
};

/// Default Monoidal Binary Combiner
template <Monoid M>
constexpr M combine(const M& a, const M& b) {
    return a + b;
}

} // namespace ojasx::core

#endif // OJASX_CORE_MONOID_HPP
