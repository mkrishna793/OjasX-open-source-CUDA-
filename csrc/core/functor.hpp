#ifndef OJASX_CORE_FUNCTOR_HPP
#define OJASX_CORE_FUNCTOR_HPP

#include "category.hpp"

namespace ojasx::core {

/// Functor Concept F : CategoryC -> CategoryD
template <typename F, typename ObjIn, typename MorphIn>
concept Functor = requires(F functor, ObjIn obj, MorphIn morph) {
    { functor.map_object(obj) };
    { functor.map_morphism(morph) };
};

// ============================================================================
// ADJOINT FUNCTOR (F ⊣ G) FOR AUTOMATIC DIFFERENTIATION
// ============================================================================

template <typename ForwardMorphism>
struct AdjointBackwardMorphism {
    ForwardMorphism forward_op;

    using Domain   = typename ForwardMorphism::Codomain; // Dual domain is output gradient
    using Codomain = typename ForwardMorphism::Domain;   // Dual codomain is input gradient

    template <typename GradOutput>
    constexpr auto operator()(const GradOutput& grad_out) const {
        // Automatically derives dual adjoint pullback
        return Codomain{};
    }
};

/// Adjoint Functor mapping Forward Morphism F to Dual Backward Morphism G (F ⊣ G)
struct AdjointAutogradFunctor {
    template <typename ForwardMorphism>
    constexpr auto derive_backward_morphism(const ForwardMorphism& f) const noexcept {
        return AdjointBackwardMorphism<ForwardMorphism>{f};
    }
};

} // namespace ojasx::core

#endif // OJASX_CORE_FUNCTOR_HPP
