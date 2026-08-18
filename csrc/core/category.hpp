#ifndef OJASX_CORE_CATEGORY_HPP
#define OJASX_CORE_CATEGORY_HPP

#include <concepts>
#include <utility>
#include <type_traits>

namespace ojasx::core {

// ============================================================================
// 1. CATEGORICAL CONCEPTS
// ============================================================================

/// Object in Category C: Any valid C++ destructible domain/codomain type
template <typename T>
concept Object = std::destructible<T>;

/// Morphism in Category C: Transformation f: Domain -> Codomain
template <typename Morph, typename Domain, typename Codomain>
concept Morphism = Object<Domain> && Object<Codomain> && requires(Morph m, const Domain& d) {
    { m(d) } -> std::same_as<Codomain>;
};

// ============================================================================
// 2. MONOIDAL UNIT OBJECT (I)
// ============================================================================
struct MonoidalUnit {
    using meaning = void;
    static constexpr size_t size() noexcept { return 0; }
};

// ============================================================================
// 3. IDENTITY MORPHISM (id_A : A -> A)
// ============================================================================
template <Object A>
struct IdentityMorphism {
    using Domain = A;
    using Codomain = A;

    constexpr A operator()(const A& a) const noexcept {
        return a;
    }

    constexpr A operator()(A&& a) const noexcept {
        return std::move(a);
    }
};

// ============================================================================
// 4. DAGGER ADJOINT MORPHISMS (f† : Codomain -> Domain)
// ============================================================================
template <typename Morph>
struct Dagger {
    Morph forward_morph;

    using Domain   = typename Morph::Codomain;
    using Codomain = typename Morph::Domain;

    template <typename Input>
    constexpr auto operator()(Input&& input) const {
        if constexpr (requires { forward_morph.adjoint(std::forward<Input>(input)); }) {
            return forward_morph.adjoint(std::forward<Input>(input));
        } else {
            return Codomain{};
        }
    }

    /// Involution Axiom: (f†)† == f
    constexpr Morph dagger() const noexcept {
        return forward_morph;
    }
};

/// Unary Operator ~ for Dagger Adjoint: ~f == f†
template <typename Morph>
constexpr auto operator~(Morph m) {
    if constexpr (requires { m.dagger(); }) {
        return m.dagger();
    } else {
        return Dagger<Morph>{m};
    }
}

// ============================================================================
// 5. MORPHISM COMPOSITION (g ∘ f : Domain(f) -> Codomain(g))
// ============================================================================
template <typename MorphF, typename MorphG>
struct Composition {
    MorphF f;
    MorphG g;

    using Domain = typename MorphF::Domain;
    using Codomain = typename MorphG::Codomain;

    // Strict Categorical Law Assertion: Codomain(f) == Domain(g)
    static_assert(std::is_same_v<typename MorphF::Codomain, typename MorphG::Domain>,
                  "❌ CATEGORY ERROR: Codomain of f does not match Domain of g in composition (g ∘ f)!");

    template <typename Input>
    constexpr auto operator()(Input&& input) const {
        return g(f(std::forward<Input>(input)));
    }

    /// Dagger Functoriality Axiom: (g ∘ f)† == (f† ∘ g†)
    constexpr auto dagger() const {
        return Composition<Dagger<MorphG>, Dagger<MorphF>>{~g, ~f};
    }
};

/// Operator Overload for Morphism Composition: g * f  ==  (g ∘ f)
template <typename MorphF, typename MorphG>
constexpr auto operator*(MorphG g, MorphF f) {
    return Composition<MorphF, MorphG>{f, g};
}

// ============================================================================
// 6. MONOIDAL TENSOR PRODUCT (f ⊗ g : (DomF ⊗ DomG) -> (CodomF ⊗ CodomG))
// ============================================================================
template <typename MorphF, typename MorphG>
struct TensorProduct {
    MorphF f;
    MorphG g;

    using Domain   = std::pair<typename MorphF::Domain, typename MorphG::Domain>;
    using Codomain = std::pair<typename MorphF::Codomain, typename MorphG::Codomain>;

    template <typename InputA, typename InputB>
    constexpr auto operator()(const std::pair<InputA, InputB>& input) const {
        return std::make_pair(f(input.first), g(input.second));
    }

    /// Monoidal Dagger Axiom: (f ⊗ g)† == (f† ⊗ g†)
    constexpr auto dagger() const {
        return TensorProduct<Dagger<MorphF>, Dagger<MorphG>>{~f, ~g};
    }
};

/// Operator Overload for Monoidal Product: f ^ g  ==  (f ⊗ g)
template <typename MorphF, typename MorphG>
constexpr auto operator^(MorphF f, MorphG g) {
    return TensorProduct<MorphF, MorphG>{f, g};
}

} // namespace ojasx::core

#endif // OJASX_CORE_CATEGORY_HPP
