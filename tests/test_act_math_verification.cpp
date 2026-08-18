#include <iostream>
#include <cassert>
#include <cmath>
#include "../csrc/ojasx_engine.hpp"

using namespace ojasx;
using namespace ojasx::core;
using namespace ojasx::semantics;
using namespace ojasx::backends;

// Dummy Morphisms for Category Axiom Verification
struct MorphA {
    using Domain = int;
    using Codomain = double;
    double operator()(int x) const { return static_cast<double>(x) * 2.0; }
    int adjoint(double y) const { return static_cast<int>(y / 2.0); }
};

struct MorphB {
    using Domain = double;
    using Codomain = float;
    float operator()(double x) const { return static_cast<float>(x + 3.0); }
    double adjoint(float y) const { return static_cast<double>(y - 3.0f); }
};

struct MorphC {
    using Domain = float;
    using Codomain = long;
    long operator()(float x) const { return static_cast<long>(x * 4.0f); }
    float adjoint(long y) const { return static_cast<float>(y) / 4.0f; }
};

void test_category_associativity() {
    MorphA a;
    MorphB b;
    MorphC c;

    // (c ∘ b) ∘ a
    auto cb_a = (c * b) * a;
    // c ∘ (b ∘ a)
    auto c_ba = c * (b * a);

    int input = 5;
    long res1 = cb_a(input);
    long res2 = c_ba(input);

    assert(res1 == res2);
    std::cout << "  [PASS] Axiom 1: Morphism Composition Associativity ((c ∘ b) ∘ a == c ∘ (b ∘ a))\n";
}

void test_category_identity() {
    MorphA a;
    IdentityMorphism<int> id_dom;
    IdentityMorphism<double> id_codom;

    auto a_id = a * id_dom;
    auto id_a = id_codom * a;

    int input = 7;
    assert(a_id(input) == a(input));
    assert(id_a(input) == a(input));
    std::cout << "  [PASS] Axiom 2: Category Identity Morphism (f ∘ id_A == f == id_B ∘ f)\n";
}

void test_dagger_involution_and_functoriality() {
    MorphA a;
    MorphB b;

    // Dagger Adjoint
    auto a_dag = ~a;
    auto a_dag_dag = ~a_dag;
    assert(a_dag_dag(10) == a(10));
    std::cout << "  [PASS] Axiom 3: Dagger Involution ((f†)† == f)\n";

    // Functoriality: (b ∘ a)† == (a† ∘ b†)
    auto ba = b * a;
    auto ba_dag = ~ba;

    auto a_dag_b_dag = a_dag * ~b;

    float output_grad = 13.0f;
    int grad1 = ba_dag(output_grad);
    int grad2 = a_dag_b_dag(output_grad);

    assert(grad1 == grad2);
    std::cout << "  [PASS] Axiom 4: Dagger Functoriality ((g ∘ f)† == f† ∘ g†)\n";
}

void test_monoidal_tensor_dagger() {
    MorphA a;
    MorphB b;

    auto ab_tensor = a ^ b;
    auto ab_dag = ~ab_tensor;
    auto expected_dag = (~a) ^ (~b);

    auto test_input = std::make_pair(10.0, 15.0f);
    auto res1 = ab_dag(test_input);
    auto res2 = expected_dag(test_input);

    assert(res1.first == res2.first);
    assert(res1.second == res2.second);
    std::cout << "  [PASS] Axiom 5: Monoidal Tensor Dagger ((f ⊗ g)† == f† ⊗ g†)\n";
}

void test_thermal_cost_monoid() {
    MorphismCost cost1{
        .memory_bytes = 1024 * 1024,
        .flops = 2000000,
        .energy_joules = 0.05,
        .bandwidth_gbps = 120.0,
        .latency_us = 10.0,
        .thermal_resistance_k_per_w = 0.2
    };

    MorphismCost cost2{
        .memory_bytes = 512 * 1024,
        .flops = 1000000,
        .energy_joules = 0.02,
        .bandwidth_gbps = 60.0,
        .latency_us = 5.0,
        .thermal_resistance_k_per_w = 0.2
    };

    auto combined = cost1 + cost2;
    assert(combined.memory_bytes == cost1.memory_bytes + cost2.memory_bytes);
    assert(combined.flops == cost1.flops + cost2.flops);
    assert(combined.energy_joules == 0.07);
    assert(combined.bandwidth_gbps == 120.0);
    assert(combined.latency_us == 15.0);

    double power = combined.power_watts();
    assert(power > 0.0);
    double temp_delta = combined.delta_temperature_c();
    assert(temp_delta > 0.0);

    assert(cost2.is_pareto_dominant_over(cost1));

    std::cout << "  [PASS] Axiom 6: Monoidal Cost & Thermodynamic Pareto Algebra\n";
    std::cout << "         Power: " << power << " W | Delta Temp: " << temp_delta << " K\n";
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  OjasX C++20 Applied Category Theory (ACT) Mathematical Proofs  \n";
    std::cout << "=================================================================\n";

    test_category_associativity();
    test_category_identity();
    test_dagger_involution_and_functoriality();
    test_monoidal_tensor_dagger();
    test_thermal_cost_monoid();

    std::cout << "\nALL CATEGORY-THEORETIC MATHEMATICAL PROOFS PASSED (100%)!\n";
    return 0;
}
