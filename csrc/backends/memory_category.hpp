#ifndef OJASX_BACKENDS_MEMORY_CATEGORY_HPP
#define OJASX_BACKENDS_MEMORY_CATEGORY_HPP

#include <iostream>
#include <cstddef>
#include "../semantics/semantic_tensor.hpp"

namespace ojasx::backends {

/// Memory Placement Category Object
template <std::size_t Bytes, bool IsInSRAM>
struct MemorySlot {
    static constexpr std::size_t size_bytes = Bytes;
    static constexpr bool is_sram = IsInSRAM;

    void print_memory_plan() const {
        std::cout << "  [Memory Category] Slot allocated: " << size_bytes
                  << " bytes (" << (IsInSRAM ? "GPU L1/L2 SRAM" : "VRAM Buffer") << ")\n";
    }
};

/// Functor F_mem: Maps Tensor Objects into Memory Category Placement Slots
struct MemoryFunctor {
    template <typename SemanticTensorObj>
    constexpr auto map_object(const SemanticTensorObj&) const noexcept {
        constexpr std::size_t bytes = SemanticTensorObj::size * sizeof(float);
        constexpr bool fits_in_sram = (bytes <= 64 * 1024); // 64KB SRAM limit
        return MemorySlot<bytes, fits_in_sram>{};
    }
};

} // namespace ojasx::backends

#endif // OJASX_BACKENDS_MEMORY_CATEGORY_HPP
