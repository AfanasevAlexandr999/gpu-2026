#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
inline float GeluScalar(float x) noexcept {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    constexpr float kGeluCubicCoeff = 0.044715f;

    const float x3 = x * x * x;
    const float tanh_arg = kSqrt2OverPi * (x + kGeluCubicCoeff * x3);
    return 0.5f * x * (1.0f + std::tanhf(tanh_arg));
}
}  // namespace

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t size = input.size();
    std::vector<float> result(size);

    if (size == 0) {
        return result;
    }

    const float* in = input.data();
    float* out = result.data();

#pragma omp parallel for simd schedule(static) default(none) shared(in, out, size)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(size); ++i) {
        out[i] = GeluScalar(in[i]);
    }

    return result;
}
