#include "clarity_calculator.hpp"
#include <cmath>
#include <limits>

double ClarityCalculator::calculate(double time_ms, const std::vector<double>& signal, double fs) {
    std::vector<double> h2(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) {
        h2[i] = signal[i] * signal[i];
    }

    size_t t = static_cast<size_t>((time_ms / 1000.0) * fs + 1);
    
    double sum_before = 0.0;
    double sum_after = 0.0;
    
    for (size_t i = 0; i < t; ++i) {
        sum_before += h2[i];
    }
    
    for (size_t i = t; i < h2.size(); ++i) {
        sum_after += h2[i];
    }

    return 10.0 * std::log10((sum_before / (sum_after + std::numeric_limits<double>::epsilon())));
} 