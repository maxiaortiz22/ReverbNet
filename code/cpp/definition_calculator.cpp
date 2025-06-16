#include "definition_calculator.hpp"
#include <cmath>

double DefinitionCalculator::calculate(const std::vector<double>& signal, double fs) {
    const double time_ms = 50.0; // 50ms of integration
    
    std::vector<double> h2(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) {
        h2[i] = signal[i] * signal[i];
    }

    size_t t = static_cast<size_t>((time_ms / 1000.0) * fs + 1);
    
    double sum_before = 0.0;
    double sum_total = 0.0;
    
    for (size_t i = 0; i < t; ++i) {
        sum_before += h2[i];
    }
    
    for (size_t i = 0; i < h2.size(); ++i) {
        sum_total += h2[i];
    }

    return 100.0 * sum_before / sum_total;
} 