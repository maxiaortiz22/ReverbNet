#pragma once
#include "audio_processor.hpp"
#include <vector>

class ClarityCalculator : public AudioProcessor {
public:
    static double calculate(double time_ms, const std::vector<double>& signal, double fs);
}; 