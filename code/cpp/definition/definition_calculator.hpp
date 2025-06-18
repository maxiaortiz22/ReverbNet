#pragma once
#include "audio_processor.hpp"
#include <vector>

class DefinitionCalculator : public AudioProcessor {
public:
    static double calculate(const std::vector<double>& signal, double fs);
}; 