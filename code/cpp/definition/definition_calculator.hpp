#pragma once
#include "audio_processor.hpp"
#include <vector>

/**
 * @file definition_calculator.hpp
 * @brief Speech Definition (D50) metric for room impulse responses.
 *
 * The ::DefinitionCalculator class exposes a single static function,
 * ::DefinitionCalculator::calculate, which evaluates the **Definition**
 * (often denoted *D50*) of an impulse response.  Definition is the ratio of
 * the early arriving energy (within the first ~50 ms after the direct sound)
 * to the total energy, and is widely used as an intelligibility indicator.
 *
 * The exact integration window and scaling follow the implementation in
 * definition_calculator.cpp; no instance state is required.
 */
class DefinitionCalculator : public AudioProcessor {
public:
    /**
     * @brief Compute the Definition (D50) value for an impulse response.
     *
     * @param signal Impulse-response samples (linear amplitude).
     * @param fs     Sampling rate in Hz.
     * @return Definition metric value (range depends on implementation; typically 0–1 or %).
     */
    static double calculate(const std::vector<double>& signal, double fs);
};