#pragma once
#include "audio_processor.hpp"
#include <vector>

/**
 * @file clarity_calculator.hpp
 * @brief Clarity metric computation (C<sub>x</sub>) for room impulse responses.
 *
 * The ::ClarityCalculator class exposes a single static function,
 * ::ClarityCalculator::calculate, which returns the clarity value (in dB)
 * for a provided impulse response and integration window. It derives from
 * ::AudioProcessor simply to share common type/utility conventions; no
 * instance state is used.
 *
 * Typical usages include:
 *
 * - C50  (speech intelligibility)  -> @c time_ms = 50
 * - C80  (music clarity)           -> @c time_ms = 80
 */
class ClarityCalculator : public AudioProcessor {
public:
    /**
     * @brief Compute clarity (C<sub>x</sub>) in dB for a given time window.
     *
     * Clarity is defined as the ratio of the early arriving energy (up to
     * @p time_ms milliseconds after the direct sound) to the late arriving
     * energy (everything thereafter), expressed in decibels:
     *
     * \f[
     *   C_x = 10 \log_{10} \frac{ \int_0^{t_x} p^2(t)\,dt }
     *                              { \int_{t_x}^{\infty} p^2(t)\,dt }.
     * \f]
     *
     * @param time_ms  Early/late boundary in milliseconds (e.g., 50 for C50).
     * @param signal   Impulse response samples (linear amplitude).
     * @param fs       Sampling rate in Hz.
     * @return Clarity value in dB.
     */
    static double calculate(double time_ms, const std::vector<double>& signal, double fs);
};