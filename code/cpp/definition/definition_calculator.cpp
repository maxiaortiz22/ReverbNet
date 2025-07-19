/**
 * @file definition_calculator.cpp
 * @brief Implementation of the Speech Definition (D50) metric.
 *
 * Definition (D50) is the ratio of the **early** energy (first 50 ms after the
 * direct sound) to the **total** energy in a room impulse response. It is often
 * expressed as a percentage and used as an indicator of speech intelligibility.
 *
 * See definition_calculator.hpp for API details.
 */

 #include "definition_calculator.hpp"
 #include <cmath>
 
 double DefinitionCalculator::calculate(const std::vector<double>& signal, double fs) {
     const double time_ms = 50.0; // Early/late boundary: 50 ms integration window.
 
     // Square the IR samples to obtain energy.
     std::vector<double> h2(signal.size());
     for (size_t i = 0; i < signal.size(); ++i) {
         h2[i] = signal[i] * signal[i];
     }
 
     // Convert 50 ms boundary to a sample index (+1 to match original behavior).
     size_t t = static_cast<size_t>((time_ms / 1000.0) * fs + 1);
     
     double sum_before = 0.0; // Early energy (0 .. t-1).
     double sum_total  = 0.0; // Total energy (0 .. end).
     
     for (size_t i = 0; i < t; ++i) {
         sum_before += h2[i];
     }
     
     for (size_t i = 0; i < h2.size(); ++i) {
         sum_total += h2[i];
     }
 
     // Definition returned as a percentage of total energy.
     return 100.0 * sum_before / sum_total;
 }