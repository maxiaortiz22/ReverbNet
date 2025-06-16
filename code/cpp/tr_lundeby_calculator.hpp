#pragma once
#include "audio_processor.hpp"
#include <vector>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

// Custom exception for noise-related errors
class NoiseError : public std::runtime_error {
public:
    NoiseError(const std::string& message) : std::runtime_error("NoiseError: " + message) {}
};

class TRLundebyCalculator : public AudioProcessor {
public:
    // Structs for return types
    struct LeastSquaresResult {
        double slope;
        double intercept;
        std::vector<double> fitted_line;
    };

    struct LundebyResult {
        size_t cross_point;
        double C;
        double noise_db;
    };

    // Method declarations
    void process(const std::vector<double>& input, std::vector<double>& output);
    static std::tuple<double, std::vector<double>, double> calculate_t30(const std::vector<double>& signal, double fs, double max_noise_db);

private:
    // Private method declarations - all made static
    static LeastSquaresResult least_squares(const std::vector<double>& x, const std::vector<double>& y);
    static std::vector<double> schroeder(const std::vector<double>& signal, size_t t, double C);
    static std::vector<double> calculate_time_axis(size_t signal_length, double fs, double time_step);
    static std::vector<double> calculate_average_power(const std::vector<double>& signal, size_t window_size);
    static LundebyResult calculate_lundeby(const std::vector<double>& signal, double fs, double time_step, double max_noise_db);
};