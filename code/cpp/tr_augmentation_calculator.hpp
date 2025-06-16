#pragma once

#include "audio_processor.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>

class TRAugmentationError : public std::runtime_error {
public:
    explicit TRAugmentationError(const std::string& message) : std::runtime_error(message) {}
};

class TRAugmentationCalculator : public AudioProcessor {
public:
    struct TemporalDecomposition {
        std::vector<double> delay;
        std::vector<double> early;
        std::vector<double> late;
    };

    struct EnvelopeParameters {
        double amplitude;
        double decay_rate;
        double noise_floor;
    };

    static std::vector<double> augment_tr(const std::vector<double>& rir, double fs, double target_tr);
    static std::vector<double> normalize_rir(const std::vector<double>& rir);
    static TemporalDecomposition temporal_decompose(const std::vector<double>& rir, double fs, double tau = 0.0025);
    static std::vector<double> get_envelope(const std::vector<double>& signal, size_t window_length);
    static EnvelopeParameters estimate_parameters(const std::vector<double>& late, size_t cross_point, double fs);
    static std::vector<double> cross_fade(const std::vector<double>& signal1, const std::vector<double>& signal2, 
                                        double fs, size_t cross_point);
    static std::vector<double> apply_augmentation(const std::vector<double>& rir, 
                                                const EnvelopeParameters& params,
                                                double fullband_decay,
                                                double target_tr,
                                                double fs);

private:
    static std::vector<double> apply_window(const std::vector<double>& signal, const std::vector<double>& window);
    static std::vector<double> create_hann_window(size_t length);
    static double estimate_fullband_decay(const std::vector<double>& rir, double fs);
}; 