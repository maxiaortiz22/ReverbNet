#include "audio_processor.hpp"
#include <cmath>

double AudioProcessor::rms(const std::vector<double>& signal) {
    double sum = 0.0;
    for (const auto& sample : signal) {
        sum += sample * sample;
    }
    return std::sqrt(sum / signal.size());
}

double AudioProcessor::snr(double signal_rms, double noise_rms) {
    return 10.0 * std::log10((signal_rms * signal_rms) / (noise_rms * noise_rms));
}

double AudioProcessor::rms_comp(double signal_rms, double noise_rms, double snr_required) {
    double rms_required = std::sqrt((signal_rms * signal_rms) / std::pow(10.0, snr_required / 10.0));
    return rms_required / noise_rms;
} 