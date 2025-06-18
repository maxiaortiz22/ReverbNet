#pragma once
#include <vector>

class AudioProcessor {
public:
    // Constructor
    AudioProcessor() = default;
    virtual ~AudioProcessor() = default;

    // Common utility functions
    static double rms(const std::vector<double>& signal);
    static double snr(double signal_rms, double noise_rms);
    static double rms_comp(double signal_rms, double noise_rms, double snr_required);
}; 