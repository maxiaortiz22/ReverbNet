#pragma once
#include <vector>

/**
 * @file audio_processor.hpp
 * @brief Lightweight audio analysis utilities (RMS, SNR, RMS compensation).
 *
 * This header declares the ::AudioProcessor class, a thin static-utility
 * container providing a few common scalar measurements used throughout the
 * Python/C++ audio-processing bridge:
 *
 * - ::AudioProcessor::rms : Root-mean-square level of a signal.
 * - ::AudioProcessor::snr : Signal-to-noise ratio in decibels given signal/noise RMS.
 * - ::AudioProcessor::rms_comp : Multiplicative factor to apply to noise so that the
 *   resulting SNR matches a desired target (in dB).
 *
 * All functions operate on linear-amplitude samples (not dB) unless otherwise noted.
 */
class AudioProcessor {
public:
    /// Default constructor.
    AudioProcessor() = default;
    virtual ~AudioProcessor() = default;

    /**
     * @brief Compute the root-mean-square (RMS) of a signal.
     *
     * @param signal Audio samples in linear amplitude.
     * @return RMS value (linear amplitude units).
     */
    static double rms(const std::vector<double>& signal);

    /**
     * @brief Compute signal-to-noise ratio (SNR) in decibels.
     *
     * @param signal_rms RMS of the signal (linear units).
     * @param noise_rms  RMS of the noise (linear units).
     * @return SNR in dB ( 20 * log10(signal_rms / noise_rms) ).
     */
    static double snr(double signal_rms, double noise_rms);

    /**
     * @brief Compute a scaling factor to apply to noise to reach a target SNR.
     *
     * Given the current @p signal_rms and @p noise_rms (linear units) and a desired
     * SNR in dB, this function returns the multiplicative gain that should be
     * applied to the noise signal so that the resulting SNR equals @p snr_required.
     *
     * @param signal_rms    RMS of the signal (linear units).
     * @param noise_rms     RMS of the noise (linear units).
     * @param snr_required  Target SNR in dB.
     * @return Multiplicative gain to apply to the noise signal.
     */
    static double rms_comp(double signal_rms, double noise_rms, double snr_required);
};