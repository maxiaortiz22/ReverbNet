#pragma once

#include <vector>

/**
 * @file octave_filter_bank.hpp
 * @brief Multi‑band octave filter bank (C++ core for Python bindings).
 *
 * Provides a bank of cascaded biquad IIR filters implementing octave‑band
 * analysis at the nominal center frequencies:
 * 125, 250, 500, 1000, 2000, 4000, and 8000 Hz.
 *
 * The bank can be constructed at 2nd‑order or 4th‑order per band (implemented
 * as one or more biquad sections in cascade). Coefficient initialization is
 * handled internally; see @c initializeCoefficients().
 *
 * Usage (C++):
 * @code
 *   OctaveFilterBank bank(4);                 // 4th‑order filters
 *   std::vector<float> mono;                  // fill with audio samples
 *   auto bands = bank.process(mono);          // bands[b][n] -> double sample
 * @endcode
 *
 * Usage (Python via pybind11 extension):
 * @code{.py}
 *   from audio_processing import OctaveFilterBank
 *   fb = OctaveFilterBank(filter_order=4)
 *   y_bands = fb.process(audio_float32_np)    # ndarray [num_bands, num_samples]
 * @endcode
 */

// Filter‑bank constants -------------------------------------------------------

/// Number of octave bands: 125, 250, 500, 1000, 2000, 4000, 8000 Hz.
constexpr int NUM_BANDS = 7; // Octave bands

class OctaveFilterBank {
public:
    /**
     * @brief Construct the octave filter bank.
     *
     * @param filter_order Filter order per band (2 or 4). Default: 4.
     */
    explicit OctaveFilterBank(int filter_order = 4);

    /**
     * @brief Reset all internal filter states (delay lines) to zero.
     */
    void reset();

    /**
     * @brief Process a mono audio signal through all octave‑band filters.
     *
     * @param input Vector of input audio samples (mono, float).
     * @return Vector of vectors (double) where each inner vector contains the
     *         band‑filtered signal. Band order follows the nominal center
     *         frequencies returned by getCenterFrequencies().
     */
    std::vector<std::vector<double>> process(const std::vector<float>& input);

    /**
     * @brief Get the number of bands in the filter bank.
     * @return Number of octave bands.
     */
    static int getNumBands() { return NUM_BANDS; }

    /**
     * @brief Get the nominal center frequencies (Hz) for each band.
     * @return Constant reference to the frequency vector (length NUM_BANDS).
     */
    static const std::vector<double>& getCenterFrequencies() { return CENTER_FREQUENCIES; }

private:
    /**
     * @brief Single 2nd‑order (biquad) IIR filter section.
     *
     * All sections are stored in direct form with explicit state @c z[]
     * (two delay elements). @c a[0] is always unity.
     */
    struct BiquadSection {
        double b[3]; ///< Numerator coefficients (b0, b1, b2).
        double a[3]; ///< Denominator coefficients (a0==1, a1, a2).
        double z[2]; ///< Delay‑line state.
    };

    /**
     * @brief Complete cascaded filter for one octave band.
     *
     * Each band owns @c num_sections biquads stored in @c sections, along with
     * its nominal center frequency (Hz).
     */
    struct FilterBand {
        std::vector<BiquadSection> sections; ///< Cascaded biquad sections.
        double center_freq;                  ///< Nominal center frequency (Hz).
        int num_sections;                    ///< Number of active sections.
    };

    // --- Private methods -----------------------------------------------------

    /// Initialize all filter coefficients for the selected @c filter_order_.
    void initializeCoefficients();

    /// Process one sample through a single @ref BiquadSection (in‑place state update).
    double processBiquad(BiquadSection& section, double input);

    /// Process one sample through an entire @ref FilterBand cascade.
    double processFilterBand(FilterBand& band, double input);

    // --- Data members --------------------------------------------------------

    int filter_order_;              ///< Configured filter order (2 or 4).
    bool initialized_;              ///< Coefficient/state initialization flag.
    std::vector<FilterBand> bands_; ///< All octave‑band filters.

    static const std::vector<double> CENTER_FREQUENCIES; ///< Nominal band centers (Hz).
};