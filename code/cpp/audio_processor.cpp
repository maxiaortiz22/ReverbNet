/**
 * @file audio_processor.cpp
 * @brief Implementation of basic audio measurement utilities (RMS, SNR, noise scaling).
 */

 #include "audio_processor.hpp"
 #include <cmath>
 
 /**
  * @brief Compute the root-mean-square (RMS) level of a signal.
  *
  * The RMS is defined as @f$ \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x_n^2} @f$ where
  * @f$ x_n @f$ are the linear-amplitude samples in @p signal.
  *
  * @param signal Audio samples (linear amplitude).
  * @return RMS value (linear amplitude units).
  */
 double AudioProcessor::rms(const std::vector<double>& signal) {
     double sum = 0.0;
     for (const auto& sample : signal) {
         sum += sample * sample;
     }
     return std::sqrt(sum / signal.size());
 }
 
 /**
  * @brief Compute signal-to-noise ratio (SNR) in decibels.
  *
  * Implemented as
  * @f[
  *   \mathrm{SNR}_{dB} = 10 \log_{10} \frac{\mathrm{signal\_rms}^2}{\mathrm{noise\_rms}^2}
  *                     = 20 \log_{10} \frac{\mathrm{signal\_rms}}{\mathrm{noise\_rms}}.
  * @f]
  *
  * @param signal_rms RMS of the signal (linear units).
  * @param noise_rms  RMS of the noise (linear units).
  * @return SNR in dB.
  */
 double AudioProcessor::snr(double signal_rms, double noise_rms) {
     return 10.0 * std::log10((signal_rms * signal_rms) / (noise_rms * noise_rms));
 }
 
 /**
  * @brief Compute scaling needed for noise to achieve a target SNR.
  *
  * Given current @p signal_rms and @p noise_rms, along with a desired
  * @p snr_required (dB), this returns the multiplicative gain to apply to the
  * noise signal:
  *
  * @f[
  *   g = \frac{\mathrm{rms\_required}}{\mathrm{noise\_rms}}, \qquad
  *   \mathrm{rms\_required} =
  *     \sqrt{ \frac{\mathrm{signal\_rms}^2}{10^{\mathrm{snr\_required}/10}} }.
  * @f]
  *
  * Multiplying the noise samples by @f$ g @f$ yields the requested SNR.
  *
  * @param signal_rms    RMS of the signal (linear units).
  * @param noise_rms     RMS of the noise (linear units).
  * @param snr_required  Target SNR in dB.
  * @return Gain factor to apply to the noise signal.
  */
 double AudioProcessor::rms_comp(double signal_rms, double noise_rms, double snr_required) {
     double rms_required = std::sqrt((signal_rms * signal_rms) / std::pow(10.0, snr_required / 10.0));
     return rms_required / noise_rms;
 }