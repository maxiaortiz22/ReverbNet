#include "tr_augmentation_calculator.hpp"
#include "tr_lundeby_calculator.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <string>

constexpr double PI = 3.14159265358979323846;

std::vector<double> TRAugmentationCalculator::normalize_rir(const std::vector<double>& rir) {
    size_t max_idx = std::max_element(rir.begin(), rir.end(), 
        [](double a, double b) { return std::abs(a) < std::abs(b); }) - rir.begin();
    
    std::vector<double> normalized = rir;
    double max_value = std::abs(rir[max_idx]);
    for (double& value : normalized) {
        value /= max_value;
    }
    return normalized;
}

TRAugmentationCalculator::TemporalDecomposition TRAugmentationCalculator::temporal_decompose(
    const std::vector<double>& rir, double fs, double tau) {
    
    size_t t_d = std::max_element(rir.begin(), rir.end(), 
        [](double a, double b) { return std::abs(a) < std::abs(b); }) - rir.begin();
    size_t t_o = static_cast<size_t>(tau * fs);
    
    size_t init_idx = (t_d > t_o) ? t_d - t_o : 0;
    size_t final_idx = std::min(t_d + t_o + 1, rir.size() - 1);
    
    TemporalDecomposition result;
    result.delay = std::vector<double>(rir.begin(), rir.begin() + init_idx);
    result.early = std::vector<double>(rir.begin() + init_idx, rir.begin() + final_idx);
    result.late = std::vector<double>(rir.begin() + final_idx, rir.end());
    
    return result;
}

std::vector<double> TRAugmentationCalculator::get_envelope(const std::vector<double>& signal, size_t window_length) {
    std::vector<double> envelope(signal.size());
    size_t half_window = window_length / 2;
    
    for (size_t i = 0; i < signal.size(); ++i) {
        size_t start = (i > half_window) ? i - half_window : 0;
        size_t end = std::min(i + half_window + 1, signal.size());
        double max_value = 0.0;
        
        for (size_t j = start; j < end; ++j) {
            max_value = std::max(max_value, std::abs(signal[j]));
        }
        envelope[i] = max_value;
    }
    
    return envelope;
}

std::vector<double> TRAugmentationCalculator::create_hann_window(size_t length) {
    std::vector<double> window(length);
    for (size_t i = 0; i < length; ++i) {
        window[i] = 0.5 * (1.0 - std::cos(2.0 * PI * i / (length - 1)));
    }
    return window;
}

std::vector<double> TRAugmentationCalculator::apply_window(const std::vector<double>& signal, 
                                                         const std::vector<double>& window) {
    std::vector<double> result(signal.size());
    size_t half_window = window.size() / 2;
    
    for (size_t i = 0; i < signal.size(); ++i) {
        result[i] = signal[i] * window[std::min(i, window.size() - 1)];
    }
    
    return result;
}

std::vector<double> TRAugmentationCalculator::cross_fade(const std::vector<double>& signal1, 
                                                       const std::vector<double>& signal2,
                                                       double fs, size_t cross_point) {
    size_t window_length = static_cast<size_t>(50.0 * 0.001 * fs); // 50ms window
    if (2 * window_length > signal1.size() - cross_point) {
        return signal1;
    }
    
    std::vector<double> window = create_hann_window(window_length);
    std::vector<double> fade_out(window.begin() + window.size()/2, window.end());
    std::vector<double> fade_in(window.begin(), window.begin() + window.size()/2);
    
    std::vector<double> result(signal1.size());
    
    // Apply fade out to signal1
    for (size_t i = 0; i < signal1.size(); ++i) {
        if (i < cross_point - fade_out.size()/2) {
            result[i] = signal1[i];
        } else if (i < cross_point + fade_out.size()/2) {
            size_t window_idx = i - (cross_point - fade_out.size()/2);
            result[i] = signal1[i] * fade_out[window_idx];
        } else {
            result[i] = 0.0;
        }
    }
    
    // Apply fade in to signal2
    for (size_t i = 0; i < signal2.size(); ++i) {
        if (i < cross_point - fade_in.size()/2) {
            result[i] += 0.0;
        } else if (i < cross_point + fade_in.size()/2) {
            size_t window_idx = i - (cross_point - fade_in.size()/2);
            result[i] += signal2[i] * fade_in[window_idx];
        } else {
            result[i] += signal2[i];
        }
    }
    
    return result;
}

double TRAugmentationCalculator::estimate_fullband_decay(const std::vector<double>& rir, double fs) {
    auto decomposition = temporal_decompose(rir, fs);
    auto late_env = get_envelope(decomposition.late, static_cast<size_t>(0.04 * fs));
    
    std::vector<double> t(late_env.size());
    for (size_t i = 0; i < t.size(); ++i) {
        t[i] = static_cast<double>(i) / fs;
    }
    
    // Simple linear regression on log of envelope
    std::vector<double> log_env(late_env.size());
    for (size_t i = 0; i < late_env.size(); ++i) {
        log_env[i] = std::log(late_env[i] + std::numeric_limits<double>::epsilon());
    }
    
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
    for (size_t i = 0; i < t.size(); ++i) {
        sum_x += t[i];
        sum_y += log_env[i];
        sum_xy += t[i] * log_env[i];
        sum_xx += t[i] * t[i];
    }
    
    double slope = (t.size() * sum_xy - sum_x * sum_y) / (t.size() * sum_xx - sum_x * sum_x);
    return -1.0 / slope;
}

TRAugmentationCalculator::EnvelopeParameters TRAugmentationCalculator::estimate_parameters(
    const std::vector<double>& late, size_t cross_point, double fs) {
    
    auto late_env = get_envelope(late, static_cast<size_t>(0.5 * fs));
    std::vector<double> t(late_env.size());
    for (size_t i = 0; i < t.size(); ++i) {
        t[i] = static_cast<double>(i) / fs;
    }
    
    // Simple exponential fit
    std::vector<double> log_env(late_env.size());
    for (size_t i = 0; i < late_env.size(); ++i) {
        log_env[i] = std::log(late_env[i] + std::numeric_limits<double>::epsilon());
    }
    
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
    for (size_t i = 0; i < cross_point; ++i) {
        sum_x += t[i];
        sum_y += log_env[i];
        sum_xy += t[i] * log_env[i];
        sum_xx += t[i] * t[i];
    }
    
    double slope = (cross_point * sum_xy - sum_x * sum_y) / (cross_point * sum_xx - sum_x * sum_x);
    double intercept = (sum_y - slope * sum_x) / cross_point;
    
    EnvelopeParameters params;
    params.amplitude = std::exp(intercept);
    params.decay_rate = -1.0 / slope;
    params.noise_floor = late_env.back();
    
    return params;
}

std::vector<double> TRAugmentationCalculator::apply_augmentation(
    const std::vector<double>& rir,
    const EnvelopeParameters& params,
    double fullband_decay,
    double target_tr,
    double fs) {
    
    std::vector<double> t(rir.size());
    for (size_t i = 0; i < t.size(); ++i) {
        t[i] = static_cast<double>(i) / fs;
    }
    
    double decay_rate_d = target_tr / std::log(1000.0);
    double ratio = decay_rate_d / fullband_decay;
    double t_md = ratio * params.decay_rate;
    
    std::vector<double> augmented(rir.size());
    for (size_t i = 0; i < rir.size(); ++i) {
        augmented[i] = rir[i] * std::exp(-t[i] * ((params.decay_rate - t_md) / (params.decay_rate * t_md)));
    }
    
    return augmented;
}

std::vector<double> TRAugmentationCalculator::augment_tr(const std::vector<double>& rir, double fs, double target_tr) {
    try {
        auto normalized = normalize_rir(rir);
        auto decomposition = temporal_decompose(normalized, fs);
        
        double fullband_decay = estimate_fullband_decay(decomposition.late, fs);
        
        // Calculate T30 and get the cross point
        auto [t30, _, __] = TRLundebyCalculator::calculate_t30(decomposition.late, fs, -60.0);
        size_t cross_point = static_cast<size_t>(t30 * fs);
        auto parameters = estimate_parameters(decomposition.late, cross_point, fs);
        
        // Create noiseless version
        std::vector<double> t(decomposition.late.size());
        for (size_t i = 0; i < t.size(); ++i) {
            t[i] = static_cast<double>(i) / fs;
        }
        
        std::vector<double> noiseless(decomposition.late.size());
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, 1.0);
        
        for (size_t i = 0; i < noiseless.size(); ++i) {
            noiseless[i] = parameters.amplitude * std::exp(-t[i] / parameters.decay_rate) * dist(gen);
        }
        
        auto denoised = cross_fade(decomposition.late, noiseless, fs, cross_point);
        auto augmented = apply_augmentation(denoised, parameters, fullband_decay, target_tr, fs);
        
        // Combine all parts
        std::vector<double> result;
        result.insert(result.end(), decomposition.delay.begin(), decomposition.delay.end());
        result.insert(result.end(), decomposition.early.begin(), decomposition.early.end());
        result.insert(result.end(), augmented.begin(), augmented.end());
        
        return result;
    } catch (const std::exception& e) {
        throw TRAugmentationError("Failed to augment TR to " + std::to_string(target_tr) + " s");
    }
} 