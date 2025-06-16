#include "tr_lundeby_calculator.hpp"
#include <string>

void TRLundebyCalculator::process(const std::vector<double>& input, std::vector<double>& output) {
    // Calculate T30 and store in output (assuming output is for Schroeder curve or similar)
    double t30;
    std::vector<double> schroeder_curve;
    double noise_db;
    std::tie(t30, schroeder_curve, noise_db) = calculate_t30(input, 44100.0, -30.0);
    output = schroeder_curve; // Store Schroeder curve in output
}

std::tuple<double, std::vector<double>, double> TRLundebyCalculator::calculate_t30(const std::vector<double>& signal, double fs, double max_noise_db) {
    // Normalize and square signal
    double max_value = *std::max_element(signal.begin(), signal.end(), [](double a, double b) { return std::abs(a) < std::abs(b); });
    if (max_value == 0.0) {
        throw std::runtime_error("Signal has zero maximum amplitude");
    }
    std::vector<double> normalized_signal = signal;
    for (double& value : normalized_signal) {
        value = (value / max_value) * (value / max_value); // Square for energy
    }

    // Trim signal from maximum onward
    size_t max_index = std::distance(normalized_signal.begin(),
        std::max_element(normalized_signal.begin(), normalized_signal.end(),
            [](double a, double b) { return std::abs(a) < std::abs(b); }));
    normalized_signal.erase(normalized_signal.begin(), normalized_signal.begin() + max_index);

    // Calculate Lundeby parameters
    auto lundeby_result = calculate_lundeby(normalized_signal, fs, 0.05, max_noise_db);

    // Calculate Schroeder curve
    std::vector<double> schroeder_curve = schroeder(normalized_signal, lundeby_result.cross_point, lundeby_result.C);
    double max_sch = *std::max_element(schroeder_curve.begin(), schroeder_curve.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    for (double& value : schroeder_curve) {
        value = 10.0 * std::log10(value / max_sch + std::numeric_limits<double>::epsilon());
    }

    // Trim Schroeder curve from maximum onward
    size_t sch_max_index = std::distance(schroeder_curve.begin(),
        std::max_element(schroeder_curve.begin(), schroeder_curve.end()));
    std::vector<double> trimmed_schroeder(schroeder_curve.begin() + sch_max_index, schroeder_curve.end());

    // Calculate T30
    std::vector<double> t30_range, db_range;
    double max_db = *std::max_element(trimmed_schroeder.begin(), trimmed_schroeder.end());
    for (size_t i = 0; i < trimmed_schroeder.size(); ++i) {
        if (trimmed_schroeder[i] <= max_db - 5.0 && trimmed_schroeder[i] > max_db - 35.0) {
            t30_range.push_back(static_cast<double>(i) / fs);
            db_range.push_back(trimmed_schroeder[i]);
        }
    }

    if (t30_range.empty()) {
        throw std::runtime_error("Could not find valid T30 range");
    }

    auto regression = least_squares(t30_range, db_range);
    double t30 = -60.0 / regression.slope;

    return std::make_tuple(t30, schroeder_curve, lundeby_result.noise_db);
}

TRLundebyCalculator::LeastSquaresResult TRLundebyCalculator::least_squares(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("x and y must have the same size");
    }
    if (x.empty()) {
        throw std::invalid_argument("Input vectors are empty");
    }

    size_t n = x.size();
    double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
    double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
    double sum_xy = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
    double sum_xx = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
    double denom = n * sum_xx - sum_x * sum_x;
    //if (std::abs(denom) < std::numeric_limits<double>::epsilon()) {
    //    throw std::runtime_error("Denominator is too close to zero, cannot compute slope");
    //}

    double slope = (n * sum_xy - sum_x * sum_y) / denom;
    double intercept = (sum_y - slope * sum_x) / n;

    std::vector<double> fitted_line(n);
    for (size_t i = 0; i < n; ++i) {
        fitted_line[i] = slope * x[i] + intercept;
    }

    return {slope, intercept, fitted_line};
}

std::vector<double> TRLundebyCalculator::schroeder(const std::vector<double>& signal, size_t t, double C) {
    std::vector<double> truncated_signal(signal.begin(), signal.begin() + std::min(t, signal.size()));
    std::vector<double> schroeder_curve(truncated_signal.size());
    double sum = 0.0;
    for (size_t i = truncated_signal.size(); i > 0; --i) {
        sum += truncated_signal[i - 1];
        schroeder_curve[i - 1] = sum;
    }
    double total_sum = std::accumulate(truncated_signal.begin(), truncated_signal.end(), 0.0);
    for (double& value : schroeder_curve) {
        value = (value + C) / (total_sum + C);
    }
    std::reverse(schroeder_curve.begin(), schroeder_curve.end());
    return schroeder_curve;
}

std::vector<double> TRLundebyCalculator::calculate_time_axis(size_t signal_length, double fs, double time_step) {
    size_t num_points = static_cast<size_t>(signal_length / (fs * time_step));
    std::vector<double> time_axis(num_points);
    size_t v = static_cast<size_t>(fs * time_step);
    for (size_t i = 0; i < num_points; ++i) {
        time_axis[i] = (std::ceil(static_cast<double>(v) / 2) + i * v) / fs;
    }
    return time_axis;
}

std::vector<double> TRLundebyCalculator::calculate_average_power(const std::vector<double>& signal, size_t window_size) {
    size_t num_windows = signal.size() / window_size;
    if (num_windows < 1) {
        throw std::runtime_error("Window size too large for signal length");
    }
    std::vector<double> average_power(num_windows);
    for (size_t i = 0; i < num_windows; ++i) {
        double sum = 0.0;
        size_t count = std::min(window_size, signal.size() - i * window_size);
        for (size_t j = 0; j < count; ++j) {
            sum += signal[i * window_size + j];
        }
        average_power[i] = sum / count;
    }
    return average_power;
}

TRLundebyCalculator::LundebyResult TRLundebyCalculator::calculate_lundeby(const std::vector<double>& signal, double fs, double time_step, double max_noise_db) {
    // Normalize and square signal (already done in calculate_t30, but ensure consistency)
    double max_value = *std::max_element(signal.begin(), signal.end());
    if (max_value == 0.0) {
        throw std::runtime_error("Signal has zero maximum amplitude");
    }
    std::vector<double> normalized_signal = signal;
    for (double& value : normalized_signal) {
        value /= max_value; // Already squared in calculate_t30
    }

    // Initial noise level from last 10% of signal
    size_t noise_start = static_cast<size_t>(normalized_signal.size() * 0.9);
    double noise_sum = std::accumulate(normalized_signal.begin() + noise_start, normalized_signal.end(), 0.0);
    double noise_db = 10.0 * std::log10(noise_sum / (normalized_signal.size() - noise_start) / max_value +
                                        std::numeric_limits<double>::epsilon());

    if (noise_db > max_noise_db) {
        throw NoiseError("Insufficient S/N ratio to perform Lundeby. Need at least " + std::to_string(max_noise_db) + " dB");
    }

    // Initial time axis and average power
    size_t window_size = static_cast<size_t>(fs * time_step);
    std::vector<double> time_axis = calculate_time_axis(normalized_signal.size(), fs, time_step);
    std::vector<double> average_power = calculate_average_power(normalized_signal, window_size);

    // Convert to dB
    std::vector<double> power_db(average_power.size());
    for (size_t i = 0; i < average_power.size(); ++i) {
        power_db[i] = 10.0 * std::log10(average_power[i] / max_value + std::numeric_limits<double>::epsilon());
    }

    // Initial regression (values 10 dB above noise)
    std::vector<double> regression_time, regression_db;
    for (size_t i = 0; i < power_db.size(); ++i) {
        if (power_db[i] > noise_db + 10.0) {
            regression_time.push_back(time_axis[i]);
            regression_db.push_back(power_db[i]);
        }
    }
    if (regression_time.empty()) {
        throw std::runtime_error("No values above noise + 10 dB");
    }

    auto result = least_squares(regression_time, regression_db);
    double slope = result.slope;
    double intercept = result.intercept;
    double cross_point = (noise_db - intercept) / slope;

    // Iterative process
    double error = 1.0;
    int max_iterations = 25;
    int iteration = 0;

    while (error > 0.0001 && iteration < max_iterations) {
        // New time intervals
        double delta = std::abs(10.0 / slope);
        window_size = static_cast<size_t>(delta / 10.0);
        if (window_size < 2) window_size = 2;

        // Recalculate average power and time axis
        size_t t = cross_point > normalized_signal.size() ? normalized_signal.size() : static_cast<size_t>(cross_point - delta);
        if (t < window_size) t = window_size;
        average_power = calculate_average_power(
            std::vector<double>(normalized_signal.begin(), normalized_signal.begin() + t), window_size);
        time_axis = calculate_time_axis(t, fs, static_cast<double>(window_size) / fs);

        // Convert to dB
        power_db.resize(average_power.size());
        for (size_t i = 0; i < average_power.size(); ++i) {
            power_db[i] = 10.0 * std::log10(average_power[i] / max_value + std::numeric_limits<double>::epsilon());
        }

        // New regression
        regression_time.clear();
        regression_db.clear();
        for (size_t i = 0; i < power_db.size(); ++i) {
            if (power_db[i] > noise_db + 10.0) {
                regression_time.push_back(time_axis[i]);
                regression_db.push_back(power_db[i]);
            }
        }
        if (regression_time.empty()) {
            throw std::runtime_error("No values above noise + 10 dB");
        }

        result = least_squares(regression_time, regression_db);
        slope = result.slope;
        intercept = result.intercept;

        // Update noise level
        noise_start = std::min(static_cast<size_t>(std::abs(cross_point + delta)), normalized_signal.size());
        if (normalized_signal.size() - noise_start < normalized_signal.size() / 10) {
            noise_start = static_cast<size_t>(normalized_signal.size() * 0.9);
        }
        noise_sum = std::accumulate(normalized_signal.begin() + noise_start, normalized_signal.end(), 0.0);
        noise_db = 10.0 * std::log10(noise_sum / (normalized_signal.size() - noise_start) / max_value +
                                     std::numeric_limits<double>::epsilon());

        // New cross point
        double new_cross_point = (noise_db - intercept) / slope;
        error = std::abs(cross_point - new_cross_point) / (std::abs(cross_point) + std::numeric_limits<double>::epsilon());
        cross_point = new_cross_point;
        iteration++;
    }

    // Cap cross_point
    if (cross_point > normalized_signal.size()) {
        cross_point = normalized_signal.size();
    }

    // Calculate C
    double C = max_value * std::pow(10.0, intercept / 10.0) *
               std::exp(slope / 10.0 / std::log10(std::exp(1.0)) * cross_point) /
               (-slope / 10.0 / std::log10(std::exp(1.0)));

    return {static_cast<size_t>(cross_point), C, noise_db};
}
