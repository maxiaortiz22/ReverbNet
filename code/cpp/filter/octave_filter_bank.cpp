#include "octave_filter_bank.hpp"
#include <stdexcept>

// Definición del miembro estático
const std::vector<double> OctaveFilterBank::CENTER_FREQUENCIES = {
    125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0
};

OctaveFilterBank::OctaveFilterBank(int filter_order)
    : filter_order_((filter_order == 2 || filter_order == 4) ? filter_order : 4),
      initialized_(false),
      bands_(NUM_BANDS)
{
    initializeCoefficients();
    reset();
}

void OctaveFilterBank::reset() {
    for (auto& band : bands_) {
        for (auto& section : band.sections) {
            section.z[0] = 0.0;
            section.z[1] = 0.0;
        }
    }
    initialized_ = true;
}

double OctaveFilterBank::processBiquad(BiquadSection& section, double input) {
    // Direct Form I: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
    double output = section.b[0] * input + section.z[0];
    section.z[0] = section.b[1] * input - section.a[1] * output + section.z[1];
    section.z[1] = section.b[2] * input - section.a[2] * output;
    return output;
}

double OctaveFilterBank::processFilterBand(FilterBand& band, double input) {
    double output = input;
    for (int i = 0; i < band.num_sections; ++i) {
        output = processBiquad(band.sections[i], output);
    }
    return output;
}

std::vector<std::vector<double>> OctaveFilterBank::process(const std::vector<float>& input) {
    if (!initialized_) {
        throw std::runtime_error("El banco de filtros no ha sido inicializado.");
    }

    if (input.empty()) {
        return std::vector<std::vector<double>>(NUM_BANDS);
    }

    // Prepara el contenedor de salida
    std::vector<std::vector<double>> output(NUM_BANDS, std::vector<double>(input.size()));

    // Procesa cada muestra a través de cada banda de forma independiente
    for (size_t i = 0; i < input.size(); ++i) {
        double sample = static_cast<double>(input[i]);
        for (int band_idx = 0; band_idx < NUM_BANDS; ++band_idx) {
            output[band_idx][i] = processFilterBand(bands_[band_idx], sample);
        }
    }

    return output;
}

void OctaveFilterBank::initializeCoefficients() {
    // Inicializa las bandas con sus frecuencias centrales y tamaño de secciones
    for (int i = 0; i < NUM_BANDS; ++i) {
        bands_[i].center_freq = CENTER_FREQUENCIES[i];
    }

    if (filter_order_ == 2) {
        // --- COEFICIENTES DE ORDEN 2 ---
        // Banda 0: 125.0 Hz
        bands_[0].num_sections = 2;
        bands_[0].sections.resize(2);
        bands_[0].sections[0] = {{2.939529427632e-04, 5.879058855264e-04, 2.939529427632e-04}, {1.0, -1.965851519013e+00, 9.698017076012e-01}, {0.0, 0.0}};
        bands_[0].sections[1] = {{1.0, -2.0, 1.0}, {1.0, -1.980311061135e+00, 9.817449495270e-01}, {0.0, 0.0}};

        // Banda 1: 250.0 Hz
        bands_[1].num_sections = 2;
        bands_[1].sections.resize(2);
        bands_[1].sections[0] = {{1.147995107502e-03, 2.295990215004e-03, 1.147995107502e-03}, {1.0, -1.924984562964e+00, 9.405383273792e-01}, {0.0, 0.0}};
        bands_[1].sections[1] = {{1.0, -2.0, 1.0}, {1.0, -1.958117714705e+00, 9.637996533768e-01}, {0.0, 0.0}};

        // Banda 2: 500.0 Hz
        bands_[2].num_sections = 2;
        bands_[2].sections.resize(2);
        bands_[2].sections[0] = {{4.382533153825e-03, 8.765066307651e-03, 4.382533153825e-03}, {1.0, -1.824549878047e+00, 8.847871871731e-01}, {0.0, 0.0}};
        bands_[2].sections[1] = {{1.0, -2.0, 1.0}, {1.0, -1.906432315295e+00, 9.287278067236e-01}, {0.0, 0.0}};

        // Banda 3: 1000.0 Hz
        bands_[3].num_sections = 2;
        bands_[3].sections.resize(2);
        bands_[3].sections[0] = {{1.604326536783e-02, 3.208653073566e-02, 1.604326536783e-02}, {1.0, -1.558936806743e+00, 7.841434241185e-01}, {0.0, 0.0}};
        bands_[3].sections[1] = {{1.0, -2.0, 1.0}, {1.0, -1.775508272613e+00, 8.611588446543e-01}, {0.0, 0.0}};

        // Banda 4: 2000.0 Hz
        bands_[4].num_sections = 2;
        bands_[4].sections.resize(2);
        bands_[4].sections[0] = {{5.473750305044e-02, 1.094750061009e-01, 5.473750305044e-02}, {1.0, -8.458879804356e-01, 6.246164232076e-01}, {0.0, 0.0}};
        bands_[4].sections[1] = {{1.0, -2.0, 1.0}, {1.0, -1.418769754759e+00, 7.314143254951e-01}, {0.0, 0.0}};

        // Banda 5: 4000.0 Hz
        bands_[5].num_sections = 2;
        bands_[5].sections.resize(2);
        bands_[5].sections[0] = {{1.702636093717e-01, -3.405272187435e-01, 1.702636093717e-01}, {1.0, -4.920012496670e-01, 4.551493425821e-01}, {0.0, 0.0}};
        bands_[5].sections[1] = {{1.0, 2.0, 1.0}, {1.0, 7.768050328828e-01, 4.928142946349e-01}, {0.0, 0.0}};
        
        // Banda 6: 8000.0 Hz (Highpass)
        bands_[6].num_sections = 1;
        bands_[6].sections.resize(1);
        bands_[6].sections[0] = {{1.261647334452e-01, -2.523294668904e-01, 1.261647334452e-01}, {1.0, 7.752263405204e-01, 2.798852743012e-01}, {0.0, 0.0}};

    } else { // filter_order_ == 4
        // --- COEFICIENTES DE ORDEN 4 ---
        // Banda 0: 125.0 Hz
        bands_[0].num_sections = 4;
        bands_[0].sections.resize(4);
        bands_[0].sections[0] = {{8.673158234242e-08, 1.734631646848e-07, 8.673158234242e-08}, {1.0, -1.960906450670e+00, 9.640525073830e-01}, {0.0, 0.0}};
        bands_[0].sections[1] = {{1.0, 2.0, 1.0}, {1.0, -1.971053025488e+00, 9.728402926431e-01}, {0.0, 0.0}};
        bands_[0].sections[2] = {{1.0, -2.0, 1.0}, {1.0, -1.978156223745e+00, 9.827183453362e-01}, {0.0, 0.0}};
        bands_[0].sections[3] = {{1.0, -2.0, 1.0}, {1.0, -1.989656792083e+00, 9.909121429471e-01}, {0.0, 0.0}};
        
        // Banda 1: 250.0 Hz
        bands_[1].num_sections = 4;
        bands_[1].sections.resize(4);
        bands_[1].sections[0] = {{1.327751925747e-06, 2.655503851494e-06, 1.327751925747e-06}, {1.0, -1.917025766153e+00, 9.293820432293e-01}, {0.0, 0.0}};
        bands_[1].sections[1] = {{1.0, 2.0, 1.0}, {1.0, -1.939330754665e+00, 9.463819662794e-01}, {0.0, 0.0}};
        bands_[1].sections[2] = {{1.0, -2.0, 1.0}, {1.0, -1.947695289975e+00, 9.657681300215e-01}, {0.0, 0.0}};
        bands_[1].sections[3] = {{1.0, -2.0, 1.0}, {1.0, -1.976898619380e+00, 9.818957786421e-01}, {0.0, 0.0}};

        // Banda 2: 500.0 Hz
        bands_[2].num_sections = 4;
        bands_[2].sections.resize(4);
        bands_[2].sections[0] = {{1.949290469893e-05, 3.898580939786e-05, 1.949290469893e-05}, {1.0, -1.815987542309e+00, 8.636387650892e-01}, {0.0, 0.0}};
        bands_[2].sections[1] = {{1.0, 2.0, 1.0}, {1.0, -1.867928105158e+00, 8.953628085922e-01}, {0.0, 0.0}};
        bands_[2].sections[2] = {{1.0, -2.0, 1.0}, {1.0, -1.862185590588e+00, 9.329620631933e-01}, {0.0, 0.0}};
        bands_[2].sections[3] = {{1.0, -2.0, 1.0}, {1.0, -1.944247162457e+00, 9.640318014612e-01}, {0.0, 0.0}};

        // Banda 3: 1000.0 Hz
        bands_[3].num_sections = 4;
        bands_[3].sections.resize(4);
        bands_[3].sections[0] = {{2.649169787105e-04, 5.298339574211e-04, 2.649169787105e-04}, {1.0, -1.567979927574e+00, 7.451188265717e-01}, {0.0, 0.0}};
        bands_[3].sections[1] = {{1.0, 2.0, 1.0}, {1.0, -1.695927721709e+00, 7.996569957415e-01}, {0.0, 0.0}};
        bands_[3].sections[2] = {{1.0, -2.0, 1.0}, {1.0, -1.602854863545e+00, 8.723357477678e-01}, {0.0, 0.0}};
        bands_[3].sections[3] = {{1.0, -2.0, 1.0}, {1.0, -1.851331408605e+00, 9.286713520011e-01}, {0.0, 0.0}};

        // Banda 4: 2000.0 Hz
        bands_[4].num_sections = 4;
        bands_[4].sections.resize(4);
        bands_[4].sections[0] = {{3.159595780133e-03, 6.319191560266e-03, 3.159595780133e-03}, {1.0, -9.406125883449e-01, 5.513951788403e-01}, {0.0, 0.0}};
        bands_[4].sections[1] = {{1.0, 2.0, 1.0}, {1.0, -1.257502232167e+00, 6.254785345181e-01}, {0.0, 0.0}};
        bands_[4].sections[2] = {{1.0, -2.0, 1.0}, {1.0, -8.230372386799e-01, 7.750507326245e-01}, {0.0, 0.0}};
        bands_[4].sections[3] = {{1.0, -2.0, 1.0}, {1.0, -1.564855070351e+00, 8.569921820626e-01}, {0.0, 0.0}};

        // Banda 5: 4000.0 Hz
        bands_[5].num_sections = 4;
        bands_[5].sections.resize(4);
        bands_[5].sections[0] = {{3.148416921763e-02, -6.296833843527e-02, 3.148416921763e-02}, {1.0, -2.188890567452e-01, 2.932491502268e-01}, {0.0, 0.0}};
        bands_[5].sections[1] = {{1.0, 2.0, 1.0}, {1.0, 4.972247967616e-01, 3.212405063543e-01}, {0.0, 0.0}};
        bands_[5].sections[2] = {{1.0, -2.0, 1.0}, {1.0, -7.073322598609e-01, 6.851480758047e-01}, {0.0, 0.0}};
        bands_[5].sections[3] = {{1.0, 2.0, 1.0}, {1.0, 1.005311669151e+00, 7.141864684917e-01}, {0.0, 0.0}};

        // Banda 6: 8000.0 Hz (highpass)
        bands_[6].num_sections = 2;
        bands_[6].sections.resize(2);
        bands_[6].sections[0] = {{1.717194755996e-02, -3.434389511992e-02, 1.717194755996e-02}, {1.0, 6.981629389111e-01, 1.526549316879e-01}, {0.0, 0.0}};
        bands_[6].sections[1] = {{1.0, -2.0, 1.0}, {1.0, 9.286324485620e-01, 5.331561042611e-01}, {0.0, 0.0}};
    }
}