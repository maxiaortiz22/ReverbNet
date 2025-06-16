#pragma once

#include <vector>

// Constantes del banco de filtros
constexpr int NUM_BANDS = 7; // Bandas de octava: 125, 250, 500, 1000, 2000, 4000, 8000 Hz

class OctaveFilterBank {
public:
    /**
     * @brief Construye el banco de filtros de octava.
     * @param filter_order El orden del filtro a utilizar (2 o 4). Por defecto es 4.
     */
    explicit OctaveFilterBank(int filter_order = 4);

    /**
     * @brief Reinicia los estados internos (líneas de retardo) de todos los filtros.
     */
    void reset();

    /**
     * @brief Procesa una señal de audio mono a través de todas las bandas de filtro.
     * @param input Vector con las muestras de audio de entrada.
     * @return Un vector de vectores, donde cada vector interno contiene la señal
     * filtrada para una banda. El orden es el de las frecuencias centrales.
     */
    std::vector<std::vector<double>> process(const std::vector<float>& input);

    /**
     * @brief Devuelve el número de bandas del filtro.
     * @return Número de bandas.
     */
    static int getNumBands() { return NUM_BANDS; }

    /**
     * @brief Devuelve las frecuencias centrales de las bandas del filtro.
     * @return Un vector constante con las frecuencias en Hz.
     */
    static const std::vector<double>& getCenterFrequencies() { return CENTER_FREQUENCIES; }

private:
    // Estructura interna para una sección de filtro de 2do orden (bicuad)
    struct BiquadSection {
        double b[3]; // Coeficientes del numerador
        double a[3]; // Coeficientes del denominador (a[0] siempre es 1.0)
        double z[2]; // Líneas de retardo (estado del filtro)
    };

    // Estructura interna para una banda de filtro completa (cascada de bicuads)
    struct FilterBand {
        std::vector<BiquadSection> sections;
        double center_freq;
        int num_sections;
    };

    // Métodos privados
    void initializeCoefficients();
    double processBiquad(BiquadSection& section, double input);
    double processFilterBand(FilterBand& band, double input);

    // Miembros de la clase
    int filter_order_;
    bool initialized_;
    std::vector<FilterBand> bands_;
    
    static const std::vector<double> CENTER_FREQUENCIES;
};