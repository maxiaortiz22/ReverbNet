#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "audio_processor.hpp"
#include "clarity_calculator.hpp"
#include "definition_calculator.hpp"
#include "tr_lundeby_calculator.hpp"
#include "tr_augmentation_calculator.hpp"
#include "octave_filter_bank.hpp"

namespace py = pybind11;

// Helper function to convert numpy array to std::vector
std::vector<double> numpy_to_vector(py::array_t<double> array) {
    py::buffer_info buf = array.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Number of dimensions must be 1");
    }
    auto *ptr = static_cast<double *>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.size);
}

PYBIND11_MODULE(audio_processing, m) {
    m.doc() = "Audio processing module with C++ implementations";

    // AudioProcessor bindings
    py::class_<AudioProcessor>(m, "AudioProcessor")
        .def(py::init<>())  // Add default constructor
        .def_static("rms", [](py::array_t<double> array) {
            return AudioProcessor::rms(numpy_to_vector(array));
        })
        .def_static("snr", [](double signal_rms, double noise_rms) {
            return AudioProcessor::snr(signal_rms, noise_rms);
        })
        .def_static("rms_comp", [](double signal_rms, double noise_rms, double snr_required) {
            return AudioProcessor::rms_comp(signal_rms, noise_rms, snr_required);
        });

    // ClarityCalculator bindings
    py::class_<ClarityCalculator>(m, "ClarityCalculator")
        .def_static("calculate", [](double time_ms, py::array_t<double> array, double fs) {
            return ClarityCalculator::calculate(time_ms, numpy_to_vector(array), fs);
        });

    // DefinitionCalculator bindings
    py::class_<DefinitionCalculator>(m, "DefinitionCalculator")
        .def_static("calculate", [](py::array_t<double> array, double fs) {
            return DefinitionCalculator::calculate(numpy_to_vector(array), fs);
        });

    // TRLundebyCalculator bindings
    py::class_<TRLundebyCalculator>(m, "TRLundebyCalculator")
        .def(py::init<>())
        .def("process", [](TRLundebyCalculator& self, py::array_t<double> input) {
            std::vector<double> input_vec = numpy_to_vector(input);
            std::vector<double> output;
            self.process(input_vec, output);
            return py::array_t<double>({output.size()}, {sizeof(double)}, output.data());
        })
        .def("calculate_t30", [](TRLundebyCalculator& self, py::array_t<double> array, double fs, double max_noise_db) {
            auto result = self.calculate_t30(numpy_to_vector(array), fs, max_noise_db);
            double t30 = std::get<0>(result);
            const std::vector<double>& schroeder_curve = std::get<1>(result);
            double noise_db = std::get<2>(result);
            
            return py::make_tuple(
                t30,
                py::array_t<double>({schroeder_curve.size()}, {sizeof(double)}, schroeder_curve.data()),
                noise_db
            );
        });

    // Expose the LundebyResult struct
    py::class_<TRLundebyCalculator::LundebyResult>(m, "LundebyResult")
        .def_readonly("cross_point", &TRLundebyCalculator::LundebyResult::cross_point)
        .def_readonly("C", &TRLundebyCalculator::LundebyResult::C)
        .def_readonly("noise_db", &TRLundebyCalculator::LundebyResult::noise_db);

    // TRAugmentationCalculator bindings
    py::class_<TRAugmentationCalculator>(m, "TRAugmentationCalculator")
        .def_static("augment_tr", [](py::array_t<double> array, double fs, double target_tr) {
            return TRAugmentationCalculator::augment_tr(numpy_to_vector(array), fs, target_tr);
        })
        .def_static("normalize_rir", [](py::array_t<double> array) {
            return TRAugmentationCalculator::normalize_rir(numpy_to_vector(array));
        })
        .def_static("temporal_decompose", [](py::array_t<double> array, double fs, double tau) {
            return TRAugmentationCalculator::temporal_decompose(numpy_to_vector(array), fs, tau);
        });

    // Expose the TemporalDecomposition struct
    py::class_<TRAugmentationCalculator::TemporalDecomposition>(m, "TemporalDecomposition")
        .def_readonly("delay", &TRAugmentationCalculator::TemporalDecomposition::delay)
        .def_readonly("early", &TRAugmentationCalculator::TemporalDecomposition::early)
        .def_readonly("late", &TRAugmentationCalculator::TemporalDecomposition::late);

    // OctaveFilterBank bindings
    py::class_<OctaveFilterBank>(m, "OctaveFilterBank")
        .def(py::init<int>(), py::arg("filter_order") = 4)
        .def("reset", &OctaveFilterBank::reset)
        .def("process", [](OctaveFilterBank& self, py::array_t<float> input) {
            std::vector<float> input_vec(input.size());
            std::memcpy(input_vec.data(), input.data(), input.size() * sizeof(float));
            auto result = self.process(input_vec);
            
            // Crear un array NumPy 2D con las dimensiones correctas
            auto num_bands = result.size();
            auto num_samples = result[0].size();
            
            // Crear el array NumPy con la forma (num_bands, num_samples)
            auto output = py::array_t<double>({num_bands, num_samples});
            auto output_buf = output.request();
            auto* output_ptr = static_cast<double*>(output_buf.ptr);
            
            // Copiar los datos al array NumPy
            for (size_t i = 0; i < num_bands; ++i) {
                std::memcpy(output_ptr + i * num_samples, result[i].data(), num_samples * sizeof(double));
            }
            
            return output;
        })
        .def_static("get_num_bands", &OctaveFilterBank::getNumBands)
        .def_static("get_center_frequencies", []() {
            const auto& freqs = OctaveFilterBank::getCenterFrequencies();
            return py::array_t<double>(
                {freqs.size()},
                {sizeof(double)},
                freqs.data()
            );
        });
} 