/**
 * @file bindings.cpp
 * @brief pybind11 bindings exposing the C++ audio-processing primitives to Python.
 *
 * The module name is ``audio_processing``. It provides Python access to:
 * - ::AudioProcessor (static utility methods: rms, snr, rms_comp)
 * - ::ClarityCalculator (clarity metrics C50/C80 via @c calculate )
 * - ::DefinitionCalculator (D50 metric via @c calculate )
 * - ::OctaveFilterBank (multi‑band filter processing)
 *
 * Implementation notes
 * --------------------
 * * NumPy arrays are converted to STL containers where needed.
 * * Only 1‑D NumPy inputs are accepted for scalar calculators (double precision).
 * * The OctaveFilterBank::process binding accepts a 1‑D float array (audio
 *   samples) and returns a 2‑D NumPy array shaped (num_bands, num_samples) in
 *   double precision.
 */

 #include <pybind11/pybind11.h>
 #include <pybind11/numpy.h>
 #include <pybind11/stl.h>
 #include "audio_processor.hpp"
 #include "clarity_calculator.hpp"
 #include "definition_calculator.hpp"
 
 namespace py = pybind11;
 
 /**
  * @brief Convert a 1‑D NumPy (double) array to a std::vector<double>.
  * @param array Input NumPy array (must be 1‑D).
  * @return std::vector<double> holding a copy of the data.
  * @throws std::runtime_error if the array is not 1‑D.
  */
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
 
     // ---------------------------------------------------------------------
     // AudioProcessor bindings
     // ---------------------------------------------------------------------
     py::class_<AudioProcessor>(m, "AudioProcessor")
         .def(py::init<>())  // Default constructor
         .def_static("rms", [](py::array_t<double> array) {
             return AudioProcessor::rms(numpy_to_vector(array));
         })
         .def_static("snr", [](double signal_rms, double noise_rms) {
             return AudioProcessor::snr(signal_rms, noise_rms);
         })
         .def_static("rms_comp", [](double signal_rms, double noise_rms, double snr_required) {
             return AudioProcessor::rms_comp(signal_rms, noise_rms, snr_required);
         });
 
     // ---------------------------------------------------------------------
     // ClarityCalculator bindings
     // ---------------------------------------------------------------------
     py::class_<ClarityCalculator>(m, "ClarityCalculator")
         .def_static("calculate", [](double time_ms, py::array_t<double> array, double fs) {
             return ClarityCalculator::calculate(time_ms, numpy_to_vector(array), fs);
         });
 
     // ---------------------------------------------------------------------
     // DefinitionCalculator bindings
     // ---------------------------------------------------------------------
     py::class_<DefinitionCalculator>(m, "DefinitionCalculator")
         .def_static("calculate", [](py::array_t<double> array, double fs) {
             return DefinitionCalculator::calculate(numpy_to_vector(array), fs);
         });
 
 }