// Copyright 2020 Wang Dai, ISTBI, Fudan University China

//#include <cfloat>
// #include <cmath>
#include "fast_dawson.hpp"

namespace py = pybind11;

// Python I/F function
void mnn_dawson1(py::array_t<float> x, py::array_t<float> y, unsigned int size)
{

  auto arr_x = x.mutable_unchecked<1>();
  auto arr_y = y.mutable_unchecked<1>();

  // Call cpu func
  dawson1_cpu(arr_x.mutable_data(0), arr_y.mutable_data(0), size);

}

// Python I/F function
// void dawson1_int_cpu(float* x, float* y, unsigned int size, float* cheb, int cheb_len, const int N =7)
// void mnn_dawson1_int(py::array_t<float> x, py::array_t<float> y, unsigned int size, py::array_t<float> cheb, int cheb_len, int N)
// py::array_t<float>
void mnn_dawson1_int(py::array_t<float> x, py::array_t<float>  &y, unsigned int size,  py::array_t<double> cheb, int cheb_len, 
                        int N)
{

  auto arr_x = x.mutable_unchecked<1>();
  auto arr_y = y.mutable_unchecked<1>();
  auto arr_cheb = cheb.mutable_unchecked<1>();

  float *x_p = arr_x.mutable_data(0);
  float *y_p = arr_y.mutable_data(0);
  double *cheb_p = arr_cheb.mutable_data(0);

// Call cpu func
//  dawson1_int_cpu(arr_x.mutable_data(0), arr_y.mutable_data(0), size, arr_cheb.mutable_data(0), cheb_len, N);
//dawson1_int_cpu(float*, float*, unsigned int, float*, int, const int);
  dawson1_int_cpu(x_p, y_p, size, cheb_p, cheb_len, N);


}

// Python I/F function
// void dawson1_int_cpu(float* x, float* y, unsigned int size, float* cheb, int cheb_len, const int N =7)
void mnn_dawson2(py::array_t<float> x, py::array_t<float> y, unsigned int size, py::array_t<double> cheb, int cheb_len, 
                  py::array_t<float> asym_neginf, int N)
//void mnn_dawson2(py::array_t<float> x, py::array_t<float> y, unsigned int size, py::array_t<float> cheb, int cheb_len, 
//                  py::array_t<float> asym_neginf, int N)
{
    auto arr_x = x.mutable_unchecked<1>();
    auto arr_y = y.mutable_unchecked<1>();
    auto arr_cheb = cheb.mutable_unchecked<1>();
    auto arr_asym_neginf = asym_neginf.mutable_unchecked<1>();

    // Call cpu func
    dawson2_cpu(arr_x.mutable_data(0), arr_y.mutable_data(0), size, arr_cheb.mutable_data(0), cheb_len, 
                  arr_asym_neginf.mutable_data(0), N);

}

// Python I/F function
// void dawson1_int_cpu(float* x, float* y, unsigned int size, float* cheb, int cheb_len, const int N =7)

// void dawson2_int_cpu(double* x, float* y, unsigned int size, double* cheb, int cheb_len, double* asym_neginf, const int N)
void mnn_dawson2_int(py::array_t<float> x, py::array_t<float> y, unsigned int size, py::array_t<double> cheb, int cheb_len, 
                        py::array_t<float> asym_neginf, int N)
{
    auto arr_x = x.mutable_unchecked<1>();
    auto arr_y = y.mutable_unchecked<1>();
    auto arr_cheb = cheb.mutable_unchecked<1>();
    auto arr_asym_neginf = asym_neginf.mutable_unchecked<1>();

    // Call cpu func
    dawson2_int_cpu(arr_x.mutable_data(0), arr_y.mutable_data(0), size, arr_cheb.mutable_data(0), cheb_len, 
                  arr_asym_neginf.mutable_data(0),  N);

}


//+++++++++++++++++++++++++++++++++++++
// Define interface for Python

// Python I/F function
// void mnn_dawson1(py::array_t<float> x, py::array_t<float> y, unsigned int size)
// void mnn_dawson1_int(py::array_t<float> x, py::array_t<float> y, unsigned int size, py::array_t<float> cheb, int cheb_len, int N)
// void mnn_dawson2(py::array_t<float> x, py::array_t<float> y, unsigned int size, py::array_t<float> cheb, int cheb_len, int N)
// void mnn_dawson2_int(py::array_t<float> x, py::array_t<float> y, unsigned int size, py::array_t<float> cheb, int cheb_len, int N)


PYBIND11_MODULE(fast_dawson, m) {
      
      m.doc() = "pybind11 I/F plugin"; // optional module docstring

      m.def("mnn_dawson1", &mnn_dawson1, "Call MNN dawson1 kernel function calculating in parallel"); //,
            //py::array_t<float>, py::array_t<float>, unsigned int );
      m.def("mnn_dawson1_int", &mnn_dawson1_int, "Call MNN dawson1 int_fast kernel function calculating in parallel"); //,
            // py::array_t<float>, py::array_t<float>, unsigned int, py::array_t<float>, int, const int);      
      m.def("mnn_dawson2", &mnn_dawson2, "Call MNN dawson2 kernel function calculating in parallel"); // ,
            // py::array_t<float>, py::array_t<float>, unsigned int, py::array_t<float>, int, const int);      
      m.def("mnn_dawson2_int", &mnn_dawson2_int, "Call MNN dawson2 int_fast kernel function calculating in parallel"); //,
            // py::array_t<float>, py::array_t<float>, unsigned int, py::array_t<float>, int, const int);

}
