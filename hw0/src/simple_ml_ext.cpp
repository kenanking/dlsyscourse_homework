#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for (size_t i = 0; i < m; i += batch) {
        size_t batch_size = std::min(batch, m - i);
        float *logits = new float[batch_size * k];
        float *gradients = new float[n * k];

        for (size_t j = 0; j < batch_size; j++) {
            float sum = 0;
            for (size_t l = 0; l < k; l++) {
                logits[j * k + l] = 0;
                for (size_t p = 0; p < n; p++) {
                    logits[j * k + l] += theta[p * k + l] * X[(i + j) * n + p];
                }
                logits[j * k + l] = std::exp(logits[j * k + l]);
                sum += logits[j * k + l];
            }

            for (size_t l = 0; l < k; l++) {
                logits[j * k + l] /= sum;
            }
        }

        // compute gradients
        for (size_t j = 0; j < n; j++) {
            for (size_t l = 0; l < k; l++) {
                gradients[j * k + l] = 0;
                for (size_t p = 0; p < batch_size; p++) {
                    gradients[j * k + l] += (logits[p * k + l] - (l == y[i + p])) * X[(i + p) * n + j];
                }
                gradients[j * k + l] /= batch_size;
            }
        }

        // update theta
        for (size_t j = 0; j < n; j++) {
            for (size_t l = 0; l < k; l++) {
                theta[j * k + l] -= lr * gradients[j * k + l];
            }
        }

        delete[] logits;
        delete[] gradients;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
