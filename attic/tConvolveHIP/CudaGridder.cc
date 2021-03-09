/// @copyright (c) 2009 CSIRO
/// Australia Telescope National Facility (ATNF)
/// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
/// PO Box 76, Epping NSW 1710, Australia
/// atnf-enquiries@csiro.au
///
/// This file is part of the ASKAP software distribution.
///
/// The ASKAP software distribution is free software: you can redistribute it
/// and/or modify it under the terms of the GNU General Public License as
/// published by the Free Software Foundation; either version 2 of the License,
/// or (at your option) any later version.
///
/// This program is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// GNU General Public License for more details.
///
/// You should have received a copy of the GNU General Public License
/// along with this program; if not, write to the Free Software
/// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
///
/// @author Ben Humphreys <ben.humphreys@csiro.au>
/// @author Tim Cornwell  <tim.cornwell@csiro.au>

// Include own header file first
#include "CudaGridder.h"

// System includes
#include <iostream>
#include <cstdlib>
#include <vector>
#include <complex>

// Cuda includes
#include <hip/hip_runtime_api.h>

// Local includes
#include "CudaGridKernel.h"
#include "Stopwatch.h"

typedef float Real;
typedef std::complex<Real> Value;

void checkerror(hipError_t err)
{
    if (err != hipSuccess)
    {
        std::cout << "CUDA Error: " << hipGetErrorString(err) << std::endl;
        exit(1);
    }
}

void gridKernelCuda(const std::vector< std::complex<float> >& data, const int support,
        const std::vector< std::complex<float> >& C, const std::vector<int>& cOffset,
        const std::vector<int>& iu, const std::vector<int>& iv,
        std::vector< std::complex<float> >& grid, const int gSize,
        double &time)
{
    // Report the type of device being used
    int device;
    hipDeviceProp_t devprop;
    hipGetDevice(&device);
    hipGetDeviceProperties(&devprop, device);
    std::cout << "    Using CUDA Device " << device << ": "
        << devprop.name << std::endl;

    // Need to convert all std::vectors to C arrays for CUDA, then call
    // the kernel exec function. NOTE: The std::vector is the only STL
    // container which you can treat as an array like we do here.

    // Allocate device memory
    Value *d_grid;
    Value *d_C;
    int *d_cOffset;
    int *d_iu;
    int *d_iv;
    Value *d_data;

    hipError_t err;
    err = hipMalloc((void **) &d_grid, grid.size() * sizeof(Value));
    checkerror(err);
    err = hipMalloc((void **) &d_C, C.size() * sizeof(Value));
    checkerror(err);
    err = hipMalloc((void **) &d_cOffset, cOffset.size() * sizeof(unsigned int));
    checkerror(err);
    err = hipMalloc((void **) &d_iu, iu.size() * sizeof(unsigned int));
    checkerror(err);
    err = hipMalloc((void **) &d_iv, iv.size() * sizeof(unsigned int));
    checkerror(err);
    err = hipMalloc((void **) &d_data, data.size() * sizeof(Value));
    checkerror(err);

    // Copy host vectors to device arrays
    err = hipMemcpy(d_grid, &grid[0], grid.size() * sizeof(Value), hipMemcpyHostToDevice);
    checkerror(err);
    err = hipMemcpy(d_C, &C[0], C.size() * sizeof(Value), hipMemcpyHostToDevice);
    checkerror(err);
    err = hipMemcpy(d_cOffset, &cOffset[0], cOffset.size() * sizeof(unsigned int), hipMemcpyHostToDevice);
    checkerror(err);
    err = hipMemcpy(d_iu, &iu[0], iu.size() * sizeof(unsigned int), hipMemcpyHostToDevice);
    checkerror(err);
    err = hipMemcpy(d_iv, &iv[0], iu.size() * sizeof(unsigned int), hipMemcpyHostToDevice);
    checkerror(err);
    err = hipMemcpy(d_data, &data[0], data.size() * sizeof(Value), hipMemcpyHostToDevice);
    checkerror(err);

    Stopwatch sw;
    sw.start();
    cuda_gridKernel((const Complex *)d_data, data.size(), support,
            (const Complex *)d_C, d_cOffset, d_iu, d_iv,
            (Complex *)d_grid, gSize,
            &iu[0], &iv[0]);
    hipDeviceSynchronize();
    time = sw.stop();

    // Copy device arrays back into the host vector
    err = hipMemcpy(&grid[0], d_grid, grid.size() * sizeof(Value), hipMemcpyDeviceToHost);
    checkerror(err);

    // Free device memory
    hipFree(d_grid);
    hipFree(d_C);
    hipFree(d_cOffset);
    hipFree(d_iu);
    hipFree(d_iv);
    hipFree(d_data);
}

void degridKernelCuda(const std::vector< std::complex<float> >& grid,
        const int gSize,
        const int support,
        const std::vector< std::complex<float> >& C,
        const std::vector<int>& cOffset,
        const std::vector<int>& iu,
        const std::vector<int>& iv,
        std::vector< std::complex<float> >& data,
        double &time)
{
    // Report the type of device being used
    int device;
    hipDeviceProp_t devprop;
    hipGetDevice(&device);
    hipGetDeviceProperties(&devprop, device);
    std::cout << "    Using CUDA Device " << device << ": "
        << devprop.name << std::endl;

    // Need to convert all std::vectors to C arrays for CUDA, then call
    // the kernel exec function. NOTE: The std::vector is the only STL
    // container which you can treat as an array like we do here.

    // Allocate device memory
    Value *d_grid;
    Value *d_C;
    int *d_cOffset;
    int *d_iu;
    int *d_iv;
    Value *d_data;

    hipError_t err;
    err = hipMalloc((void **) &d_grid, grid.size() * sizeof(Value));
    checkerror(err);
    err = hipMalloc((void **) &d_C, C.size() * sizeof(Value));
    checkerror(err);
    err = hipMalloc((void **) &d_cOffset, cOffset.size() * sizeof(unsigned int));
    checkerror(err);
    err = hipMalloc((void **) &d_iu, iu.size() * sizeof(unsigned int));
    checkerror(err);
    err = hipMalloc((void **) &d_iv, iv.size() * sizeof(unsigned int));
    checkerror(err);
    err = hipMalloc((void **) &d_data, data.size() * sizeof(Value));
    checkerror(err);

    // Copy host vectors to device arrays
    err = hipMemcpy(d_grid, &grid[0], grid.size() * sizeof(Value), hipMemcpyHostToDevice);
    checkerror(err);
    err = hipMemcpy(d_C, &C[0], C.size() * sizeof(Value), hipMemcpyHostToDevice);
    checkerror(err);
    err = hipMemcpy(d_cOffset, &cOffset[0], cOffset.size() * sizeof(unsigned int), hipMemcpyHostToDevice);
    checkerror(err);
    err = hipMemcpy(d_iu, &iu[0], iu.size() * sizeof(unsigned int), hipMemcpyHostToDevice);
    checkerror(err);
    err = hipMemcpy(d_iv, &iv[0], iv.size() * sizeof(unsigned int), hipMemcpyHostToDevice);
    checkerror(err);
    err = hipMemcpy(d_data, &data[0], data.size() * sizeof(Value), hipMemcpyHostToDevice);
    checkerror(err);

    Stopwatch sw;
    sw.start();
    cuda_degridKernel((const Complex *)d_grid, gSize, support,
            (const Complex *)d_C, d_cOffset, d_iu, d_iv,
            (Complex *)d_data, data.size());
    hipDeviceSynchronize();
    time = sw.stop();

    // Copy device arrays back into the host vector
    err = hipMemcpy(&data[0], d_data, data.size() * sizeof(Value), hipMemcpyDeviceToHost);
    checkerror(err);

    // Free device memory
    hipFree(d_grid);
    hipFree(d_C);
    hipFree(d_cOffset);
    hipFree(d_iu);
    hipFree(d_iv);
    hipFree(d_data);
}

