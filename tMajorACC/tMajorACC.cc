/// @copyright (c) 2017 CSIRO
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
/// @author Ben Humphreys   <ben.humphreys@csiro.au>
/// @author Tim Cornwell    <tim.cornwell@csiro.au>
/// @author Daniel Mitchell <tim.cornwell@csiro.au>

// System includes
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <complex>
#include <vector>
#include <algorithm>
#include <limits>
#include <cassert>

#include <fftw3.h>

// OpenACC includes
#include <openacc.h>

// CUDA includes
#ifdef GPU
#include <cufft.h>
#endif

// Local includes
#include "Stopwatch.h"

#if defined(VERIFY)
	#define RUN_CPU 1
	#define RUN_VERIFY 1
#elif defined(SINGLECPU)
	#define RUN_CPU 1
#endif

using std::cout;
using std::endl;
using std::abs;

// Typedefs for easy testing
// Cost of using double for Coord is low, cost for
// double for float is also low
typedef double Coord;


/////////////////////////////////////////////////////////////////////////////////
// The next two functions are the kernel of the gridding/degridding.
// The data are presented as a vector. Offsets for the convolution function
// and for the grid location are precalculated so that the kernel does
// not need to know anything about world coordinates or the shape of
// the convolution function. The ordering of cOffset and iu, iv is
// random - some presorting might be advantageous.
//
// Perform gridding
//
// data - values to be gridded in a 1D vector
// support - Total width of convolution function=2*support+1
// C - convolution function shape: (2*support+1, 2*support+1, *)
// cOffset - offset into convolution function per data point
// iu, iv - integer locations of grid points
// grid - Output grid: shape (gSize, *)
// gSize - size of one axis of grid
void gridKernel(const std::vector<std::complex<float> >& data, const int support,
                const std::vector<std::complex<float> >& C, const std::vector<int>& cOffset,
                const std::vector<int>& iu, const std::vector<int>& iv,
                std::vector<std::complex<float> >& grid, const int gSize, const bool isPSF)
{
    const int sSize = 2 * support + 1;

    std::complex<float> *d_grid = grid.data();
    const std::complex<float> *d_data = data.data();
    const std::complex<float> *d_C = C.data();

    float dre, dim;
    if ( isPSF ) {
        dre = 1.0;
        dim = 0.0;
    }

    for (int dind = 0; dind < int(data.size()); ++dind) {
        // The actual grid point
        int gind = iu[dind] + gSize * iv[dind] - support;
        // The Convoluton function point from which we offset
        int cind = cOffset[dind];
        int suppu, suppv;
        if ( !isPSF ) {
            dre = d_data[dind].real();
            dim = d_data[dind].imag();
        }

        for (suppv = 0; suppv < sSize; suppv++) {
            for (suppu = 0; suppu < sSize; suppu++) {
                float *dref = (float *)&d_grid[gind+suppv*gSize+suppu];
                const int supp = cind + suppv*sSize+suppu;
                const float reval = dre * d_C[supp].real() - dim * d_C[supp].imag();
                const float imval = dim * d_C[supp].real() + dre * d_C[supp].imag();
                dref[0] = dref[0] + reval;
                dref[1] = dref[1] + imval;
            }
        }

    }
}

void gridKernelACC(const std::vector<std::complex<float> >& data, const int support,
        const std::vector<std::complex<float> >& C, const std::vector<int>& cOffset,
        const std::vector<int>& iu, const std::vector<int>& iv,
        std::vector<std::complex<float> >& grid, const int gSize, const bool isPSF)
{
    const int sSize = 2 * support + 1;

    // std::complex<float> = std::complex<float> = std::complex<float>
    //float *d_grid = (float *)grid.data();
    std::complex<float> *d_grid = grid.data();
    const int d_size = data.size();
    const std::complex<float> *d_data = data.data();
    const std::complex<float> *d_C = C.data();
    const int c_size = C.size();
    const int *d_cOffset = cOffset.data();
    const int *d_iu = iu.data();
    const int *d_iv = iv.data();

    //int dind;
    // Both of the following approaches are the same when running on multicore CPUs.
    // Using "gang vector" here without the inner pragma below sets 1 vis per vector element (c.f. CUDA thread)
    //#pragma acc parallel loop gang vector
    // Using "gang" here with the inner pragma below sets 1 vis per gang, and 1 pixel per vec element

        //int suppu, suppv;

    float dre, dim;
    //std::complex<float> dval;
    if ( isPSF ) {
        dre = 1.0;
        dim = 0.0;
        //dval = 1.0;
    }

#ifdef GPU

    // tile is giving an ~25% improvement over the collapse(2) loop version on initial P100 tests, but is fragile.
    //  - if the total number of cores is above 1386 or so our of 2048 in our P100 tests, verification fails.
    //  - letting the compiler choose using tile(*,*,*) should work but isn't. Will be fixed in the PGI release after
    //    next (currently using pgc++ 18.3-0).
    // wait(1): wait until async(1) is finished...
    #ifdef GPU
    #pragma acc parallel loop tile(77,6,3) \
            present(d_grid[0:gSize*gSize],d_data[0:d_size],d_C[0:c_size], \
                    d_cOffset[0:d_size],d_iu[0:d_size],d_iv[0:d_size]) wait(1)
    #else
    #pragma acc parallel loop
    #endif
    for (int dind = 0; dind < d_size; ++dind) {
        for (int suppv = 0; suppv < sSize; suppv++) {
            for (int suppu = 0; suppu < sSize; suppu++) {

        int cind = d_cOffset[dind];
        //const float *c_C = (float *)&d_C[cind];
        //#pragma acc cache(c_C[0:2*sSize*sSize])

        // The actual grid point
        int gind = d_iu[dind] + gSize * d_iv[dind] - support;
        // The Convoluton function point from which we offset
        if ( !isPSF ) {
            dre = d_data[dind].real();
            dim = d_data[dind].imag();
            //dval = d_data[dind];
        }

        //#pragma acc loop collapse(2)
        //for (int suppv = 0; suppv < sSize; suppv++) {
        //    for (int suppu = 0; suppu < sSize; suppu++) {
                float *dref = (float *)&d_grid[gind+suppv*gSize+suppu];
                const int supp = cind + suppv*sSize+suppu;
                const float reval = dre * d_C[supp].real() - dim * d_C[supp].imag();
                const float imval = dim * d_C[supp].real() + dre * d_C[supp].imag();
                // note the real mults above are only really needed on the CPUs...
                //const std::complex<float> cval = dval * d_C[cind+suppv*sSize+suppu];
                #pragma acc atomic update
                dref[0] = dref[0] + reval;
                //dref[0] = dref[0] + cval.real();
                #pragma acc atomic update
                dref[1] = dref[1] + imval;
                //dref[1] = dref[1] + cval.imag();
            }
        }
    }
#else
    for (int dind = 0; dind < d_size; ++dind) {
        int cind = d_cOffset[dind];
        //const float *c_C = (float *)&d_C[cind];
        //#pragma acc cache(c_C[0:2*sSize*sSize])

        // The actual grid point
        int gind = d_iu[dind] + gSize * d_iv[dind] - support;
        // The Convoluton function point from which we offset
        int suppu, suppv;
        if ( !isPSF ) {
            dre = d_data[dind].real();
            dim = d_data[dind].imag();
        }

        #pragma acc parallel loop gang vector collapse(2)
        for (suppv = 0; suppv < sSize; suppv++) {
            for (suppu = 0; suppu < sSize; suppu++) {
                float *dref = (float *)&d_grid[gind+suppv*gSize+suppu];
                //const int suppre = 2 * (suppv*sSize+suppu);
                const int supp = cind + suppv*sSize + suppu;
                const float reval = dre * d_C[supp].real() - dim * d_C[supp].imag();
                const float imval = dim * d_C[supp].real() + dre * d_C[supp].imag();
                dref[0] = dref[0] + reval;
                dref[1] = dref[1] + imval;
            }
        }
    }
#endif

}

// Simple degridding to set visibilities
void degridNN(const std::vector<std::complex<float> >& grid, const int gSize,
              const std::vector<int>& iu, const std::vector<int>& iv,
              std::vector<std::complex<float> >& data)
{
    std::complex<float> *d_data = data.data();
    const std::complex<float> *d_grid = grid.data();
    for (int dind = 0; dind < int(data.size()); ++dind) {
        d_data[dind] = d_grid[iu[dind] + gSize * iv[dind]];
    }
}

// Perform degridding
void degridKernel(const std::vector<std::complex<float> >& grid, const int gSize, const int support,
                  const std::vector<std::complex<float> >& C, const std::vector<int>& cOffset,
                  const std::vector<int>& iu, const std::vector<int>& iv,
                  std::vector<std::complex<float> >& data)
{
    const int sSize = 2 * support + 1;

    std::complex<float> *d_data = data.data();
    const std::complex<float> *d_grid = grid.data();
    const std::complex<float> *d_C = C.data();

    for (int dind = 0; dind < int(data.size()); ++dind) {

        // The actual grid point from which we offset
        int gind = iu[dind] + gSize * iv[dind] - support;
        // The Convoluton function point from which we offset
        const int cind = cOffset[dind];

        float re = 0.0, im = 0.0;
        for (int suppv = 0; suppv < sSize; suppv++) {
            for (int suppu = 0; suppu < sSize; suppu++) {
                re = re + d_grid[gind+suppv*gSize+suppu].real() * d_C[cind+suppv*sSize+suppu].real() -
                          d_grid[gind+suppv*gSize+suppu].imag() * d_C[cind+suppv*sSize+suppu].imag();
                im = im + d_grid[gind+suppv*gSize+suppu].imag() * d_C[cind+suppv*sSize+suppu].real() +
                          d_grid[gind+suppv*gSize+suppu].real() * d_C[cind+suppv*sSize+suppu].imag();
            }
        }
        d_data[dind] = std::complex<float>(re,im);

    }
}

void degridKernelACC(const std::vector<std::complex<float> >& grid, const int gSize, const int support,
                     const std::vector<std::complex<float> >& C, const std::vector<int>& cOffset,
                     const std::vector<int>& iu, const std::vector<int>& iv,
                     std::vector<std::complex<float> >& data)
{
    const int sSize = 2 * support + 1;

    const int d_size = data.size();
    std::complex<float> *d_data = data.data();
    const std::complex<float> *d_grid = grid.data();
    const int c_size = C.size();
    const std::complex<float> *d_C = C.data();
    const int *d_cOffset = cOffset.data();
    const int *d_iu = iu.data();
    const int *d_iv = iv.data();

    int dind;

    #pragma acc parallel loop present(d_grid[0:gSize*gSize],d_data[0:d_size],d_C[0:c_size], \
                                      d_cOffset[0:d_size],d_iu[0:d_size],d_iv[0:d_size])
    for (dind = 0; dind < d_size; ++dind) {

        // The actual grid point from which we offset
        int gind = d_iu[dind] + gSize * d_iv[dind] - support;
        // The Convoluton function point from which we offset
        int cind = d_cOffset[dind];
        float re = 0.0, im = 0.0;

/*
#ifdef GPU
        #pragma acc loop reduction(+:re,im) collapse(2)
        for (int suppv = 0; suppv < sSize; suppv++) {
            for (int suppu = 0; suppu < sSize; suppu++) {
                const std::complex<float> cval = d_grid[gind+suppv*gSize+suppu] * d_C[cind+suppv*sSize+suppu];
                re = re + cval.real();
                im = im + cval.imag();
            }
        }
#else
*/
        for (int suppv = 0; suppv < sSize; suppv++) {
            for (int suppu = 0; suppu < sSize; suppu++) {
                re = re + d_grid[gind+suppv*gSize+suppu].real() * d_C[cind+suppv*sSize+suppu].real() -
                          d_grid[gind+suppv*gSize+suppu].imag() * d_C[cind+suppv*sSize+suppu].imag();
                im = im + d_grid[gind+suppv*gSize+suppu].imag() * d_C[cind+suppv*sSize+suppu].real() +
                          d_grid[gind+suppv*gSize+suppu].real() * d_C[cind+suppv*sSize+suppu].imag();
            }
        }
//#endif

        d_data[dind] = std::complex<float>(re,im);

    }

}

/////////////////////////////////////////////////////////////////////////////////
// Initialize W project convolution function
// - This is application specific and should not need any changes.
//
// freq - temporal frequency (inverse wavelengths)
// cellSize - size of one grid cell in wavelengths
// support - Total width of convolution function=2*support+1
// wCellSize - size of one w grid cell in wavelengths
// wSize - Size of lookup table in w
void initC(const std::vector<Coord>& freq, const Coord cellSize,
           const Coord baseline,
           const int wSize, int& support, int& overSample,
           Coord& wCellSize, std::vector<std::complex<float> >& C)
{
    cout << "Initializing W projection convolution function" << endl;
    // changed 1.5x to 0.9x to reduce kernel memory below 48 KB.
    support = static_cast<int>(1.5 * sqrt(std::abs(baseline) * static_cast<Coord>(cellSize)
                                          * freq[0]) / cellSize);
    overSample = 8;
    cout << "Support = " << support << " pixels" << endl;
    wCellSize = 2 * baseline * freq[0] / wSize;
    cout << "W cellsize = " << wCellSize << " wavelengths" << endl;

    // Convolution function. This should be the convolution of the
    // w projection kernel (the Fresnel term) with the convolution
    // function used in the standard case. The latter is needed to
    // suppress aliasing. In practice, we calculate entire function
    // by Fourier transformation. Here we take an approximation that
    // is good enough.
    const int sSize = 2 * support + 1;

    const int cCenter = (sSize - 1) / 2;

    C.resize(sSize*sSize*overSample*overSample*wSize);
    cout << "Size of convolution function = " << sSize*sSize*overSample
         *overSample*wSize*sizeof(std::complex<float>) / (1024*1024) << " MB" << std::endl;
    cout << "Shape of convolution function = [" << sSize << ", " << sSize << ", "
             << overSample << ", " << overSample << ", " << wSize << "]" << std::endl;

    for (int k = 0; k < wSize; k++) {
        double w = double(k - wSize / 2);
        double fScale = sqrt(abs(w) * wCellSize * freq[0]) / cellSize;

        for (int osj = 0; osj < overSample; osj++) {
            for (int osi = 0; osi < overSample; osi++) {
                for (int j = 0; j < sSize; j++) {
                    const double j2 = std::pow((double(j - cCenter) + double(osj) / double(overSample)), 2);

                    for (int i = 0; i < sSize; i++) {
                        const double r2 = j2 + std::pow((double(i - cCenter) + double(osi) / double(overSample)), 2);
                        const int cind = i + sSize * (j + sSize * (osi + overSample * (osj + overSample * k)));
if ( i==0 && j==0 ) C[cind] = 1.0;
/*
                        if (w != 0.0) {
                            C[cind] = static_cast<std::complex<float> >(std::cos(r2 / (w * fScale)));
                        } else {
                            C[cind] = static_cast<std::complex<float> >(std::exp(-r2));
                        }
*/

                    }
                }
            }
        }
    }

    // Now normalise the convolution function
    float sumC = 0.0;

    for (int i = 0; i < sSize*sSize*overSample*overSample*wSize; i++) {
        sumC += abs(C[i]);
    }

    for (int i = 0; i < sSize*sSize*overSample*overSample*wSize; i++) {
        C[i] *= std::complex<float>(wSize * overSample * overSample / sumC);
    }
}

// Initialize Lookup function
// - This is application specific and should not need any changes.
//
// freq - temporal frequency (inverse wavelengths)
// cellSize - size of one grid cell in wavelengths
// gSize - size of grid in pixels (per axis)
// support - Total width of convolution function=2*support+1
// wCellSize - size of one w grid cell in wavelengths
// wSize - Size of lookup table in w
void initCOffset(const std::vector<Coord>& u, const std::vector<Coord>& v,
                 const std::vector<Coord>& w, const std::vector<Coord>& freq,
                 const Coord cellSize, const Coord wCellSize,
                 const int wSize, const int gSize, const int support, const int overSample,
                 std::vector<int>& cOffset, std::vector<int>& iu,
                 std::vector<int>& iv)
{
    const int nSamples = u.size();
    const int nChan = freq.size();

    const int sSize = 2 * support + 1;

    // Now calculate the offset for each visibility point
    cOffset.resize(nSamples*nChan);
    iu.resize(nSamples*nChan);
    iv.resize(nSamples*nChan);

    for (int i = 0; i < nSamples; i++) {
        for (int chan = 0; chan < nChan; chan++) {

            const int dind = i * nChan + chan;

            const Coord uScaled = freq[chan] * u[i] / cellSize;
            iu[dind] = int(uScaled);

            if (uScaled < Coord(iu[dind])) {
                iu[dind] -= 1;
            }

            const int fracu = int(overSample * (uScaled - Coord(iu[dind])));
            iu[dind] += gSize / 2;

            const Coord vScaled = freq[chan] * v[i] / cellSize;
            iv[dind] = int(vScaled);

            if (vScaled < Coord(iv[dind])) {
                iv[dind] -= 1;
            }

            const int fracv = int(overSample * (vScaled - Coord(iv[dind])));
            iv[dind] += gSize / 2;

            // The beginning of the convolution function for this point
            Coord wScaled = freq[chan] * w[i] / wCellSize;
            int woff = wSize / 2 + int(wScaled);
            cOffset[dind] = sSize * sSize * (fracu + overSample * (fracv + overSample * woff));
        }
    }

}

// Generate and execute an FFTW plan.
int fftExec(std::vector<std::complex<float> >& grid, const int gSize, const bool forward)
{
    const size_t nPixels = grid.size();
    if (nPixels != gSize*gSize) {
        cout << "bad vector size" << endl;
        return 1;
    }
    if (gSize%2 == 1) {
        cout << "fftExec does not currently support odd sized arrays (fix fftshfit)" << endl;
        return 1;
    }

    std::complex<float> *dataPtr = grid.data();

    // rotate input because the origin for FFTW is at 0, not n/2 (i.e. fftshfit)
    std::vector<std::complex<float> > rotgrid(nPixels);
    std::complex<float> *buffer = rotgrid.data();

    fftwf_plan plan;
    plan = fftwf_plan_dft_2d( gSize, gSize, (fftwf_complex*)buffer, (fftwf_complex*)buffer,
                              (forward) ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_ESTIMATE );

    for (int col = 0; col < gSize; col++) {
        const int colin = col * gSize;
        const int colout = ( ( col + gSize/2 ) % gSize ) * gSize;
        //std::rotate_copy(dataPtr + colin,
        //                 dataPtr + colin + (gSize / 2),
        //                 dataPtr + colin + gSize,
        //                 reinterpret_cast<std::complex<float> *>(buffer + colout));
        for (int row = 0; row < gSize/2; row++) {
            buffer[colout + row] = dataPtr[colin + gSize/2 + row];
            buffer[colout + gSize/2 + row] = dataPtr[colin + row];
        }
    }

    fftwf_execute(plan);

    // rotate back
    for (int col = 0; col < gSize; col++) {
        const int colin = col * gSize;
        const int colout = ( ( col + gSize/2 ) % gSize ) * gSize;
        //std::rotate_copy(reinterpret_cast<std::complex<float> *>(buffer) + colin,
        //                 reinterpret_cast<std::complex<float> *>(buffer) + colin + (gSize / 2),
        //                 reinterpret_cast<std::complex<float> *>(buffer) + colin + gSize,
        //                 dataPtr + colout);
        for (int row = 0; row < gSize/2; row++) {
            dataPtr[colout + row] = buffer[colin + gSize/2 + row];
            dataPtr[colout + gSize/2 + row] = buffer[colin + row];
        }
    }

    // Delete the plan and temporary buffer
    fftwf_destroy_plan(plan);
    //fftwf_free(buffer);

    return 0;

}

// Generate and execute a CUFFT plan.
int fftExecGPU(std::vector<std::complex<float> >& grid, const int gSize, const bool forward)
{
    #ifdef GPU
    const size_t nPixels = grid.size();
    if (nPixels != gSize*gSize) {
        cout << "bad vector size" << endl;
        return 1;
    }
    if (gSize%2 == 1) {
        cout << "fftExecGPU does not currently support odd sized arrays (fix fftshfit)" << endl;
        return 1;
    }

    cufftHandle plan;

    if ( cufftPlan2d( &plan, gSize, gSize, CUFFT_C2C ) != CUFFT_SUCCESS ) {
        cout << "CUFFT error: Plan creation failed" << endl;
        return 1;
    }

    std::complex<float> *dataPtr = grid.data();

    // use a buffer to rotate the pixels for the fft
    std::vector<std::complex<float> > rotgrid(nPixels);
    std::complex<float> *buffer = rotgrid.data();

    // buffer only ever needs to be on the device. So create it then ignore the cpu version
    #pragma acc enter data create(buffer[0:gSize*gSize])

    // rotate input because the origin for CUFFT is at 0, not n/2 (i.e. fftshfit)
    #pragma acc parallel loop collapse(2) present(dataPtr[0:gSize*gSize],buffer[0:gSize*gSize])
    for (int col = 0; col < gSize; col++) {
        for (int row = 0; row < gSize/2; row++) {
            const int colin = col * gSize;
            const int colout = ( ( col + gSize/2 ) % gSize ) * gSize;
            buffer[colout + row] = dataPtr[colin + gSize/2 + row];
            buffer[colout + gSize/2 + row] = dataPtr[colin + row];
        }
    }

    cufftResult fftErr;
    #pragma acc host_data use_device(buffer)
    {
        fftErr = cufftExecC2C(plan, (cufftComplex*)buffer, (cufftComplex*)buffer, (forward) ? CUFFT_FORWARD : CUFFT_INVERSE);
    }
    if ( fftErr != CUFFT_SUCCESS ) {
        cout << "CUFFT error: Forward FFT failed" << endl;
        return 1;
    }
 
    // rotate back
    #pragma acc parallel loop collapse(2) present(dataPtr[0:gSize*gSize],buffer[0:gSize*gSize])
    for (int col = 0; col < gSize; col++) {
        for (int row = 0; row < gSize/2; row++) {
            const int colin = col * gSize;
            const int colout = ( ( col + gSize/2 ) % gSize ) * gSize;
            dataPtr[colout + row] = buffer[colin + gSize/2 + row];
            dataPtr[colout + gSize/2 + row] = buffer[colin + row];
        }
    }

    #pragma acc exit data delete(buffer[0:gSize*gSize])

    // Delete the plan
    cufftDestroy(plan);
    #endif

    return 0;

}

// Generate and execute an FFTW plan.
void fftFix(std::vector<std::complex<float> >& grid, const float scale)
{
    const size_t nPixels = grid.size();
    for (size_t i = 0; i < nPixels; i++) {
        grid[i] = grid[i].real() * scale;
    }
}

// Generate and execute an FFTW plan.
void fftFixGPU(std::vector<std::complex<float> >& grid, const float scale)
{
    #ifdef GPU
    const size_t nPixels = grid.size();
    std::complex<float> *dataPtr = grid.data();
    #pragma acc parallel loop present(dataPtr[0:nPixels])
    for (size_t i = 0; i < nPixels; i++) {
        dataPtr[i] = dataPtr[i].real() * scale;
    }
    #endif
}

static bool abs_compare(std::complex<float> a, std::complex<float> b)
{
    return (std::abs(a) < std::abs(b));
}

// Return a pseudo-random integer in the range 0..2147483647
// Based on an algorithm in Kernighan & Ritchie, "The C Programming Language"
static unsigned long next = 1;
int randomInt()
{
    const unsigned int maxint = std::numeric_limits<int>::max();
    next = next * 1103515245 + 12345;
    return ((unsigned int)(next / 65536) % maxint);
}

// ------------------------------------------------------------------------- //
// Hogbom stuff

void writeImage(const std::string& filename, std::vector<std::complex<float> >& image)
{
    std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
    std::vector<float> realpart(image.size());
    for (int i=0; i<image.size(); i++) realpart[i] = image[i].real();
    file.write(reinterpret_cast<char *>(&realpart[0]), realpart.size() * sizeof(float));
    file.close();
}

struct Position {
    Position(int _x, int _y) : x(_x), y(_y) { };
    int x;
    int y;
};

Position idxToPos(const int idx, const size_t width)
{
    const int y = idx / width;
    const int x = idx % width;
    return Position(x, y);
}

size_t posToIdx(const size_t width, const Position& pos)
{
    return (pos.y * width) + pos.x;
}

void findPeak(const std::vector<std::complex<float> >& image,
              float& maxVal, size_t& maxPos)
{
    maxVal = 0.0;
    maxPos = 0;
    const size_t size = image.size();
    for (size_t i = 0; i < size; ++i) {
        if (abs(image[i].real()) > abs(maxVal)) {
            maxVal = image[i].real();
            maxPos = i;
        }
    }
}

void findPeakACC(const std::complex<float> *data,
              float& maxVal, size_t& maxPos, const size_t size)
{
    size_t tmpPos=0;
    float threadAbsMaxVal = 0.0;

    #pragma acc parallel loop reduction(max:threadAbsMaxVal) gang vector present(data[0:size])
    for (size_t i = 0; i < size; ++i) {
        threadAbsMaxVal = fmaxf( threadAbsMaxVal, abs(data[i].real()) );
    }
    #pragma acc parallel loop gang vector present(data[0:size]) copyout(tmpPos)
    for (size_t i = 0; i < size; ++i) {
        if (abs(data[i].real()) == threadAbsMaxVal) tmpPos = i;
    }

    maxVal = threadAbsMaxVal;
    maxPos = tmpPos;

}

void subtractPsf(const std::vector<std::complex<float> >& psf,
        const size_t psfWidth,
        std::vector<std::complex<float> >& residual,
        const size_t residualWidth,
        const size_t peakPos, const size_t psfPeakPos,
        const float absPeakVal,
        const float gain)
{
    const int rx = idxToPos(peakPos, residualWidth).x;
    const int ry = idxToPos(peakPos, residualWidth).y;

    const int px = idxToPos(psfPeakPos, psfWidth).x;
    const int py = idxToPos(psfPeakPos, psfWidth).y;

    const int diffx = rx - px;
    const int diffy = ry - px;

    const int startx = std::max(0, rx - px);
    const int starty = std::max(0, ry - py);

    const int stopx = std::min(residualWidth - 1, rx + (psfWidth - px - 1));
    const int stopy = std::min(residualWidth - 1, ry + (psfWidth - py - 1));

    for (int y = starty; y <= stopy; ++y) {
        for (int x = startx; x <= stopx; ++x) {
            residual[posToIdx(residualWidth, Position(x, y))] -= gain * absPeakVal
                * psf[posToIdx(psfWidth, Position(x - diffx, y - diffy))].real();
        }
    }
}

void subtractPsfACC(const std::complex<float> *psfdata,
        const size_t psfWidth,
        std::complex<float> *resdata,
        const size_t residualWidth,
        const size_t peakPos, const size_t psfPeakPos,
        const float absPeakVal,
        const float gain)
{
    const int rx = idxToPos(peakPos, residualWidth).x;
    const int ry = idxToPos(peakPos, residualWidth).y;

    const int px = idxToPos(psfPeakPos, psfWidth).x;
    const int py = idxToPos(psfPeakPos, psfWidth).y;

    const int diffx = rx - px;
    const int diffy = ry - px;

    const int startx = std::max(0, rx - px);
    const int starty = std::max(0, ry - py);

    const int stopx = std::min(residualWidth - 1, rx + (psfWidth - px - 1));
    const int stopy = std::min(residualWidth - 1, ry + (psfWidth - py - 1));

    #pragma acc parallel loop collapse(2) gang vector present(resdata[0:residualWidth*residualWidth],psfdata[0:psfWidth*psfWidth])
    for (int y = starty; y <= stopy; ++y) {
        for (int x = startx; x <= stopx; ++x) {
            resdata[posToIdx(residualWidth, Position(x, y))] -= gain * absPeakVal
                * psfdata[posToIdx(psfWidth, Position(x - diffx, y - diffy))].real();
        }
    }
}

void deconvolve(std::vector<std::complex<float> >& residual,
                const size_t dirtyWidth,
                const std::vector<std::complex<float> >& psf,
                const size_t psfWidth,
                std::vector<std::complex<float> >& model,
                const int g_niters)
{

    const float g_gain = 0.1;
    // disable this to keep things more readily comparable
    //float g_threshold;

    // Find the peak of the PSF
    float psfPeakVal = 0.0;
    size_t psfPeakPos = 0;
    findPeak(psf, psfPeakVal, psfPeakPos);
    cout << "    Found peak of PSF: " << "Maximum = " << psfPeakVal
        << " at location " << idxToPos(psfPeakPos, psfWidth).x << ","
       << idxToPos(psfPeakPos, psfWidth).y << endl;

    for (unsigned int i = 0; i < g_niters; ++i) {
        // Find the peak in the residual image
        float absPeakVal = 0.0;
        size_t absPeakPos = 0;
        findPeak(residual, absPeakVal, absPeakPos);

        // Check if threshold has been reached
        //if (i==0) g_threshold = 1e-3 * abs(absPeakVal);
        //if (abs(absPeakVal) < g_threshold) {
        //    cout << "Reached stopping threshold" << endl;
        //    break;
        //}

        // Add to model
        model[absPeakPos] += absPeakVal * g_gain;

        // Subtract the PSF from the residual image
        subtractPsf(psf, psfWidth, residual, dirtyWidth, absPeakPos, psfPeakPos, absPeakVal, g_gain);
    }
}

void deconvolveACC(std::vector<std::complex<float> >& residual,
                const size_t dirtyWidth,
                const std::vector<std::complex<float> >& psf,
                const size_t psfWidth,
                std::vector<std::complex<float> >& model,
                const int g_niters)
{

    const float g_gain = 0.1;
    // disable this to keep things more readily comparable
    //float g_threshold;

    // referece the basic data arrays for use in the parallel loop
    const std::complex<float> *psfdata = psf.data();
    std::complex<float> *resdata = residual.data();
    const size_t psfsize = psf.size();
    const size_t ressize = residual.size();

    // Find the peak of the PSF
    float psfPeakVal = 0.0;
    size_t psfPeakPos = 0;
    findPeakACC(psfdata, psfPeakVal, psfPeakPos, psfsize);
    cout << "    Found peak of PSF: " << "Maximum = " << psfPeakVal
        << " at location " << idxToPos(psfPeakPos, psfWidth).x << ","
       << idxToPos(psfPeakPos, psfWidth).y << endl;

    for (unsigned int i = 0; i < g_niters; ++i) {
        // Find the peak in the residual image
        float absPeakVal = 0.0;
        size_t absPeakPos = 0;
        findPeakACC(resdata, absPeakVal, absPeakPos, ressize);

        // Check if threshold has been reached
        //if (i==0) g_threshold = 1e-3 * abs(absPeakVal);
        //if (abs(absPeakVal) < g_threshold) {
        //    cout << "Reached stopping threshold" << endl;
        //    break;
        //}

        // Add to model
        model[absPeakPos] += absPeakVal * g_gain;

        // Subtract the PSF from the residual image
        subtractPsfACC(psfdata, psfWidth, resdata, dirtyWidth, absPeakPos, psfPeakPos, absPeakVal, g_gain);
    }
}

// ------------------------------------------------------------------------- //
// Main testing routine
int main()
{
    // Change these if necessary to adjust run time
    const int nSamples = 160000; // Number of data samples
    const int wSize = 33; // Number of lookup planes in w projection
    const int nChan = 1; // Number of spectral channels

    // Don't change any of these numbers unless you know what you are doing!
    const int gSize = 4096; // Size of output grid in pixels
    const Coord cellSize = 5.0; // Cellsize of output grid in wavelengths
    const int baseline = 2000; // Maximum baseline in meters

    const int nMajor = 5; // Number of major cycle iterations
    const int nMinor = 100; // Number of minor cycle iterations

    const unsigned int maxint = std::numeric_limits<int>::max();

    // Initialize the uvw data 
    std::vector<Coord> u(nSamples);
    std::vector<Coord> v(nSamples);
    std::vector<Coord> w(nSamples);
    for (int i = 0; i < nSamples; i++) {
        u[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;
        v[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;
        w[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;
    }

    std::vector<std::complex<float> > grid(gSize*gSize);
    grid.assign(grid.size(), std::complex<float>(0.0));

    // Measure frequency in inverse wavelengths
    std::vector<Coord> freq(nChan);

    for (int i = 0; i < nChan; i++) {
        freq[i] = (1.4e9 - 2.0e5 * Coord(i) / Coord(nChan)) / 2.998e8;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Initialize convolution function and offsets
    ///////////////////////////////////////////////////////////////////////////

    std::vector<std::complex<float> > C;
    int support, overSample;
    std::vector<int> cOffset;
    // Vectors of grid centers
    std::vector<int> iu;
    std::vector<int> iv;
    Coord wCellSize;

    initC(freq, cellSize, baseline, wSize, support, overSample, wCellSize, C);
    initCOffset(u, v, w, freq, cellSize, wCellSize, wSize, gSize, support,
                overSample, cOffset, iu, iv);

    const int sSize = 2 * support + 1;

    // asynchronously copy coords to the device while we are doing other initialisation
    int *iu_d = iu.data();
    int *iv_d = iv.data();
    int *cOffset_d = cOffset.data();
    std::complex<float> *C_d = C.data();
    #pragma acc enter data copyin(C_d[0:sSize*sSize*overSample*overSample*wSize], cOffset_d[0:nSamples*nChan], \
                                  iu_d[0:nSamples*nChan], iv_d[0:nSamples*nChan]) async(1)

    const double griddings = (double(nSamples * nChan) * double((sSize) * (sSize)));

    ///////////////////////////////////////////////////////////////////////////
    // Generate initial sky model and visibilties
    ///////////////////////////////////////////////////////////////////////////

    // make an image of point sources (the true sky)
    const int nSources = 100;
    std::vector<std::complex<float> > trueGrid(gSize*gSize);
    trueGrid.assign(trueGrid.size(), std::complex<float>(0.0));
    // record a few test positions for later
    float max1=-1.0, max2=-1.0;
    int tPix1, tPix2;
    for (int i = 0; i < nSources; i++) {
        int l = gSize * Coord(randomInt()) / Coord(maxint);
        int m = gSize * Coord(randomInt()) / Coord(maxint);
        trueGrid[m*gSize+l] += std::complex<float>(randomInt()) / std::complex<float>(maxint);
        if ( trueGrid[m*gSize+l].real() > max1 ) {
            tPix1 = m*gSize+l;
            max1 = trueGrid[tPix1].real();
        }
        if ( trueGrid[m*gSize+l].real() > max2 && trueGrid[m*gSize+l].real() < 0.5 ) {
            tPix2 = m*gSize+l;
            max2 = trueGrid[tPix2].real();
        }
    }

    // store the location and value of the maximum pixel
    std::vector<std::complex<float> >::iterator result;
    //result = std::max_element(trueGrid.begin(), trueGrid.end(), abs_compare);
    //tPix1 = std::distance(trueGrid.begin(), result);
    cout << "test val 1 at: " << tPix1 << ", value = " << trueGrid[tPix1] << endl;
    cout << "test val 2 at: " << tPix2 << ", value = " << trueGrid[tPix2] << endl;

    // FFT to the true uv grid
    if ( fftExec(trueGrid, gSize, true) != 0 ) {
        cout << "fftExec error" << endl;
        return -1;
    }

    result = std::max_element(trueGrid.begin(), trueGrid.end(), abs_compare);
    int testUV = std::distance(trueGrid.begin(), result);
    cout << "max uv val at: " << testUV << ", value = " << trueGrid[testUV] << endl;

    // degrid true visibiltiies from sky true model

    // generate data on GPU, but in the CPU object so that we have to send it to the GPU for profiling
    std::vector<std::complex<float> > visData(nSamples*nChan);
    std::complex<float> *visData_d = visData.data();
    std::complex<float> *trueGrid_d = trueGrid.data();
    #pragma acc enter data create(visData_d[0:nSamples*nChan]) copyin(trueGrid_d[0:gSize*gSize])
    degridKernelACC(trueGrid, gSize, support, C, cOffset, iu, iv, visData);
    #pragma acc exit data delete(trueGrid_d[0:gSize*gSize]) async(2)
    // pull the data back to the CPU and delete/deallocate the GPU copy
    #pragma acc exit data copyout(visData_d[0:nSamples*nChan])

    // make a single-core cpu copy
    std::vector<std::complex<float> > cpuData(visData);
    std::vector<std::complex<float> > cpuModel(nSamples*nChan);
    for (int i = 0; i < nSamples*nChan; i++) {
        cpuModel[i] = 0.0;
    }

    // make an acc copy and send initial visibility data to the device
    std::vector<std::complex<float> > accData(visData);
    std::vector<std::complex<float> > accModel(nSamples*nChan);
    std::complex<float> *accData_d = accData.data();
    #pragma acc enter data copyin(accData_d[0:nSamples*nChan]) async(1)
    std::complex<float> *accModel_d = accModel.data();
    #pragma acc enter data create(accModel_d[0:nSamples*nChan])
    #pragma acc parallel loop present(accModel_d[0:nSamples*nChan])
    for (int i = 0; i < nSamples*nChan; i++) {
        accModel_d[i] = 0.0;
    }

    double time;

    for ( int it_major=0; it_major<nMajor; ++it_major ) {

#ifdef RUN_CPU

        ///////////////////////////////////////////////////////////////////////////
        // CPU single core
        ///////////////////////////////////////////////////////////////////////////
        cout << endl << "+++++ CPU single core +++++" << endl << endl;

        Stopwatch sw_cpu;
        sw_cpu.start();

        //-----------------------------------------------------------------------//
        // DO GRIDDING
        std::vector<std::complex<float> > cpupsfgrid(gSize*gSize);
        cpupsfgrid.assign(cpupsfgrid.size(), std::complex<float>(0.0));
        std::vector<std::complex<float> > cpuImgGrid(gSize*gSize);
        cpuImgGrid.assign(cpuImgGrid.size(), std::complex<float>(0.0));
        {
            // Now we can do the timing for the CPU implementation
            cout << "Gridding PSF" << endl;

            Stopwatch sw;
            sw.start();
            gridKernel(cpuData, support, C, cOffset, iu, iv, cpupsfgrid, gSize, true);
            time = sw.stop();

            // Report on timings
            cout << "    Time " << time << " (s) " << endl;
            cout << "    Time per visibility sample " << 1e6*time / double(cpuData.size()) << " (us) " << endl;
            cout << "    Time per gridding   " << 1e9*time / double(cpuData.size()*sSize*sSize) << " (ns) " << endl;
            cout << "    Gridding rate   " << griddings/1e6/time << " (million grid points per second)" << endl;
        }
        {
            cout << "Gridding data" << endl;

            Stopwatch sw;
            sw.start();
            gridKernel(cpuData, support, C, cOffset, iu, iv, cpuImgGrid, gSize, false);
            time = sw.stop();

            // Report on timings
            cout << "    Time " << time << " (s) " << endl;
            cout << "    Time per visibility sample " << 1e6*time / double(cpuData.size()) << " (us) " << endl;
            cout << "    Time per gridding   " << 1e9*time / double(cpuData.size()*sSize*sSize) << " (ns) " << endl;
            cout << "    Gridding rate   " << griddings/1e6/time << " (million grid points per second)" << endl;
        }

        std::vector<std::complex<float> > cpuuvPsf(cpupsfgrid);
        std::vector<std::complex<float> > cpuuvGrid(cpuImgGrid);

        writeImage("dirty_cpu.img", cpupsfgrid);
        writeImage("psf_cpu.img", cpuImgGrid);

        //-----------------------------------------------------------------------//
        // Form dirty image and run the minor cycle

        // FFT gridded data to form dirty image
        {
            cout << "Inverse FFTs" << endl;

            Stopwatch sw;
            sw.start();
            if ( fftExec(cpupsfgrid, gSize, false) != 0 ) {
                cout << "inverse fftExec error" << endl;
                return -1;
            }
            fftFix(cpupsfgrid, 1.0/float(cpuData.size()));
 
            if ( fftExec(cpuImgGrid, gSize, false) != 0 ) {
                cout << "inverse fftExec error" << endl;
                return -1;
            }
            fftFix(cpuImgGrid, 1.0/float(cpuData.size()));
            time = sw.stop();

            // Report on timings
            cout << "    Time " << time << " (s) " << endl;
        }

        std::vector<std::complex<float> > cpulmPsf(cpupsfgrid);
        std::vector<std::complex<float> > cpulmgrid(cpuImgGrid);

        //-------------------------------------------------------------------//
        // Do Hogbom CLEAN

        std::vector<std::complex<float> > cpuModelGrid(cpuImgGrid.size());
        cpuModelGrid.assign(cpuModelGrid.size(), std::complex<float>(0.0));
        {
            // Now we can do the timing for the serial (Golden) CPU implementation
            cout << "Hogbom clean" << endl;

            Stopwatch sw;
            sw.start();
            deconvolve(cpuImgGrid, gSize, cpupsfgrid, gSize, cpuModelGrid, nMinor);
            time = sw.stop();

            // Report on results
            cout << "    pix 1: "<<cpulmgrid[tPix1]<<" -> "<<cpuImgGrid[tPix1]<<", model = "<<cpuModelGrid[tPix1]<< endl;
            cout << "    pix 2: "<<cpulmgrid[tPix2]<<" -> "<<cpuImgGrid[tPix2]<<", model = "<<cpuModelGrid[tPix2]<< endl;

            // Report on timings
            cout << "    Time " << time << " (s) " << endl;
            cout << "    Time per cycle " << time / nMinor * 1000 << " (ms)" << endl;
            cout << "    Cleaning rate  " << nMinor / time << " (iterations per second)" << endl;
        }

        std::vector<std::complex<float> > cpulmRes(cpuImgGrid);

        cpuImgGrid = cpuModelGrid;

        //-------------------------------------------------------------------//
        // FFT deconvolved model image for degridding
        {
            cout << "Forward FFT" << endl;

            Stopwatch sw;
            sw.start();
            // this should be the model, not cpuImgGrid
            if ( fftExec(cpuImgGrid, gSize, true) != 0 ) {
                cout << "forward fftExec error" << endl;
                return -1;
            }
            time = sw.stop();

            // Report on timings
            cout << "    Time " << time << " (s) " << endl;
        }

        //-----------------------------------------------------------------------//
        // DO DEGRIDDING
        {
            // Now we can do the timing for the CPU implementation
            cout << "Degridding data" << endl;

            Stopwatch sw;
            sw.start();
            degridKernel(cpuImgGrid, gSize, support, C, cOffset, iu, iv, cpuModel);
            time = sw.stop();

            // Report on timings
            cout << "    Time " << time << " (s) " << endl;
            cout << "    Time per visibility sample " << 1e6*time / double(cpuData.size()) << " (us) " << endl;
            cout << "    Time per degridding   " << 1e9*time / double(cpuData.size()*sSize*sSize) << " (ns) " << endl;
            cout << "    Degridding rate   " << griddings/1e6/time << " (million grid points per second)" << endl;
        }

        double cpu_time = sw_cpu.stop();
        cout << "CPU single core took " << cpu_time << " (s)" << endl;

#endif

        ///////////////////////////////////////////////////////////////////////////
        // OpenACC
        ///////////////////////////////////////////////////////////////////////////
        cout << endl << "+++++ OpenACC +++++" << endl << endl;

        Stopwatch sw_acc;
        sw_acc.start();

        //-----------------------------------------------------------------------//
        // DO GRIDDING
        std::vector<std::complex<float> > accpsfgrid(gSize*gSize);
        std::vector<std::complex<float> > accImgGrid(gSize*gSize);
        //accpsfgrid.assign(accpsfgrid.size(), std::complex<float>(0.0));
        //accImgGrid.assign(accImgGrid.size(), std::complex<float>(0.0));

        // later: do this in a parallel loop...
        std::complex<float> *accpsfgrid_d = accpsfgrid.data();
        std::complex<float> *accImgGrid_d = accImgGrid.data();
        if (it_major==0) {
            //#pragma acc enter data copyin(accpsfgrid_d[0:gSize*gSize], accImgGrid_d[0:gSize*gSize]) async(1)
            #pragma acc enter data create(accpsfgrid_d[0:gSize*gSize], accImgGrid_d[0:gSize*gSize])
        }
        #pragma acc parallel loop present(accpsfgrid_d[0:gSize*gSize], accImgGrid_d[0:gSize*gSize])
        for (unsigned int i = 0; i < gSize*gSize; ++i) {
            accpsfgrid_d[i] = 0.0;
            accImgGrid_d[i] = 0.0;
        }

        {
            // Now we can do the timing for the GPU implementation
            cout << "Gridding PSF" << endl;

            // Time is measured inside this function call, unlike the CPU versions
            Stopwatch sw;
            sw.start();
            gridKernelACC(accData, support, C, cOffset, iu, iv, accpsfgrid, gSize, true);
            const double acctime = sw.stop();

            // Report on timings
            //cout << "    Time " << acctime << " (s) = serial version / " << time/acctime << endl;
            cout << "    Time " << acctime << " (s)" << endl;
            cout << "    Time per visibility sample " << 1e6*acctime / double(accData.size()) << " (us) " << endl;
            cout << "    Time per gridding   " << 1e9*acctime / double(accData.size()*sSize*sSize) << " (ns) " << endl;
            cout << "    Gridding rate   " << griddings/1e6/acctime << " (million grid points per second)" << endl;
        }
        {
            cout << "Gridding data" << endl;

            // Time is measured inside this function call, unlike the CPU versions
            Stopwatch sw;
            sw.start();
            gridKernelACC(accData, support, C, cOffset, iu, iv, accImgGrid, gSize, false);
            const double acctime = sw.stop();

            // Report on timings
            //cout << "    Time " << acctime << " (s) = serial version / " << time/acctime << endl;
            cout << "    Time " << acctime << " (s)" << endl;
            cout << "    Time per visibility sample " << 1e6*acctime / double(accData.size()) << " (us) " << endl;
            cout << "    Time per gridding   " << 1e9*acctime / double(accData.size()*sSize*sSize) << " (ns) " << endl;
            cout << "    Gridding rate   " << griddings/1e6/acctime << " (million grid points per second)" << endl;
        }

#ifdef RUN_VERIFY
        #pragma acc update host(accpsfgrid_d[0:gSize*gSize], accImgGrid_d[0:gSize*gSize])
        std::vector<std::complex<float> > accuvPsf(accpsfgrid);
        std::vector<std::complex<float> > accuvGrid(accImgGrid);
#endif

        //-----------------------------------------------------------------------//
        // Form dirty image and run the minor cycle

        std::complex<float> testvalue = accImgGrid.data()[tPix1];

        // use cufft

        testvalue *= (float)(gSize*gSize);

        // FFT gridded data to form dirty image
        {
            cout << "Inverse FFTs" << endl;

            Stopwatch sw;
            sw.start();
            #ifdef GPU
            if ( fftExecGPU(accpsfgrid, gSize, false) != 0 ) {
                cout << "inverse fftExecGPU error" << endl;
                return -1;
            }
            fftFixGPU(accpsfgrid, 1.0/float(accData.size()));
 
            if ( fftExecGPU(accImgGrid, gSize, false) != 0 ) {
                cout << "inverse fftExecGPU error" << endl;
                return -1;
            }
            fftFixGPU(accImgGrid, 1.0/float(accData.size()));
            #else
            // note that we should really enable multithreaded FFTW in this situation
            if ( fftExec(accpsfgrid, gSize, false) != 0 ) {
                cout << "inverse fftExec error" << endl;
                return -1;
            }
            fftFix(accpsfgrid, 1.0/float(cpuData.size()));
 
            if ( fftExec(accImgGrid, gSize, false) != 0 ) {
                cout << "inverse fftExec error" << endl;
                return -1;
            }
            fftFix(accImgGrid, 1.0/float(cpuData.size()));
            #endif
            const double acctime = sw.stop();

            // Report on timings
            cout << "    Time " << acctime << " (s)" << endl;

        }

#ifdef RUN_VERIFY
        #pragma acc update host(accpsfgrid_d[0:gSize*gSize], accImgGrid_d[0:gSize*gSize])
        std::vector<std::complex<float> > acclmPsf(accpsfgrid);
        std::vector<std::complex<float> > acclmgrid(accImgGrid);
#endif

        //-------------------------------------------------------------------//
        // Do Hogbom CLEAN

        std::vector<std::complex<float> > accModelGrid(accpsfgrid.size());
        accModelGrid.assign(accModelGrid.size(), std::complex<float>(0.0));

        {
            // Now we can do the timing for the serial (Golden) CPU implementation
            cout << "Hogbom clean" << endl;

            Stopwatch sw;
            sw.start();
            deconvolveACC(accImgGrid, gSize, accpsfgrid, gSize, accModelGrid, nMinor);
            time = sw.stop();

#ifdef RUN_VERIFY
            // Report on results
            #pragma acc update host(accImgGrid_d[0:gSize*gSize])
            cout << "    pix 1: "<<acclmgrid[tPix1]<<" -> "<<accImgGrid[tPix1]<<", model = "<<accModelGrid[tPix1]<< endl;
            cout << "    pix 2: "<<acclmgrid[tPix2]<<" -> "<<accImgGrid[tPix2]<<", model = "<<accModelGrid[tPix2]<< endl;
#endif

            // Report on timings
            cout << "    Time " << time << " (s) " << endl;
            cout << "    Time per cycle " << time / nMinor * 1000 << " (ms)" << endl;
            cout << "    Cleaning rate  " << nMinor / time << " (iterations per second)" << endl;
        }

#ifdef RUN_VERIFY
        #pragma acc update host(accImgGrid_d[0:gSize*gSize])
        std::vector<std::complex<float> > acclmRes(accImgGrid);
#endif

        accImgGrid = accModelGrid;
        #pragma acc update device(accImgGrid_d[0:gSize*gSize])

        //-------------------------------------------------------------------//
        // FFT deconvolved model image for degridding
        {
            cout << "Forward FFT" << endl;

            Stopwatch sw;
            sw.start();
            #ifdef GPU
            if ( fftExecGPU(accImgGrid, gSize, true) != 0 ) {
                cout << "forward fftExecGPU error" << endl;
                return -1;
            }
            #else
            // note that we should really enable multithreaded FFTW in this situation
            if ( fftExec(accImgGrid, gSize, true) != 0 ) {
                cout << "forward fftExec error" << endl;
                return -1;
            }
            #endif
            const double acctime = sw.stop();

            // Report on timings
            cout << "    Time " << acctime << " (s)" << endl;

        }

        //-----------------------------------------------------------------------//
        // DO DEGRIDDING
        {
            // Now we can do the timing for the GPU implementation
            cout << "Degridding data" << endl;

            // Time is measured inside this function call, unlike the CPU versions
            Stopwatch sw;
            sw.start();
            degridKernelACC(accImgGrid, gSize, support, C, cOffset, iu, iv, accModel);
            const double acctime = sw.stop();

            // Report on timings
            //cout << "    Time " << acctime << " (s) = serial version / " << time/acctime << endl;
            cout << "    Time " << acctime << " (s)" << endl;
            cout << "    Time per visibility sample " << 1e6*acctime / double(accData.size()) << " (us) " << endl;
            cout << "    Time per degridding   " << 1e9*acctime / double(accData.size()*sSize*sSize) << " (ns) " << endl;
            cout << "    Degridding rate   " << griddings/1e6/time << " (million grid points per second)" << endl;
        }

        //-----------------------------------------------------------------------//
        // Copy GPU data back to CPU

        //#pragma acc exit data copyout(accImgGrid_d[0:gSize*gSize],accModel_d[0:nSamples*nChan]) 
        #pragma acc update host(accImgGrid_d[0:gSize*gSize],accModel_d[0:nSamples*nChan])

        double acc_time = sw_acc.stop();
        cout << "OpenACC took " << acc_time << " (s)" << endl;

#ifdef RUN_VERIFY

        ///////////////////////////////////////////////////////////////////////////
        // Verify results
        ///////////////////////////////////////////////////////////////////////////
        cout << endl << "+++++ Verifying results +++++" << endl << endl;

        //-----------------------------------------------------------------------//
        // Verify gridding results
        cout << "Gridding PSF: ";

        if (cpuuvPsf.size() != accuvPsf.size()) {
            cout << "Fail (Grid sizes differ)" << std::endl;
            return 1;
        }

        for (unsigned int i = 0; i < cpuuvPsf.size(); ++i) {
            if (fabs(cpuuvPsf[i].real() - accuvPsf[i].real()) > 0.00001) {
                cout << "Fail (Expected " << cpuuvPsf[i].real() << " got "
                         << accuvPsf[i].real() << " at index " << i << ")"
                         << std::endl;
                return 1;
            }
        }

        result = std::max_element(cpulmPsf.begin(), cpulmPsf.end(), abs_compare);
        int psfPixel = std::distance(cpulmPsf.begin(), result);

        cout << "Pass" << std::endl;

        cout << "Gridding data: ";

        if (cpuuvGrid.size() != accuvGrid.size()) {
            cout << "Fail (Grid sizes differ)" << std::endl;
            return 1;
        }

        for (unsigned int i = 0; i < cpuuvGrid.size(); ++i) {
            if (fabs(cpuuvGrid[i].real() - accuvGrid[i].real()) > 0.00001) {
                cout << "Fail (Expected " << cpuuvGrid[i].real() << " got "
                         << accuvGrid[i].real() << " at index " << i << ")"
                         << std::endl;
                return 1;
            }
        }

        cout << "Pass" << std::endl;

        //-----------------------------------------------------------------------//
        // Verify Inverse FFT results
        cout << "Inverse FFT (PSF): ";

        if (cpulmPsf.size() != acclmPsf.size()) {
            cout << "Fail (Grid sizes differ)" << std::endl;
            return 1;
        }

        for (unsigned int i = 0; i < cpulmPsf.size(); ++i) {
            if (fabs(cpulmPsf[i].real() - acclmPsf[i].real()) / fabs(cpulmPsf[psfPixel].real()) > 0.00001) {
                cout << "Fail (Expected " << cpulmPsf[i].real() << " got "
                         << acclmPsf[i].real() << " at index " << i << ")"
                         << std::endl;
                return 1;
            }
        }

        cout << "Pass" << std::endl;

        cout << "Inverse FFT: ";

        if (cpulmgrid.size() != acclmgrid.size()) {
            cout << "Fail (Grid sizes differ)" << std::endl;
            return 1;
        }

        for (unsigned int i = 0; i < cpulmgrid.size(); ++i) {
            if (fabs(cpulmgrid[i].real() - acclmgrid[i].real()) / fabs(cpulmPsf[psfPixel].real()) > 0.00001) {
                cout << "Fail (Expected " << cpulmgrid[i].real() << " got "
                         << acclmgrid[i].real() << " at index " << i << ")"
                         << std::endl;
                return 1;
            }
        }

        cout << "Pass" << std::endl;

        //cout << cpulmgrid[tPix1].real() << "/psf = " << cpulmgrid[tPix1].real()/fabs(cpulmPsf[psfPixel].real()) << endl;
        //cout << cpulmgrid[tPix2].real() << "/psf = " << cpulmgrid[tPix2].real()/fabs(cpulmPsf[psfPixel].real()) << endl;

        //-----------------------------------------------------------------------//
        // Verify Hogbom clean results
        cout << "Hogbom clean: ";

        if (cpulmRes.size() != acclmRes.size()) {
            cout << "Fail (Grid sizes differ)" << std::endl;
            return 1;
        }

        for (unsigned int i = 0; i < cpulmRes.size(); ++i) {
            if (fabs(cpulmRes[i].real() - acclmRes[i].real()) / fabs(cpulmPsf[psfPixel].real()) > 0.00001) {
                cout << "Fail (Expected " << cpulmRes[i].real() << " got "
                         << acclmRes[i].real() << " at index " << i << ")"
                         << std::endl;
                return 1;
            }
        }

        if (cpuModelGrid.size() != accModelGrid.size()) {
            cout << "Fail (Grid sizes differ)" << std::endl;
            return 1;
        }

        for (unsigned int i = 0; i < cpuModelGrid.size(); ++i) {
            if (fabs(cpuModelGrid[i].real() - accModelGrid[i].real()) / fabs(cpulmPsf[psfPixel].real()) > 0.00001) {
                cout << "Fail (Expected " << cpuModelGrid[i].real() << " got "
                         << accModelGrid[i].real() << " at index " << i << ")"
                         << std::endl;
                return 1;
            }
        }

        cout << "Pass" << std::endl;

        //-----------------------------------------------------------------------//
        // Verify Forward FFT results
        cout << "Forward FFT: ";

        if (cpuImgGrid.size() != accImgGrid.size()) {
            cout << "Fail (Grid sizes differ)" << std::endl;
            return 1;
        }

        result = std::max_element(cpuImgGrid.begin(), cpuImgGrid.end(), abs_compare);
        int cpuPixel = std::distance(cpuImgGrid.begin(), result);

        for (unsigned int i = 0; i < cpuImgGrid.size(); ++i) {
            if (fabs(cpuImgGrid[i].real() - accImgGrid[i].real()) / fabs(cpuImgGrid[cpuPixel].real()) > 0.00001) {
                cout << "Fail (Expected " << cpuImgGrid[i].real() << " got "
                         << accImgGrid[i].real() << " at index " << i << ")"
                         << std::endl;
                return 1;
            }
        }

        cout << "Pass" << std::endl;

        //-----------------------------------------------------------------------//
        // degridding results
        cout << "Reverse processing: ";

        if (cpuModel.size() != accModel.size()) {
            cout << "Fail (Data vector sizes differ)" << std::endl;
            return 1;
        }

        for (unsigned int i = 0; i < cpuModel.size(); ++i) {
            if (fabs(cpuModel[i].real() - accModel[i].real()) > 0.00001) {
                cout << "Fail (Expected " << cpuModel[i].real() << " got "
                         << accModel[i].real() << " at index " << i << ")"
                         << std::endl;
                return 1;
            }
        }

        cout << "Pass" << std::endl;
#endif

        // subtract the model vis and cycle back
        for (unsigned int i = 0; i < nSamples*nChan; ++i) {
            cpuData[i] = cpuData[i] - cpuModel[i];
        }

        #pragma acc parallel loop present(accData_d[0:nSamples*nChan],accModel_d[0:nSamples*nChan])
        for (unsigned int i = 0; i < nSamples*nChan; ++i) {
            accData_d[i] = accData_d[i] - accModel_d[i];
        }

    } // it_major

    return 0;
}
