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

// Local includes
#include "../tConvolveCommon/common.h"

// OpenACC includes
#include <openacc.h>

// CUDA includes
#ifdef GPU
#include <cufft.h>
#endif

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
void gridKernel(const std::vector<Value>& data, const int support,
                const std::vector<Value>& C, const std::vector<int>& cOffset,
                const std::vector<int>& iu, const std::vector<int>& iv,
                std::vector<Value>& grid, const int gSize)
{
    const int sSize = 2 * support + 1;

    Value *d_grid = grid.data();
    const Value *d_data = data.data();
    const Value *d_C = C.data();

    for (int dind = 0; dind < int(data.size()); ++dind) {
        // The actual grid point
        int gind = iu[dind] + gSize * iv[dind] - support;
        // The Convoluton function point from which we offset
        int cind = cOffset[dind];
        int suppu, suppv;
        const Real dre = d_data[dind].real();
        const Real dim = d_data[dind].imag();

        for (suppv = 0; suppv < sSize; suppv++) {
            for (suppu = 0; suppu < sSize; suppu++) {
                Real *dref = (Real *)&d_grid[gind+suppv*gSize+suppu];
                const int supp = cind + suppv*sSize+suppu;
                const Real reval = dre * d_C[supp].real() - dim * d_C[supp].imag();
                const Real imval = dim * d_C[supp].real() + dre * d_C[supp].imag();
                dref[0] = dref[0] + reval;
                dref[1] = dref[1] + imval;
            }
        }

    }
}

void gridKernelACC(const std::vector<Value>& data, const int support,
        const std::vector<Value>& C, const std::vector<int>& cOffset,
        const std::vector<int>& iu, const std::vector<int>& iv,
        std::vector<Value>& grid, const int gSize)
{
    const int sSize = 2 * support + 1;

    // Value = std::complex<Real> = std::complex<float>
    //Real *d_grid = (Real *)grid.data();
    Value *d_grid = grid.data();
    const int d_size = data.size();
    const Value *d_data = data.data();
    const Value *d_C = C.data();
    const int *d_cOffset = cOffset.data();
    const int *d_iu = iu.data();
    const int *d_iv = iv.data();

    int dind;
    // Both of the following approaches are the same when running on multicore CPUs.
    // Using "gang vector" here without the inner pragma below sets 1 vis per vector element (c.f. CUDA thread)
    //#pragma acc parallel loop gang vector
    // Using "gang" here with the inner pragma below sets 1 vis per gang, and 1 pixel per vec element

#ifdef GPU
    #pragma acc parallel loop
#endif
    for (dind = 0; dind < d_size; ++dind) {

        int cind = d_cOffset[dind];
        //const Real *c_C = (Real *)&d_C[cind];
        //#pragma acc cache(c_C[0:2*sSize*sSize])

        // The actual grid point
        int gind = d_iu[dind] + gSize * d_iv[dind] - support;
        // The Convoluton function point from which we offset
        int suppu, suppv;
        const Real dre = d_data[dind].real();
        const Real dim = d_data[dind].imag();

#ifdef GPU
        #pragma acc loop collapse(2)
        for (suppv = 0; suppv < sSize; suppv++) {
            for (suppu = 0; suppu < sSize; suppu++) {
                Real *dref = (Real *)&d_grid[gind+suppv*gSize+suppu];
                const std::complex<Real> cval = d_data[dind] * d_C[cind+suppv*sSize+suppu];
                #pragma acc atomic update
                dref[0] = dref[0] + cval.real();
                #pragma acc atomic update
                dref[1] = dref[1] + cval.imag();
            }
        }
#else
        #pragma acc parallel loop gang vector collapse(2)
        for (suppv = 0; suppv < sSize; suppv++) {
            for (suppu = 0; suppu < sSize; suppu++) {
                Real *dref = (Real *)&d_grid[gind+suppv*gSize+suppu];
                //const int suppre = 2 * (suppv*sSize+suppu);
                const int supp = cind + suppv*sSize + suppu;
                const Real reval = dre * d_C[supp].real() - dim * d_C[supp].imag();
                const Real imval = dim * d_C[supp].real() + dre * d_C[supp].imag();
                dref[0] = dref[0] + reval;
                dref[1] = dref[1] + imval;
            }
        }
#endif

    }

}

// Perform degridding
void degridKernel(const std::vector<Value>& grid, const int gSize, const int support,
                  const std::vector<Value>& C, const std::vector<int>& cOffset,
                  const std::vector<int>& iu, const std::vector<int>& iv,
                  std::vector<Value>& data)
{
    const int sSize = 2 * support + 1;

    Value *d_data = data.data();
    const Value *d_grid = grid.data();
    const Value *d_C = C.data();

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
        d_data[dind] = Value(re,im);

    }
}

void degridKernelACC(const std::vector<Value>& grid, const int gSize, const int support,
                     const std::vector<Value>& C, const std::vector<int>& cOffset,
                     const std::vector<int>& iu, const std::vector<int>& iv,
                     std::vector<Value>& data)
{
    const int sSize = 2 * support + 1;

    const int d_size = data.size();
    Value *d_data = data.data();
    const Value *d_grid = grid.data();
    const Value *d_C = C.data();
    const int *d_cOffset = cOffset.data();
    const int *d_iu = iu.data();
    const int *d_iv = iv.data();

    int dind;
    // Both of the following approaches are the same when running on multicore CPUs.
    // Using "gang vector" here without the inner pragma below sets 1 vis per vector element (c.f. CUDA thread)
    //#pragma acc parallel loop gang vector
    // Using "gang" here with the inner pragma below sets 1 vis per gang, and 1 pixel per vec element

    #pragma acc parallel loop
    for (dind = 0; dind < d_size; ++dind) {

        // The actual grid point from which we offset
        int gind = d_iu[dind] + gSize * d_iv[dind] - support;
        // The Convoluton function point from which we offset
        int cind = d_cOffset[dind];
        float re = 0.0, im = 0.0;

#ifdef GPU
        #pragma acc loop reduction(+:re,im) collapse(2)
        for (int suppv = 0; suppv < sSize; suppv++) {
            for (int suppu = 0; suppu < sSize; suppu++) {
                const Value cval = d_grid[gind+suppv*gSize+suppu] * d_C[cind+suppv*sSize+suppu];
                re = re + cval.real();
                im = im + cval.imag();
            }
        }
#else
        for (int suppv = 0; suppv < sSize; suppv++) {
            for (int suppu = 0; suppu < sSize; suppu++) {
                re = re + d_grid[gind+suppv*gSize+suppu].real() * d_C[cind+suppv*sSize+suppu].real() -
                          d_grid[gind+suppv*gSize+suppu].imag() * d_C[cind+suppv*sSize+suppu].imag();
                im = im + d_grid[gind+suppv*gSize+suppu].imag() * d_C[cind+suppv*sSize+suppu].real() +
                          d_grid[gind+suppv*gSize+suppu].real() * d_C[cind+suppv*sSize+suppu].imag();
            }
        }
#endif

        d_data[dind] = Value(re,im);

    }

}


// Main testing routine
int main(int argc, char* argv[])
{
    Options opt;
    getinput(argc, argv, opt);
    // Change these if necessary to adjust run time
    int nSamples = opt.nSamples;
    int wSize = opt.wSize;
    int nChan = opt.nChan;
    Coord cellSize = opt.cellSize;
    const int gSize = opt.gSize;
    const int baseline = opt.baseline; 

    // Initialize the data to be gridded
    std::vector<Coord> u(nSamples);
    std::vector<Coord> v(nSamples);
    std::vector<Coord> w(nSamples);
    std::vector<Value> data(nSamples*nChan);
    std::vector<Value> cpuoutdata(nSamples*nChan);
    std::vector<Value> accoutdata(nSamples*nChan);

    const unsigned int maxint = std::numeric_limits<int>::max();

    for (int i = 0; i < nSamples; i++) {
        u[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;
        v[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;
        w[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;

        for (int chan = 0; chan < nChan; chan++) {
            data[i*nChan+chan] = 1.0;
            cpuoutdata[i*nChan+chan] = 0.0;
            accoutdata[i*nChan+chan] = 0.0;
        }
    }

    std::vector<Value> grid(gSize*gSize);
    grid.assign(grid.size(), Value(0.0));

    // Measure frequency in inverse wavelengths
    std::vector<Coord> freq(nChan);

    for (int i = 0; i < nChan; i++) {
        freq[i] = (1.4e9 - 2.0e5 * Coord(i) / Coord(nChan)) / 2.998e8;
    }

    // Initialize convolution function and offsets
    std::vector<Value> C;
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

    const double griddings = (double(nSamples * nChan) * double((sSize) * (sSize)));

    double time;

    ///////////////////////////////////////////////////////////////////////////
    // DO GRIDDING
    ///////////////////////////////////////////////////////////////////////////
    std::vector<Value> cpugrid(gSize*gSize);
    cpugrid.assign(cpugrid.size(), Value(0.0));
    {
        // Now we can do the timing for the CPU implementation
        cout << "+++++ Forward processing (CPU single core) +++++" << endl;

        Stopwatch sw;
        sw.start();
        gridKernel(data, support, C, cOffset, iu, iv, cpugrid, gSize);
        time = sw.stop();

        // Report on timings
        cout << "    Time " << time << " (s) " << endl;
        cout << "    Time per visibility spectral sample " << 1e6*time / double(data.size()) << " (us) " << endl;
        cout << "    Time per gridding   " << 1e9*time / (double(data.size())* double((sSize)*(sSize))) << " (ns) " << endl;
        cout << "    Gridding rate   " << (griddings / 1000000) / time << " (million grid points per second)" << endl;

        cout << "Done" << endl;
    }

    std::vector<Value> accgrid(gSize*gSize);
    accgrid.assign(accgrid.size(), Value(0.0));
    {
        // Now we can do the timing for the GPU implementation
        cout << "+++++ Forward processing (OpenACC) +++++" << endl;

        // Time is measured inside this function call, unlike the CPU versions
        Stopwatch sw;
        sw.start();
        gridKernelACC(data, support, C, cOffset, iu, iv, accgrid, gSize);
        const double acctime = sw.stop();

        // Report on timings
        cout << "    Time " << acctime << " (s) = serial version / " << time/acctime << endl;
        cout << "    Time per visibility spectral sample " << 1e6*acctime / double(data.size()) << " (us) " << endl;
        cout << "    Time per gridding   " << 1e9*acctime / (double(data.size())* double((sSize)*(sSize))) << " (ns) " << endl;
        cout << "    Gridding rate   " << (griddings / 1000000) / acctime << " (million grid points per second)" << endl;

        cout << "Done" << endl;
    }

    cout << "Verifying result...";

    if (cpugrid.size() != accgrid.size()) {
        cout << "Fail (Grid sizes differ)" << std::endl;
        return 1;
    }

    for (unsigned int i = 0; i < cpugrid.size(); ++i) {
        if (fabs(cpugrid[i].real() - accgrid[i].real()) > 0.00001) {
            cout << "Fail (Expected " << cpugrid[i].real() << " got "
                     << accgrid[i].real() << " at index " << i << ")"
                     << std::endl;
            return 1;
        }
    }

    cout << "Pass" << std::endl;

    ///////////////////////////////////////////////////////////////////////////
    // DO IFFT THEN BACK WITH FFT
    ///////////////////////////////////////////////////////////////////////////

/*
 * getting a bus error on Hyperion's GTX Titan, so comment out for now.
 *

    int testpoint = sSize/2*sSize+sSize/2;
    Value testvalue = data.data()[testpoint];

    #ifdef GPU

    // use cufft

    testvalue *= (Real)(sSize*sSize);

    cufftHandle plan;

    if ( cufftPlan2d( &plan, sSize, sSize, CUFFT_C2C ) != CUFFT_SUCCESS ) {
        cout << "CUFFT error: Plan creation failed" << endl;
        return 1;
    }

    Real *d_data = (Real *)data.data();
    #pragma acc host_data use_device(d_data)
    {
        // FFT gridded data to form dirty image
        cout << "+++++ Forward FFT (CUFFT) +++++" << endl;
        cufftResult fftErr;
        fftErr = cufftExecC2C(plan, (cufftComplex*)d_data, (cufftComplex*)d_data, CUFFT_FORWARD);
        if ( fftErr != CUFFT_SUCCESS ) {
            cout << "CUFFT error: Forward FFT failed" << endl;
            return 1;
        }
        cout << "Done" << endl;

        // <INSERT DECONVOLUTION HERE>

        // FFT deconvolved model image for degridding. Just use data for now
        cout << "+++++ Inverse FFT (CUFFT) +++++" << endl;
        fftErr = cufftExecC2C(plan, (cufftComplex*)d_data, (cufftComplex*)d_data, CUFFT_INVERSE);
        if ( fftErr != CUFFT_SUCCESS ) {
            cout << "CUFFT error: Inverse FFT failed" << endl;
            return 1;
        }
        cout << "Done" << endl;
    }

    cufftDestroy(plan);

    #else

    // use fftw

    #endif

    // Verify FFT results
    cout << "Verifying result...";

    if ( abs(testvalue - data.data()[testpoint]) > 1e-3 ) {
        cout << "Fail (" << data.data()[testpoint] << " != " << testvalue << ")" << endl;
        cout << " - " << testvalue - data.data()[testpoint] << endl;
        cout << " - " << abs(testvalue - data.data()[testpoint]) << endl;
        return 1;
    }

    cout << "Pass" << std::endl;

*/

    ///////////////////////////////////////////////////////////////////////////
    // DO DEGRIDDING
    ///////////////////////////////////////////////////////////////////////////
    {
        cpugrid.assign(cpugrid.size(), Value(1.0));
        // Now we can do the timing for the CPU implementation
        cout << "+++++ Reverse processing (CPU with complex mult) +++++" << endl;

        Stopwatch sw;
        sw.start();
        degridKernel(cpugrid, gSize, support, C, cOffset, iu, iv, cpuoutdata);
        time = sw.stop();

        // Report on timings
        cout << "    Time " << time << " (s) " << endl;
        cout << "    Time per visibility spectral sample " << 1e6*time / double(data.size()) << " (us) " << endl;
        cout << "    Time per degridding   " << 1e9*time / (double(data.size())* double((sSize)*(sSize))) << " (ns) " << endl;
        cout << "    Degridding rate   " << (griddings / 1000000) / time << " (million grid points per second)" << endl;

        cout << "Done" << endl;
    }

    {
        accgrid.assign(accgrid.size(), Value(1.0));
        // Now we can do the timing for the GPU implementation
        cout << "+++++ Reverse processing (OpenACC) +++++" << endl;

        // Time is measured inside this function call, unlike the CPU versions
        Stopwatch sw;
        sw.start();
        degridKernelACC(accgrid, gSize, support, C, cOffset, iu, iv, accoutdata);
        const double acctime = sw.stop();

        // Report on timings
        cout << "    Time " << acctime << " (s) = serial version / " << time/acctime << endl;
        cout << "    Time per visibility spectral sample " << 1e6*acctime / double(data.size()) << " (us) " << endl;
        cout << "    Time per degridding   " << 1e9*acctime / (double(data.size())* double((sSize)*(sSize))) << " (ns) " << endl;
        cout << "    Degridding rate   " << (griddings / 1000000) / acctime << " (million grid points per second)" << endl;

        cout << "Done" << endl;
    }

    // Verify degridding results
    cout << "Verifying result...";

    if (cpuoutdata.size() != accoutdata.size()) {
        cout << "Fail (Data vector sizes differ)" << std::endl;
        return 1;
    }

    for (unsigned int i = 0; i < cpuoutdata.size(); ++i) {
        if (fabs(cpuoutdata[i].real() - accoutdata[i].real()) > 0.00001) {
            cout << "Fail (Expected " << cpuoutdata[i].real() << " got "
                     << accoutdata[i].real() << " at index " << i << ")"
                     << std::endl;
            return 1;
        }
    }

    cout << "Pass" << std::endl;

    return 0;
}
