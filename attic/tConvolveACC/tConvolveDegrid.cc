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
/// @author Daniel Mitchell <daniel.mitchell@csiro.au>

// Local includes
#include "../tConvolveCommon/common.h"

// OpenACC includes
#include <openacc.h>

// CUDA includes
#ifdef GPU
#include <cufft.h>
#endif


void degridKernelReductionReal(const std::vector<Value>& grid, const int gSize, const int support,
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
    #pragma acc parallel loop
    for (dind = 0; dind < d_size; ++dind) {

        // The actual grid point from which we offset
        int gind = d_iu[dind] + gSize * d_iv[dind] - support;
        // The Convoluton function point from which we offset
        int cind = d_cOffset[dind];

        float re = 0.0, im = 0.0;
        #pragma acc loop reduction(+:re,im) collapse(2)
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

void degridKernelReductionComplex(const std::vector<Value>& grid, const int gSize, const int support,
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
    #pragma acc parallel loop
    for (dind = 0; dind < d_size; ++dind) {

        // The actual grid point from which we offset
        int gind = d_iu[dind] + gSize * d_iv[dind] - support;
        // The Convoluton function point from which we offset
        int cind = d_cOffset[dind];

        float re = 0.0, im = 0.0;
        #pragma acc loop reduction(+:re,im) collapse(2)
        for (int suppv = 0; suppv < sSize; suppv++) {
            for (int suppu = 0; suppu < sSize; suppu++) {
                const Value cval = d_grid[gind+suppv*gSize+suppu] * d_C[cind+suppv*sSize+suppu];
                re = re + cval.real();
                im = im + cval.imag();
            }
        }
        d_data[dind] = Value(re,im);

    }

}

void degridKernelDataLoopReal(const std::vector<Value>& grid, const int gSize, const int support,
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
    #pragma acc parallel loop gang vector
    for (dind = 0; dind < d_size; ++dind) {

        // The actual grid point from which we offset
        int gind = d_iu[dind] + gSize * d_iv[dind] - support;
        // The Convoluton function point from which we offset
        int cind = d_cOffset[dind];

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

void degridKernelDataLoopComplex(const std::vector<Value>& grid, const int gSize, const int support,
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
    #pragma acc parallel loop gang vector
    for (dind = 0; dind < d_size; ++dind) {

        // The actual grid point from which we offset
        int gind = d_iu[dind] + gSize * d_iv[dind] - support;
        // The Convoluton function point from which we offset
        int cind = d_cOffset[dind];

        Value cmplx = 0.0;

        for (int suppv = 0; suppv < sSize; suppv++) {
            for (int suppu = 0; suppu < sSize; suppu++) {
                cmplx = cmplx + d_grid[gind+suppv*gSize+suppu] * d_C[cind+suppv*sSize+suppu];
            }
        }
        d_data[dind] = cmplx;

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
    std::vector<Value> outdata0(nSamples*nChan);
    std::vector<Value> outdata1(nSamples*nChan);

    const unsigned int maxint = std::numeric_limits<int>::max();

    for (int i = 0; i < nSamples; i++) {
        u[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;
        v[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;
        w[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;

        for (int chan = 0; chan < nChan; chan++) {
            data[i*nChan+chan] = 1.0;
            outdata0[i*nChan+chan] = 0.0;
            outdata1[i*nChan+chan] = 0.0;
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

    double timeReal, time;

    std::vector<Value> cpugrid(gSize*gSize);

    ///////////////////////////////////////////////////////////////////////////
    // DO DEGRIDDING
    ///////////////////////////////////////////////////////////////////////////
    {
        cpugrid.assign(cpugrid.size(), Value(1.0));
        // Now we can do the timing for the CPU implementation
        cout << "+++++ Reverse processing (CPU with real mult & reduction) +++++" << endl;

        Stopwatch sw;
        sw.start();
        degridKernelReductionReal(cpugrid, gSize, support, C, cOffset, iu, iv, outdata0);
        timeReal = sw.stop();

        // Report on timings
        cout << "    Time " << timeReal << " (s) " << endl;
        cout << "    Time per visibility spectral sample " << 1e6*timeReal / double(data.size()) << " (us) " << endl;
        cout << "    Time per degridding   " << 1e9*timeReal / (double(data.size())* double((sSize)*(sSize))) << " (ns) " << endl;
        cout << "    Degridding rate   " << (griddings / 1000000) / timeReal << " (million grid points per second)" << endl;

        cout << "Done" << endl;
    }

    {
        cpugrid.assign(cpugrid.size(), Value(1.0));
        // Now we can do the timing for the GPU implementation
        cout << "+++++ Reverse processing (OpenACC with complex mult & reduction) +++++" << endl;

        // Time is measured inside this function call, unlike the CPU versions
        Stopwatch sw;
        sw.start();
        degridKernelReductionComplex(cpugrid, gSize, support, C, cOffset, iu, iv, outdata1);
        time = sw.stop();

        // Report on timings
        cout << "    Time " << time << " (s) = real mult / " << timeReal/time << endl;
        cout << "    Time per visibility spectral sample " << 1e6*time / double(data.size()) << " (us) " << endl;
        cout << "    Time per degridding   " << 1e9*time / (double(data.size())* double((sSize)*(sSize))) << " (ns) " << endl;
        cout << "    Degridding rate   " << (griddings / 1000000) / time << " (million grid points per second)" << endl;

        cout << "Done" << endl;
    }

    // Verify degridding results
    cout << "Verifying result...";

    if (outdata0.size() != outdata1.size()) {
        cout << "Fail (Data vector sizes differ)" << std::endl;
        return 1;
    }

    for (unsigned int i = 0; i < outdata0.size(); ++i) {
        if (fabs(outdata0[i].real() - outdata1[i].real()) > 0.00001) {
            cout << "Fail (Expected " << outdata0[i].real() << " got "
                     << outdata1[i].real() << " at index " << i << ")"
                     << std::endl;
            return 1;
        }
    }

    cout << "Pass" << std::endl;

    {
        cpugrid.assign(cpugrid.size(), Value(1.0));
        // Now we can do the timing for the GPU implementation
        cout << "+++++ Reverse processing (OpenACC with real data loop) +++++" << endl;

        // Time is measured inside this function call, unlike the CPU versions
        Stopwatch sw;
        sw.start();
        degridKernelDataLoopReal(cpugrid, gSize, support, C, cOffset, iu, iv, outdata1);
        timeReal = sw.stop();

        // Report on timings
        cout << "    Time " << timeReal << " (s)" << endl;
        cout << "    Time per visibility spectral sample " << 1e6*timeReal / double(data.size()) << " (us) " << endl;
        cout << "    Time per degridding   " << 1e9*timeReal / (double(data.size())* double((sSize)*(sSize))) << " (ns) " << endl;
        cout << "    Degridding rate   " << (griddings / 1000000) / timeReal << " (million grid points per second)" << endl;

        cout << "Done" << endl;
    }

    // Verify degridding results
    cout << "Verifying result...";

    if (outdata0.size() != outdata1.size()) {
        cout << "Fail (Data vector sizes differ)" << std::endl;
        return 1;
    }

    for (unsigned int i = 0; i < outdata0.size(); ++i) {
        if (fabs(outdata0[i].real() - outdata1[i].real()) > 0.00001) {
            cout << "Fail (Expected " << outdata0[i].real() << " got "
                     << outdata1[i].real() << " at index " << i << ")"
                     << std::endl;
            return 1;
        }
    }

    cout << "Pass" << std::endl;

    {
        cpugrid.assign(cpugrid.size(), Value(1.0));
        // Now we can do the timing for the GPU implementation
        cout << "+++++ Reverse processing (OpenACC with complex data loop) +++++" << endl;

        // Time is measured inside this function call, unlike the CPU versions
        Stopwatch sw;
        sw.start();
        degridKernelDataLoopComplex(cpugrid, gSize, support, C, cOffset, iu, iv, outdata1);
        time = sw.stop();

        // Report on timings
        cout << "    Time " << time << " (s) = real mult / " << timeReal/time << endl;
        cout << "    Time per visibility spectral sample " << 1e6*time / double(data.size()) << " (us) " << endl;
        cout << "    Time per degridding   " << 1e9*time / (double(data.size())* double((sSize)*(sSize))) << " (ns) " << endl;
        cout << "    Degridding rate   " << (griddings / 1000000) / time << " (million grid points per second)" << endl;

        cout << "Done" << endl;
    }

    // Verify degridding results
    cout << "Verifying result...";

    if (outdata0.size() != outdata1.size()) {
        cout << "Fail (Data vector sizes differ)" << std::endl;
        return 1;
    }

    for (unsigned int i = 0; i < outdata0.size(); ++i) {
        if (fabs(outdata0[i].real() - outdata1[i].real()) > 0.00001) {
            cout << "Fail (Expected " << outdata0[i].real() << " got "
                     << outdata1[i].real() << " at index " << i << ")"
                     << std::endl;
            return 1;
        }
    }

    cout << "Pass" << std::endl;

    return 0;
}
