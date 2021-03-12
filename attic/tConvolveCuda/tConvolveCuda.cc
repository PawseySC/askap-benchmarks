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


// Local includes
#include "common.h"
#include "CudaGridder.h"

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

    for (int dind = 0; dind < int(data.size()); ++dind) {
        // The actual grid point
        int gind = iu[dind] + gSize * iv[dind] - support;

        // The Convoluton function point from which we offset
        int cind = cOffset[dind];

        for (int suppv = 0; suppv < sSize; suppv++) {
            Value* gptr = &grid[gind];
            const Value* cptr = &C[cind];
            const Value d = data[dind];

            for (int suppu = 0; suppu < sSize; suppu++) {
                *(gptr++) += d * (*(cptr++));
            }

            gind += gSize;
            cind += sSize;
        }
    }
}

// Perform degridding
void degridKernel(const std::vector<Value>& grid, const int gSize, const int support,
                  const std::vector<Value>& C, const std::vector<int>& cOffset,
                  const std::vector<int>& iu, const std::vector<int>& iv,
                  std::vector<Value>& data)
{
    const int sSize = 2 * support + 1;

    for (int dind = 0; dind < int(data.size()); ++dind) {

        data[dind] = 0.0;

        // The actual grid point from which we offset
        int gind = iu[dind] + gSize * iv[dind] - support;

        // The Convoluton function point from which we offset
        int cind = cOffset[dind];

        for (int suppv = 0; suppv < sSize; suppv++) {
            Value* d = &data[dind];
            const Value* gptr = &grid[gind];
            const Value* cptr = &C[cind];

            for (int suppu = 0; suppu < sSize; suppu++) {
                (*d) += (*(gptr++)) * (*(cptr++));
            }

            gind += gSize;
            cind += sSize;
        }

    }
}


// Main testing routine
int main(int argc, char* argv[])
{
    Options opt;
    getinput(argc,argv,opt);
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
    std::vector<Value> compoutdata(nSamples*nChan);

    const unsigned int maxint = std::numeric_limits<int>::max();

    for (int i = 0; i < nSamples; i++) {
        u[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;
        v[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;
        w[i] = baseline * Coord(randomInt()) / Coord(maxint) - baseline / 2;

        for (int chan = 0; chan < nChan; chan++) {
            data[i*nChan+chan] = 1.0;
            cpuoutdata[i*nChan+chan] = 0.0;
            compoutdata[i*nChan+chan] = 0.0;
        }
    }

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

    std::string process, calctyperef, calctypecomp;
    std::vector<double> timingsref, timingscomp;
    Stopwatch sw;
    double time;

    ///////////////////////////////////////////////////////////////////////////
    // DO GRIDDING
    ///////////////////////////////////////////////////////////////////////////
    process = "Forward processing";
    calctyperef = "CPU";
    calctypecomp = "GPU CUDA";
    timingsref.clear();
    timingscomp.clear();
    std::vector<Value> cpugrid(gSize*gSize);
    std::vector<Value> compgrid(gSize*gSize);
    
    cout << "First iteration "<<endl;
    // Run reference CPU calculation 
    cout << "+++++ "<<process<<" "<<calctyperef<<" +++++" << endl;
    cpugrid.assign(cpugrid.size(), Value(0.0));
    sw.start();
    gridKernel(data, support, C, cOffset, iu, iv, cpugrid, gSize);
    time = sw.stop();
    report_timings(time, opt, sSize, griddings);
    timingsref.push_back(time);
    cout << "Done" << endl;
    // Now we can do the timing for other implementation
    cout << "+++++ "<<process<<" "<<calctypecomp<<" +++++" << endl;
    compgrid.assign(compgrid.size(), Value(0.0));
    time = 0;
    gridKernelCuda(data, support, C, cOffset, iu, iv, compgrid, gSize, time);
    timingscomp.push_back(time);
    report_timings(time, opt, sSize, griddings);
    verify_result(process+" : "+calctyperef+"<->"+calctypecomp, cpugrid, compgrid);

    if (opt.nIterations > 1) 
    { 
        cout <<" Now iterating to check times "<<endl;
        cout << "+++++ "<<process<<" "<<calctyperef<<" +++++" << endl;
        std::vector<Value> cpugriditer(gSize*gSize);
        std::vector<Value> compgriditer(gSize*gSize);

        for (auto i=1; i<opt.nIterations;i++) {
            cpugriditer.assign(cpugriditer.size(), Value(0.0));
            sw.start();
            gridKernel(data, support, C, cOffset, iu, iv, cpugriditer, gSize);
            time = sw.stop();
            timingsref.push_back(time);
        }
        report_timings(timingsref, opt, sSize, griddings);
        cout << "+++++ "<<process<<" "<<calctypecomp<<" +++++" << endl;
        for (auto i=1; i<opt.nIterations;i++) {
            compgriditer.assign(compgriditer.size(), Value(0.0));
            time = 0.0;
            gridKernelCuda(data, support, C, cOffset, iu, iv, compgriditer, gSize, time);
            timingscomp.push_back(time);
        }
        report_timings(timingscomp, opt, sSize, griddings);
    }

    ///////////////////////////////////////////////////////////////////////////
    // DO DEGRIDDING
    ///////////////////////////////////////////////////////////////////////////
    process = "Reverse processing";
    calctyperef = "CPU";
    calctypecomp = "GPU CUDA";
    timingsref.clear();
    timingscomp.clear();

    cout << "First iteration "<<endl;
    // Run reference CPU calculation 
    cout << "+++++ "<<process<<" "<<calctyperef<<" +++++" << endl;
    cpugrid.assign(cpugrid.size(), Value(1.0));
    sw.start();
    degridKernel(cpugrid, gSize, support, C, cOffset, iu, iv, cpuoutdata);
    time = sw.stop();
    report_timings(time, opt, sSize, griddings);
    timingsref.push_back(time);
    cout << "Done" << endl;
    // Now we can do the timing for other implementation
    cout << "+++++ "<<process<<" "<<calctypecomp<<" +++++" << endl;
    compgrid.assign(compgrid.size(), Value(1.0));
    time = 0.0;
    degridKernelCuda(compgrid, gSize, support, C, cOffset, iu, iv, compoutdata, time);
    timingscomp.push_back(time);
    report_timings(time, opt, sSize, griddings);
    verify_result(process+" : "+calctyperef+"<->"+calctypecomp, cpuoutdata, compoutdata);

    if (opt.nIterations > 1) 
    { 
        cout <<" Now iterating to check times "<<endl;
        cout << "+++++ "<<process<<" "<<calctyperef<<" +++++" << endl;
        std::vector<Value> cpugriditer(gSize*gSize);
        std::vector<Value> compgriditer(gSize*gSize);

        for (auto i=1; i<opt.nIterations;i++) {
            cpugriditer.assign(cpugriditer.size(), Value(1.0));
            sw.start();
            degridKernel(cpugriditer, gSize, support, C, cOffset, iu, iv, cpuoutdata);
            time = sw.stop();
            timingsref.push_back(time);
        }
        report_timings(timingsref, opt, sSize, griddings);
        cout << "+++++ "<<process<<" "<<calctypecomp<<" +++++" << endl;
        for (auto i=1; i<opt.nIterations;i++) {
            compgriditer.assign(compgriditer.size(), Value(1.0));
            time = 0.0;
            degridKernelCuda(compgriditer, gSize, support, C, cOffset, iu, iv, compoutdata, time);
            timingscomp.push_back(time);
        }
        report_timings(timingscomp, opt, sSize, griddings);
    }

    return 0;
}
