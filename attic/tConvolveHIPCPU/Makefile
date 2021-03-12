# Copyright (c) 2009 CSIRO
# Australia Telescope National Facility (ATNF)
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# PO Box 76, Epping NSW 1710, Australia
# atnf-enquiries@csiro.au
#
# This file is part of the ASKAP software distribution.
#
# The ASKAP software distribution is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the License,
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
#

CXX = g++
CFLAGS = -Wall -Wextra -O2 -fstrict-aliasing -fcx-limited-range -fPIC

CUDA = $(CUDATOOLKIT_HOME)/include
HIPCPU = /group/pawsey0001/mcytowski/HIP-CPU/askap-benchmarks/attic/HIP-CPU/include
EXENAME = tConvolveHIPCPU
EXEOBJS = $(EXENAME).o
EXELIBS = -lHIPCPUGridder

LIBNAME=libHIPCPUGridder.so
LIBOBJS=HIPCPUGridder.o HIPCPUGridKernel.o common.o

HIPCPUFLAGS= -fPIC -std=c++2a
#-std=c++20

all:		$(EXENAME) $(LIBNAME)

%.o:		%.cc %.h
		$(CXX) $(CFLAGS) -I$(HIPCPU) -c $<

$(LIBNAME):	HIPCPUGridder.cc HIPCPUGridder.h HIPCPUGridKernel.cu common.o
		gcc $(HIPCPUFLAGS) -I$(HIPCPU) -c HIPCPUGridKernel.cc
		gcc $(HIPCPUFLAGS) -I$(HIPCPU) -c HIPCPUGridder.cc
		gcc $(HIPCPUFLAGS) -shared -o $(LIBNAME) $(LIBOBJS)

$(EXENAME):	$(EXEOBJS) $(LIBNAME)
		$(CXX) $(CFLAGS) -L. -o $(EXENAME) $(EXEOBJS) $(EXELIBS)

clean: 
		rm -f $(EXENAME) $(LIBNAME) *.o

