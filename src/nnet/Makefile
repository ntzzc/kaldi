

all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES = nnet-randomizer-test nnet-component-test

OBJFILES = nnet-nnet.o nnet-component.o nnet-loss.o \
           nnet-pdf-prior.o nnet-randomizer.o \
	   nnet-example.o nnet-model-sync.o \
           nnet-compute-sequential-parallel.o nnet-compute-parallel.o \
           nnet-affine-preconditioned-transform.o nnet-model-merge-function.o \
	   nnet-compute-lstm-parallel.o nnet-compute-lstm-asgd.o

LIBNAME = kaldi-nnet

ADDLIBS = ../cudamatrix/kaldi-cudamatrix.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a  ../util/kaldi-util.a 

include ../makefiles/default_rules.mk

