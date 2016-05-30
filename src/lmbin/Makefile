
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)
LDLIBS += $(MPICH_LDLIBS)

BINFILES = arpa2fst arpa-to-const-arpa \
			lm-train-lstm-parallel lm-train-lstm-parallel-mpi

OBJFILES =

TESTFILES =

ADDLIBS = ../lm/kaldi-lm.a ../util/kaldi-util.a ../thread/kaldi-thread.a \
          ../base/kaldi-base.a ../nnet/kaldi-nnet.a ../cudamatrix/kaldi-cudamatrix.a ../matrix/kaldi-matrix.a

include ../makefiles/default_rules.mk
