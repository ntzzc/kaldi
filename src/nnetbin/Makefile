
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)
LDLIBS += $(MPICH_LDLIBS)

BINFILES = nnet-train-frmshuff \
        nnet-train-perutt \
        nnet-train-mmi-sequential \
        nnet-train-mpe-sequential \
        nnet-train-lstm-streams nnet-train-blstm-streams \
        rbm-train-cd1-frmshuff rbm-convert-to-nnet \
        nnet-forward nnet-copy nnet-info nnet-concat \
        transf-to-nnet cmvn-to-nnet nnet-initialize \
        nnet-kl-hmm-acc nnet-kl-hmm-mat-to-component \
	feat-to-post paste-post train-transitions \
	cuda-gpu-available \
	nnet-svd \
	nnet-train-sequential-parallel nnet-train-sequential-parallel-mpi \
        nnet-train-frmshuff-parallel nnet-train-frmshuff-parallel-mpi \
        nnet-train-lstm-mpe-sequential  nnet-train-lstm-mmi-sequential

OBJFILES =



TESTFILES =

ADDLIBS = ../nnet/kaldi-nnet.a ../nnet2/kaldi-nnet2.a ../cudamatrix/kaldi-cudamatrix.a ../lat/kaldi-lat.a \
          ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a \
          ../util/kaldi-util.a ../thread/kaldi-thread.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
