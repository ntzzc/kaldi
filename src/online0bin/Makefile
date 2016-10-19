
all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)
LDLIBS += $(MPICH_LDLIBS)

BINFILES = online-nnet-decoder  online-nnet-forward

OBJFILES =

TESTFILES =

ADDLIBS = ../online0/kaldi-online0.a ../nnet0/kaldi-nnet0.a\
          ../cudamatrix/kaldi-cudamatrix.a ../decoder/kaldi-decoder.a \
          ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
          ../feat/kaldi-feat.a ../transform/kaldi-transform.a \
          ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../thread/kaldi-thread.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a 
include ../makefiles/default_rules.mk
