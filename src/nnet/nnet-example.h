// nnet/nnet-example.h

// Copyright 2015-2016   Shanghai Jiao Tong University (author: Wei Deng)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef NNET_NNET_EXAMPLE_H_
#define NNET_NNET_EXAMPLE_H_

#include "nnet/nnet-compute-parallel.h"
#include "nnet/nnet-compute-sequential-parallel.h"
#include "nnet/nnet-compute-ctc-parallel.h"

namespace kaldi {
namespace nnet1 {


struct NnetExample{

	SequentialBaseFloatMatrixReader *feature_reader;

	std::string utt;
	Matrix<BaseFloat> input_frames;

	NnetExample(SequentialBaseFloatMatrixReader *feature_reader):
		feature_reader(feature_reader){}
	virtual ~NnetExample(){}

	virtual bool PrepareData() = 0;


};

struct DNNNnetExample : NnetExample
{


	RandomAccessPosteriorReader *targets_reader;
	RandomAccessBaseFloatVectorReader *weights_reader;

	NnetModelSync *model_sync;
	NnetStats *stats;
	const NnetUpdateOptions *opts;


	Posterior targets;
	Vector<BaseFloat> frames_weights;

	DNNNnetExample(SequentialBaseFloatMatrixReader *feature_reader,
					RandomAccessPosteriorReader *targets_reader,
					RandomAccessBaseFloatVectorReader *weights_reader,

					NnetModelSync *model_sync,
					NnetStats *stats,
					const NnetUpdateOptions *opts):
	NnetExample(feature_reader), targets_reader(targets_reader), weights_reader(weights_reader),
	model_sync(model_sync), stats(stats), opts(opts)
	{

	}
	bool PrepareData();
};

struct CTCNnetExample : NnetExample
{
	RandomAccessInt32VectorReader *targets_reader;

	NnetModelSync *model_sync;
	NnetCtcStats *stats;
	const NnetUpdateOptions *opts;

	std::vector<int32> targets;

	CTCNnetExample(SequentialBaseFloatMatrixReader *feature_reader,
					RandomAccessInt32VectorReader *targets_reader,

					NnetModelSync *model_sync,
					NnetCtcStats *stats,
					const NnetUpdateOptions *opts):
	NnetExample(feature_reader), targets_reader(targets_reader),
	model_sync(model_sync), stats(stats), opts(opts)
	{

	}
	bool PrepareData();
};

struct SequentialNnetExample : NnetExample
{
	RandomAccessLatticeReader *den_lat_reader;
	RandomAccessInt32VectorReader *num_ali_reader;
	NnetModelSync *model_sync;
	NnetSequentialStats *stats;
	const NnetSequentialUpdateOptions *opts;

	 /// The numerator alignment
	std::vector<int32> num_ali;
	Lattice den_lat;

	std::vector<int32> state_times;


	SequentialNnetExample(SequentialBaseFloatMatrixReader *feature_reader,
							RandomAccessLatticeReader *den_lat_reader,
							RandomAccessInt32VectorReader *num_ali_reader,
							NnetModelSync *model_sync,
							NnetSequentialStats *stats,
							const NnetSequentialUpdateOptions *opts):
								NnetExample(feature_reader), den_lat_reader(den_lat_reader),
								num_ali_reader(num_ali_reader),model_sync(model_sync),stats(stats),opts(opts)
							{

							}

	bool PrepareData();

};


struct LstmNnetExample: NnetExample
{
    Vector<BaseFloat> frame_mask;
    Posterior target;
    Matrix<BaseFloat> feat;
    std::vector<int> new_utt_flags;

    LstmNnetExample(Vector<BaseFloat> &mask, Posterior &tgt, Matrix<BaseFloat> &ft, std::vector<int> &flags)
    :NnetExample(NULL)
    {
    	frame_mask = mask;
    	target = tgt;
    	feat = ft;
    	new_utt_flags = flags;
    }
    bool PrepareData();
};

struct FeatureExample: NnetExample
{
	FeatureExample(SequentialBaseFloatMatrixReader *feature_reader)
	:NnetExample(feature_reader){}

	bool PrepareData()
	{
		utt = feature_reader->Key();
		input_frames = feature_reader->Value();
		return true;
	}

};

/** This struct stores neural net training examples to be used in
    multi-threaded training.  */
class ExamplesRepository {
 public:
  /// The following function is called by the code that reads in the examples.
  void AcceptExample(NnetExample *example);

  /// The following function is called by the code that reads in the examples,
  /// when we're done reading examples; it signals this way to this class
  /// that the stream is now empty
  void ExamplesDone();

  /// This function is called by the code that does the training.  If there is
  /// an example available it will provide it, or it will sleep till one is
  /// available.  It returns NULL when there are no examples left and
  /// ExamplesDone() has been called.
  NnetExample *ProvideExample();

  ExamplesRepository(): buffer_size_(128),
                                      empty_semaphore_(buffer_size_),
                                      done_(false) { }
 private:
  int32 buffer_size_;
  Semaphore full_semaphore_;
  Semaphore empty_semaphore_;
  Mutex examples_mutex_; // mutex we lock to modify examples_.

  std::deque<NnetExample*> examples_;
  bool done_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(ExamplesRepository);
};








} // namespace nnet
} // namespace kaldi


#endif /* NNET_NNET_EXAMPLE_H_ */
