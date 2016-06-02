// lm/lm-model-sync.h

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

#ifndef LM_LM_MODEL_SYNC_H_
#define LM_LM_MODEL_SYNC_H_

#include "thread/kaldi-semaphore.h"
#include "thread/kaldi-mutex.h"
#include "thread/kaldi-barrier.h"
#include "nnet/nnet-nnet.h"

#include "cudamatrix/cu-device.h"
#include <mpi.h>

#include "nnet/nnet-model-sync.h"

namespace kaldi {
namespace lm {
typedef nnet1::Nnet Nnet;
typedef nnet1::NnetParallelOptions NnetParallelOptions;

#if HAVE_CUDA == 1
class StreamCache {
public:
	StreamCache (int size = 20)
	{
		cache_pos_ = 0;
		streamlist_.resize(size);
		for (int i = 0; i < size; i++)
			cudaStreamCreateWithFlags(&streamlist_[i], cudaStreamNonBlocking);
	}

	virtual ~StreamCache()
	{
		for (int i = 0; i < streamlist_.size(); i++)
			cudaStreamDestroy(streamlist_[i]);
	}

	int size()
	{
		return streamlist_.size();
	}

	inline cudaStream_t GetCudaStream()
	{
		int idx = cache_pos_;
		cache_pos_ = (cache_pos_+1)%streamlist_.size();
		return streamlist_[idx];
	}

private:
	int cache_pos_;
	std::vector<cudaStream_t > streamlist_;
};
#endif

class LmModelSync{
public:
	LmModelSync(Nnet *nnet, const NnetParallelOptions *opts=NULL):
		initialized_(false),is_lastmerge_(false),data_(NULL),free_data_(NULL),gradient_data_(NULL),
		dim_(0),num_threads_(opts->num_threads),left_merge_(opts->num_merge),
		mmt_(opts->global_momentum),learnrate_(opts->global_learnrate),nnet(nnet),opts_(opts)

	{
		MultiMachineInit();
	}

	~LmModelSync()
	{
		Destory();
	}

	void LockModel() {
		model_mutex_.Lock();
	}
	void UnlockModel(){
		model_mutex_.Unlock();
	}

	void LockStates() {
		stats_mutex_.Lock();
	}
	void UnlockStates(){
		stats_mutex_.Unlock();
	}

	void GetWeight(Nnet *nnet, int32 thread_idx, int32 buffer_idx=-1);

	void SetWeight(Nnet *nnet, int32 thread_idx, int32 buffer_idx=-1);

	void ThreadSync(int32 thread_idx, int status);

	void CrossMachineSyncStatus(int32 status);

	bool isLastMerge() { return is_lastmerge_;}

	void Destory();

	int32 Dim(){return this->dim_;};

	void CopyToHost(Nnet *nnet)
	{
		*(this->nnet) = *nnet;
	}

	void MultiMachineInit();

    int32 leftMerge(){ return left_merge_;}

	void Initialize(Nnet *nnet, int32 thread_idx)
	{
		model_mutex_.Lock();
		if (!initialized_)
		{
			barrier_.SetThreshold(num_threads_);
            stream_cache_.resize(num_threads_, NULL);
            if (NULL == stream_cache_[thread_idx])
                stream_cache_[thread_idx] = new StreamCache;
			this->GetWeight(nnet, thread_idx);
			initialized_ = true;
		}
        if (NULL == stream_cache_[thread_idx])
           stream_cache_[thread_idx] = new StreamCache;
		model_mutex_.Unlock();
	}

private:


	int GetDim(Nnet *nnet);
	void Init(Nnet *nnet);
	void CrossMachineSync();

	bool	initialized_;
	bool	is_lastmerge_;
	BaseFloat *data_;
	BaseFloat *free_data_;
	BaseFloat *gradient_data_;
	int32 dim_;
	int32 num_threads_;
	int32 left_merge_;
	BaseFloat mmt_;
	BaseFloat learnrate_;
	Nnet *nnet;
	const NnetParallelOptions *opts_;

	Barrier barrier_;
	Mutex model_mutex_;
	Mutex stats_mutex_;
	std::vector<BaseFloat*>	thread_data_;
	std::vector<BaseFloat*> thread_free_data_;


public:

#if HAVE_CUDA == 1
  kaldi::MPIGpuInfo *gpuinfo_;
  MPI_Win win;
  std::vector<StreamCache*> stream_cache_;
#endif
};



class LmParallelUtil{
public:
	std::string AddSuffix(std::string filename, int idx);
	std::string FAddSuffix(std::string filename, int idx);
	std::string GetFilename(std::string filename);
	int NumofMerge(std::string fn, int merge_size);
	int NumofCEMerge(std::string fn, int merge_size);
};

} // namespace nnet
} // namespace kaldi

#endif /* LM_LM_MODEL_SYNC_H_ */
