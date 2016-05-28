// lm/lm-model-sync.cc

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

#include "nnet/nnet-affine-transform.h"
#include "nnet/nnet-lstm-projected-streams-fast.h"
#include "nnet/nnet-lstm-projected-streams.h"
#include "nnet/nnet-lstm-streams.h"
#include "nnet/nnet-class-affine-transform.h"
#include "nnet/nnet-word-vector-transform.h"

#include "lm/lm-model-sync.h"

namespace kaldi {
namespace lm {

void LmModelSync::Init(Nnet *nnet)
{
	if (NULL != this->free_data_)
		return;

	size_t size = 0;
	void *data = NULL;
	void *free_data = NULL;
	int32 dim = 0;

	dim = this->GetDim(nnet);
	this->dim_ = dim;

	size = dim * sizeof(BaseFloat)+16;

	thread_data_.resize(num_threads_+2);
	thread_free_data_.resize(num_threads_+2);

	for (int i = 0; i < thread_data_.size(); i++)
	{
		CU_SAFE_CALL(cudaHostAlloc((void**) &free_data, size, cudaHostAllocPortable)); //cudaHostAllocDefault
		data = (free_data ? (void *)( (((unsigned long)*(&free_data)) + 15) & ~0xFUL ) : NULL) ;
		if (NULL != data)
		{
			this->thread_data_[i] = static_cast<BaseFloat*> (data);
			this->thread_free_data_[i] = static_cast<BaseFloat*> (free_data);
		}
		else
		{
		    throw std::bad_alloc();
		}
	}

	this->data_ = thread_data_.back() - 1;
	this->gradient_data_ = thread_data_.back();

	CU_SAFE_CALL(cudaMemset(this->gradient_data_, 0, dim_*sizeof(BaseFloat)));

}

void LmModelSync::MultiMachineInit()
{
    if (opts_->num_procs > 1)
    {
        //p_merge_func_ = ModelMergeFunction::Factory(opts_, this);
#if HAVE_CUDA == 1
        gpuinfo_ = (MPIGpuInfo*)malloc(opts_->num_procs * opts_->num_threads * sizeof(MPIGpuInfo));
        std::memset(gpuinfo_, 0, opts_->num_procs * opts_->num_threads * sizeof(MPIGpuInfo));
#endif
    }
}

void LmModelSync::Destory()
{
	if (NULL != this->data_)
	{
		for (int i = 0; i < thread_free_data_.size(); i++)
		{
			CU_SAFE_CALL(cudaFreeHost(thread_free_data_[i]));
			this->thread_free_data_[i] = NULL;
			this->thread_data_[i] = NULL;
		}
		this->data_ = NULL;
		this->gradient_data_ = NULL;
		this->dim_ = 0;
	}
}

int LmModelSync::GetDim(Nnet *nnet)
{
	int dim = 0;
	nnet1::AffineTransform* aff_t;
	nnet1::LstmProjectedStreamsFast *lstm_t;
	nnet1::LstmProjectedStreams *plstm_t;
	nnet1::LstmStreams *stlstm_t;
	nnet1::ClassAffineTransform *class_affine;
	nnet1::WordVectorTransform *word_transf;

	for (int32 n = 0; n < nnet->components_.size(); n++)
	{
			if (nnet->components_[n]->IsUpdatable()) {
				switch (nnet->components_[n]->GetType()) {
				case nnet1::Component::kLstmProjectedStreamsFast:
					lstm_t = (nnet1::LstmProjectedStreamsFast*)(nnet->components_[n]);
					dim += lstm_t->w_gifo_x_.SizeInBytes()/sizeof(BaseFloat);
					dim += lstm_t->w_gifo_r_.SizeInBytes()/sizeof(BaseFloat);
					dim += lstm_t->bias_.Dim();
					dim += lstm_t->peephole_i_c_.Dim();
					dim += lstm_t->peephole_f_c_.Dim();
					dim += lstm_t->peephole_o_c_.Dim();
					dim += lstm_t->w_r_m_.SizeInBytes()/sizeof(BaseFloat);
					break;
				case nnet1::Component::kLstmStreams:
					stlstm_t = (nnet1::LstmStreams*)(nnet->components_[n]);
					dim += stlstm_t->w_gifo_x_.SizeInBytes()/sizeof(BaseFloat);
					dim += stlstm_t->w_gifo_m_.SizeInBytes()/sizeof(BaseFloat);
					dim += stlstm_t->bias_.Dim();
					dim += stlstm_t->peephole_i_c_.Dim();
					dim += stlstm_t->peephole_f_c_.Dim();
					dim += stlstm_t->peephole_o_c_.Dim();
					break;
				case nnet1::Component::kAffineTransform:
					aff_t = (nnet1::AffineTransform*)(nnet->components_[n]);
					dim += aff_t->linearity_.SizeInBytes()/sizeof(BaseFloat);
					dim += aff_t->bias_.Dim();
					break;
				case nnet1::Component::kClassAffineTransform:
					class_affine = (nnet1::ClassAffineTransform*)(nnet->components_[n]);
					dim += class_affine->linearity_.SizeInBytes()/sizeof(BaseFloat);
					dim += class_affine->bias_.Dim();
					break;
				case nnet1::Component::kWordVectorTransform:
					word_transf = (nnet1::WordVectorTransform*)(nnet->components_[n]);
					dim += word_transf->wordvector_.SizeInBytes()/sizeof(BaseFloat);
					break;
				default:
						KALDI_ERR<< "Unimplemented access to parameters "
						<< "of updatable component "
						<< nnet1::Component::TypeToMarker(nnet->components_[n]->GetType());
				}
			}
	}
	return dim;
}

void LmModelSync::GetWeight(Nnet *nnet, int32 thread_idx)
{
	if (NULL == this->data_)
	{
		this->Init(nnet);
	}

	KALDI_ASSERT(thread_idx <= num_threads_ - 1 );

	int32 pos = 0;
	void *host_data_ = thread_idx < 0 ? (void*)this->data_ : this->thread_data_[thread_idx];
	int32 dst_pitch, src_pitch, width, row, size;
	MatrixDim dim;
	nnet1::AffineTransform* aff_t;
	nnet1::LstmProjectedStreamsFast *lstm_t;
	nnet1::LstmProjectedStreams *plstm_t;
	nnet1::LstmStreams *stlstm_t;
	nnet1::ClassAffineTransform *class_affine;
	nnet1::WordVectorTransform *word_transf;

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
        Timer tim;
      for (int32 n = 0; n < nnet->components_.size(); n++) {
		if (nnet->components_[n]->IsUpdatable()) {
			switch (nnet->components_[n]->GetType()) {
			case nnet1::Component::kLstmProjectedStreamsFast:
				lstm_t = (nnet1::LstmProjectedStreamsFast*)(nnet->components_[n]);

				dim = lstm_t->w_gifo_x_.Dim();
				src_pitch = dim.stride*sizeof(BaseFloat);
				dst_pitch = src_pitch;
				width = dim.cols*sizeof(BaseFloat);
				cudaMemcpy2DAsync(host_data_+pos, dst_pitch, lstm_t->w_gifo_x_.Data(), src_pitch, width, dim.rows,
						cudaMemcpyDeviceToHost, stream_cache_.GetCudaStream());
				pos += lstm_t->w_gifo_x_.SizeInBytes();

				dim = lstm_t->w_gifo_r_.Dim();
				src_pitch = dim.stride*sizeof(BaseFloat);
				dst_pitch = src_pitch;
				width = dim.cols*sizeof(BaseFloat);
				cudaMemcpy2DAsync(host_data_+pos, dst_pitch, lstm_t->w_gifo_r_.Data(), src_pitch, width, dim.rows,
						cudaMemcpyDeviceToHost, stream_cache_.GetCudaStream());
				pos += lstm_t->w_gifo_r_.SizeInBytes();

				size = lstm_t->bias_.Dim()*sizeof(BaseFloat);
				cudaMemcpyAsync(host_data_+pos, lstm_t->bias_.Data(), size,
						cudaMemcpyDeviceToHost, stream_cache_.GetCudaStream());
				pos += size;

				size = lstm_t->peephole_i_c_.Dim()*sizeof(BaseFloat);
				cudaMemcpyAsync(host_data_+pos, lstm_t->peephole_i_c_.Data(), size,
						cudaMemcpyDeviceToHost, stream_cache_.GetCudaStream());
				pos += size;

				size = lstm_t->peephole_f_c_.Dim()*sizeof(BaseFloat);
				cudaMemcpyAsync(host_data_+pos, lstm_t->peephole_f_c_.Data(), size,
						cudaMemcpyDeviceToHost, stream_cache_.GetCudaStream());
				pos += size;

				size = lstm_t->peephole_o_c_.Dim()*sizeof(BaseFloat);
				cudaMemcpyAsync(host_data_+pos, lstm_t->peephole_o_c_.Data(), size,
						cudaMemcpyDeviceToHost, stream_cache_.GetCudaStream());
				pos += size;

				dim = lstm_t->w_r_m_.Dim();
				src_pitch = dim.stride*sizeof(BaseFloat);
				dst_pitch = src_pitch;
				width = dim.cols*sizeof(BaseFloat);
				cudaMemcpy2DAsync(host_data_+pos, dst_pitch, lstm_t->w_r_m_.Data(), src_pitch, width, dim.rows,
						cudaMemcpyDeviceToHost, stream_cache_.GetCudaStream());
				pos += lstm_t->w_r_m_.SizeInBytes();

				break;

			case nnet1::Component::kLstmStreams:
				stlstm_t = (LstmStreams*)(nnet->components_[n]);

				dim = stlstm_t->w_gifo_x_.Dim();
				src_pitch = dim.stride*sizeof(BaseFloat);
				dst_pitch = src_pitch;
				width = dim.cols*sizeof(BaseFloat);
				cudaMemcpy2DAsync(host_data_+pos, dst_pitch, stlstm_t->w_gifo_x_.Data(), src_pitch, width, dim.rows,
						cudaMemcpyDeviceToHost, stream_cache_.GetCudaStream());
				pos += stlstm_t->w_gifo_x_.SizeInBytes();

				dim = stlstm_t->w_gifo_m_.Dim();
				src_pitch = dim.stride*sizeof(BaseFloat);
				dst_pitch = src_pitch;
				width = dim.cols*sizeof(BaseFloat);
				cudaMemcpy2DAsync(host_data_+pos, dst_pitch, stlstm_t->w_gifo_m_.Data(), src_pitch, width, dim.rows,
						cudaMemcpyDeviceToHost, stream_cache_.GetCudaStream());
				pos += stlstm_t->w_gifo_m_.SizeInBytes();

				size = stlstm_t->bias_.Dim()*sizeof(BaseFloat);
				cudaMemcpyAsync(host_data_+pos, stlstm_t->bias_.Data(), size,
						cudaMemcpyDeviceToHost, stream_cache_.GetCudaStream());
				pos += size;

				size = stlstm_t->peephole_i_c_.Dim()*sizeof(BaseFloat);
				cudaMemcpyAsync(host_data_+pos, stlstm_t->peephole_i_c_.Data(), size,
						cudaMemcpyDeviceToHost, stream_cache_.GetCudaStream());
				pos += size;

				size = stlstm_t->peephole_f_c_.Dim()*sizeof(BaseFloat);
				cudaMemcpyAsync(host_data_+pos, stlstm_t->peephole_f_c_.Data(), size,
						cudaMemcpyDeviceToHost, stream_cache_.GetCudaStream());
				pos += size;

				size = stlstm_t->peephole_o_c_.Dim()*sizeof(BaseFloat);
				cudaMemcpyAsync(host_data_+pos, stlstm_t->peephole_o_c_.Data(), size,
						cudaMemcpyDeviceToHost, stream_cache_.GetCudaStream());
				pos += size;

				break;

			case nnet1::Component::kAffineTransform:
				dim = aff_t->linearity_.Dim();
				src_pitch = dim.stride*sizeof(BaseFloat);
				dst_pitch = src_pitch;
				width = dim.cols*sizeof(BaseFloat);

				cudaMemcpy2DAsync(host_data_+pos, dst_pitch, aff_t->linearity_.Data(), src_pitch, width, dim.rows,
						cudaMemcpyDeviceToHost, stream_cache_.GetCudaStream());

				pos += aff_t->linearity_.SizeInBytes();

				size = aff_t->bias_.Dim()*sizeof(BaseFloat);
				cudaMemcpyAsync(host_data_+pos, aff_t->bias_.Data(), size,
						cudaMemcpyDeviceToHost, stream_cache_.GetCudaStream());

				pos += size;
				break;

			case nnet1::Component::kClassAffineTransform:
				class_affine = (nnet1::ClassAffineTransform*)(nnet->components_[n]);
				dim = class_affine->linearity_.Dim();
				src_pitch = dim.stride*sizeof(BaseFloat);
				dst_pitch = src_pitch;
				width = dim.cols*sizeof(BaseFloat);

				cudaMemcpy2DAsync(host_data_+pos, dst_pitch, class_affine->linearity_.Data(), src_pitch, width, dim.rows,
						cudaMemcpyDeviceToHost, stream_cache_.GetCudaStream());

				pos += class_affine->linearity_.SizeInBytes();

				size = class_affine->bias_.Dim()*sizeof(BaseFloat);
				cudaMemcpyAsync(host_data_+pos, class_affine->bias_.Data(), size,
						cudaMemcpyDeviceToHost, stream_cache_.GetCudaStream());

				pos += size;
				break;

			case nnet1::Component::kWordVectorTransform:
				word_transf = (nnet1::WordVectorTransform*)(nnet->components_[n]);
				dim = word_transf->wordvector_.Dim();
				src_pitch = dim.stride*sizeof(BaseFloat);
				dst_pitch = src_pitch;
				width = dim.cols*sizeof(BaseFloat);

				cudaMemcpy2DAsync(host_data_+pos, dst_pitch, word_transf->wordvector_.Data(), src_pitch, width, dim.rows,
						cudaMemcpyDeviceToHost, stream_cache_.GetCudaStream());

				pos += word_transf->wordvector_.SizeInBytes();
				break;

			default:
				KALDI_ERR<< "Unimplemented access to parameters "
				<< "of updatable component "
				<< nnet1::Component::TypeToMarker(nnet->components_[n]->GetType());
			}
		}
      }
	  CU_SAFE_CALL(cudaGetLastError());

	  CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
} else
#endif
	{
		// not implemented for CPU yet
	}

}

void LmModelSync::SetWeight(Nnet *nnet, int32 thread_idx)
{
	KALDI_ASSERT(this->data_ != NULL);

	KALDI_ASSERT(thread_idx <= num_threads_ - 1 );

	int32 pos = 0;
	void *host_data_ = thread_idx < 0 ? (void*)this->data_ : this->thread_data_[thread_idx];
	int32 dst_pitch, src_pitch, width,  size;
	MatrixDim dim;
	nnet1::AffineTransform* aff_t;
	nnet1::LstmProjectedStreamsFast *lstm_t;
	nnet1::LstmProjectedStreams *plstm_t;
	nnet1::LstmStreams *stlstm_t;
	nnet1::ClassAffineTransform *class_affine;
	nnet1::WordVectorTransform *word_transf;

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
        Timer tim;
	for (int32 n = 0; n < nnet->components_.size(); n++) {
		if (nnet->components_[n]->IsUpdatable()) {
			switch (nnet->components_[n]->GetType()) {
			case nnet1::Component::kLstmProjectedStreamsFast:
				lstm_t = (nnet1::LstmProjectedStreamsFast*)(nnet->components_[n]);

				dim = lstm_t->w_gifo_x_.Dim();
				src_pitch = dim.stride*sizeof(BaseFloat);
				dst_pitch = src_pitch;
				width = dim.cols*sizeof(BaseFloat);
				cudaMemcpy2DAsync(lstm_t->w_gifo_x_.Data(), dst_pitch, host_data_+pos, src_pitch, width, dim.rows,
						cudaMemcpyHostToDevice, stream_cache_.GetCudaStream());
				pos += lstm_t->w_gifo_x_.SizeInBytes();

				dim = lstm_t->w_gifo_r_.Dim();
				src_pitch = dim.stride*sizeof(BaseFloat);
				dst_pitch = src_pitch;
				width = dim.cols*sizeof(BaseFloat);
				cudaMemcpy2DAsync(lstm_t->w_gifo_r_.Data(), dst_pitch, host_data_+pos, src_pitch, width, dim.rows,
						cudaMemcpyHostToDevice, stream_cache_.GetCudaStream());
				pos += lstm_t->w_gifo_r_.SizeInBytes();

				size = lstm_t->bias_.Dim()*sizeof(BaseFloat);
				cudaMemcpyAsync(lstm_t->bias_.Data(), host_data_+pos, size,
						cudaMemcpyHostToDevice, stream_cache_.GetCudaStream());
				pos += size;

				size = lstm_t->peephole_i_c_.Dim()*sizeof(BaseFloat);
				cudaMemcpyAsync(lstm_t->peephole_i_c_.Data(), host_data_+pos, size,
						cudaMemcpyHostToDevice, stream_cache_.GetCudaStream());
				pos += size;

				size = lstm_t->peephole_f_c_.Dim()*sizeof(BaseFloat);
				cudaMemcpyAsync(lstm_t->peephole_f_c_.Data(), host_data_+pos, size,
						cudaMemcpyHostToDevice, stream_cache_.GetCudaStream());
				pos += size;

				size = lstm_t->peephole_o_c_.Dim()*sizeof(BaseFloat);
				cudaMemcpyAsync(lstm_t->peephole_o_c_.Data(), host_data_+pos, size,
						cudaMemcpyHostToDevice, stream_cache_.GetCudaStream());
				pos += size;

				dim = lstm_t->w_r_m_.Dim();
				src_pitch = dim.stride*sizeof(BaseFloat);
				dst_pitch = src_pitch;
				width = dim.cols*sizeof(BaseFloat);
				cudaMemcpy2DAsync(lstm_t->w_r_m_.Data(), dst_pitch, host_data_+pos, src_pitch, width, dim.rows,
						cudaMemcpyHostToDevice, stream_cache_.GetCudaStream());
				pos += lstm_t->w_r_m_.SizeInBytes();
				break;

			case nnet1::Component::kLstmStreams:
				stlstm_t = (LstmStreams*)(nnet->components_[n]);

				dim = stlstm_t->w_gifo_x_.Dim();
				src_pitch = dim.stride*sizeof(BaseFloat);
				dst_pitch = src_pitch;
				width = dim.cols*sizeof(BaseFloat);
				cudaMemcpy2DAsync(stlstm_t->w_gifo_x_.Data(), dst_pitch, host_data_+pos, src_pitch, width, dim.rows,
						cudaMemcpyHostToDevice, stream_cache_.GetCudaStream());
				pos += stlstm_t->w_gifo_x_.SizeInBytes();

				dim = stlstm_t->w_gifo_m_.Dim();
				src_pitch = dim.stride*sizeof(BaseFloat);
				dst_pitch = src_pitch;
				width = dim.cols*sizeof(BaseFloat);
				cudaMemcpy2DAsync(stlstm_t->w_gifo_m_.Data(), dst_pitch, host_data_+pos, src_pitch, width, dim.rows,
						cudaMemcpyHostToDevice, stream_cache_.GetCudaStream());
				pos += stlstm_t->w_gifo_m_.SizeInBytes();

				size = stlstm_t->bias_.Dim()*sizeof(BaseFloat);
				cudaMemcpyAsync(stlstm_t->bias_.Data(), host_data_+pos, size,
						cudaMemcpyHostToDevice, stream_cache_.GetCudaStream());
				pos += size;

				size = stlstm_t->peephole_i_c_.Dim()*sizeof(BaseFloat);
				cudaMemcpyAsync(stlstm_t->peephole_i_c_.Data(), host_data_+pos, size,
						cudaMemcpyHostToDevice, stream_cache_.GetCudaStream());
				pos += size;

				size = stlstm_t->peephole_f_c_.Dim()*sizeof(BaseFloat);
				cudaMemcpyAsync(stlstm_t->peephole_f_c_.Data(), host_data_+pos, size,
						cudaMemcpyHostToDevice, stream_cache_.GetCudaStream());
				pos += size;

				size = stlstm_t->peephole_o_c_.Dim()*sizeof(BaseFloat);
				cudaMemcpyAsync(stlstm_t->peephole_o_c_.Data(), host_data_+pos, size,
						cudaMemcpyHostToDevice, stream_cache_.GetCudaStream());
				pos += size;

				break;
			case nnet1::Component::kAffineTransform:
				// get the component
				aff_t = (nnet1::AffineTransform*)(nnet->components_[n]);
				dim = aff_t->linearity_.Dim();
				dst_pitch = dim.stride*sizeof(BaseFloat);
				src_pitch = dst_pitch;
				width = dim.cols*sizeof(BaseFloat);


				cudaMemcpy2DAsync(aff_t->linearity_.Data(), dst_pitch, host_data_+pos, src_pitch, width, dim.rows,
						cudaMemcpyHostToDevice, stream_cache_.GetCudaStream());

				pos += aff_t->linearity_.SizeInBytes();

				size = aff_t->bias_.Dim()*sizeof(BaseFloat);

				cudaMemcpyAsync(aff_t->bias_.Data(), host_data_+pos, size,
						cudaMemcpyHostToDevice, stream_cache_.GetCudaStream());

				pos += size;

				break;
			case nnet1::Component::kClassAffineTransform:
				class_affine = (nnet1::ClassAffineTransform*)(nnet->components_[n]);
				dim = class_affine->linearity_.Dim();
				src_pitch = dim.stride*sizeof(BaseFloat);
				dst_pitch = src_pitch;
				width = dim.cols*sizeof(BaseFloat);

				cudaMemcpy2DAsync(class_affine->linearity_.Data(), dst_pitch, host_data_+pos, src_pitch, width, dim.rows,
						cudaMemcpyHostToDevice, stream_cache_.GetCudaStream());

				pos += class_affine->linearity_.SizeInBytes();

				size = class_affine->bias_.Dim()*sizeof(BaseFloat);
				cudaMemcpyAsync(class_affine->bias_.Data(), host_data_+pos, size,
						cudaMemcpyHostToDevice, stream_cache_.GetCudaStream());

				pos += size;
				break;
			case nnet1::Component::kWordVectorTransform:
				word_transf = (nnet1::WordVectorTransform*)(nnet->components_[n]);
				dim = word_transf->wordvector_.Dim();
				src_pitch = dim.stride*sizeof(BaseFloat);
				dst_pitch = src_pitch;
				width = dim.cols*sizeof(BaseFloat);

				cudaMemcpy2DAsync(word_transf->wordvector_.Data(), dst_pitch, host_data_+pos, src_pitch, width, dim.rows,
						cudaMemcpyHostToDevice, stream_cache_.GetCudaStream());

				pos += word_transf->wordvector_.SizeInBytes();
				break;
			default:
				KALDI_ERR<< "Unimplemented access to parameters "
				<< "of updatable component "
				<< nnet1::Component::TypeToMarker(nnet->components_[n]->GetType());
			}
		}
	}
	  CU_SAFE_CALL(cudaGetLastError());

	  CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
} else
#endif
	{
		// not implemented for CPU yet
	}

}

void LmModelSync::CrossMachineSync(int status)
{
	// cross machine reduce
	int total_status = 0;
	if (left_merge_ <= 1 && !is_lastmerge_)
	{
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Allreduce(&status, (void*)(&total_status), 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		if (total_status < opts_->num_procs)
			is_lastmerge_ = true;
	}

	if (left_merge_ > 1 || !is_lastmerge_)
	{
		void *srcaddr = (void *) (opts_->myid==0 ? MPI_IN_PLACE : this->thread_data_[0]);
		void *dstaddr = (void *) this->thread_data_[0];
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(srcaddr, dstaddr, this->dim_, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
		left_merge_--;
	}
}

void LmModelSync::ThreadSync(int32 thread_idx, int status)
{
	double t1, t2, tk;
	Timer tm;

	tm.Reset();
	this->barrier_.Wait();

	BaseFloat *cur_model, *thread_data, *last_model, *gradient;
	int offset = this->dim_/num_threads_ * thread_idx;
	int len = (thread_idx == num_threads_-1 ? dim_-offset : dim_/num_threads_);

	cur_model = this->thread_data_[0] + offset;
	last_model = this->data_ + offset;
	gradient = this->gradient_data_ + offset;

	for (int i = 1; i < num_threads_; i++)
	{
		thread_data = this->thread_data_[i] + offset;
		cblas_Xaxpy(len, 1.0, thread_data, 1, cur_model, 1);
	}

	KALDI_VLOG(1) << "THREAD_Reduce: " << tm.Elapsed();

	tm.Reset();
	// cross machine reduce
	if (opts_->num_procs > 1)
	{
		CrossMachineSync(status);
		this->barrier_.Wait();
	}
	KALDI_VLOG(1) << "MPI_Reduce: " << tm.Elapsed();

	tm.Reset();
	// model merge ...
	// average W(t)
	cblas_Xscal(len, 1.0/num_threads_, cur_model, 1);
	// global gradient G(t) = average W(t) - W(t-1)
	cblas_Xaxpy(len, -1, last_model, 1, cur_model, 1);
	// delta(t) = mmt * delta_(t-1) + lr * G(t)
	if (mmt_ < 0.0) mmt_ = 1.0 - 1.0/num_threads_;
	cblas_Xscal(this->dim_, mmt_, gradient, 1);
	cblas_Xaxpy(this->dim_, learnrate_, cur_model, 1, gradient, 1);

	// CBM: W(t) = W(t-1) + delta(t)
	//cblas_Xaxpy(this->dim_, 1.0, this->gradient_data_, 1, this->nnet_data_, 1);
	// NBM: W(t) = W(t-1) + delta(t) + mmt*delta(t)
	cblas_Xaxpy(this->dim_, 1.0+mmt_, gradient, 1, last_model, 1);

	this->barrier_.Wait();
	KALDI_VLOG(2) << "THREAD_Merge: " << tm.Elapsed();

}

/*
 * 'ark,o:copy-feats scp:exp/tri_dnn_mmi/scplist/train.scp ark:- |'
 */

std::string LmParallelUtil::AddSuffix(std::string filename, int idx)
{
  char buf[1024];
  char suf[1024], ext[1024], fn[1024];
  int  len;

  const char *pfn = filename.c_str();
  len = strlen(pfn);
  const char *p1, *p2;
  p1 = strstr(pfn,"scp:");
  if (NULL == p1) return "";
  p2 = strchr(p1, ' ');
  if (NULL == p2) p2 = pfn+len;

  strncpy(fn, pfn, p2-pfn); fn[p2-pfn] = '\0';
  int l1 = strlen(fn);
  char *p3 = strrchr(fn, '.');
  *p3='\0';

  strncpy(suf,p3+1, fn+l1-p3); suf[fn+l1-p3]='\0';

  strncpy(ext, p2, pfn+len-p2); ext[pfn+len-p2]='\0';

  sprintf(buf,"%s.%d.%s%s",fn,idx,suf, ext);

  return buf;
}

std::string LmParallelUtil::FAddSuffix(std::string filename, int idx)
{
  char buf[1024];
  char ext[128], fn[128];
  int  len;

  const char *pfn = filename.c_str();
  len = strlen(pfn);
  const char *p2;

  p2 = strchr(pfn, '.');

  strncpy(fn,pfn, p2-pfn); fn[p2-pfn]='\0';
  strncpy(ext, p2+1, pfn+len-p2); ext[pfn+len-p2]='\0';

  sprintf(buf,"%s.%d.%s",fn,idx,ext);

  return buf;
}

std::string LmParallelUtil::GetFilename(std::string filename)
{
  char fn[128];

  const char *pfn = filename.c_str();
  const char *p1, *p2;
  p1 = strstr(pfn,"scp:");
  p2 = strchr(p1, ' ');


  strncpy(fn,p1+4, p2-p1-4); fn[p2-p1-4]='\0';

  return fn;
}

int LmParallelUtil::NumofMerge(std::string fn, int merge_size)
{
	std::string sfn = fn+".len";
	std::ifstream in(sfn.c_str());
	std::string str, featname;
	int len, piece = 0;
	size_t frames = 0;
	while(std::getline(in, str))
	{
		std::istringstream ss(str);
		ss>>featname>>len;

		if (frames + len > merge_size)
		{
			piece++;
			frames = 0;
		}
		frames += len;
	}

	if (frames > merge_size/3)
		piece++;

	return piece;
}

int LmParallelUtil::NumofCEMerge(std::string fn, int merge_size)
{
	std::string sfn = fn+".len";
	std::ifstream in(sfn.c_str());
	std::string str, featname;
	int len, piece = 0;
	size_t frames = 0;
	while(std::getline(in, str))
	{
		std::istringstream ss(str);
		ss>>featname>>len;

		frames += len;
	}

	piece = frames/merge_size + 1;

	return piece;
}

} // namespace nnet
} // namespace kaldi
