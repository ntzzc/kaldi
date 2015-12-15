// nnet/nnet-model-merge-function.cc

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


#include "matrix/cblas-wrappers.h"

#include "nnet/nnet-model-merge-function.h"


namespace kaldi {
namespace nnet1 {


ModelMergeFunction*
ModelMergeFunction::Factory(const NnetParallelOptions *opts, NnetModelSync *model_sync)
{
	ModelMergeFunction* ret = NULL;
	MerFunType type;
	if (opts->merge_func == "average")
		type = AVERAGE;
	else if (opts->merge_func == "globalsum")
		type = GLOBAL_SUM;
	else
		type = GLOBAL_ADAGRAD;

	switch(type) {
	case AVERAGE:  ret = new ModelAverageMerge(opts, model_sync);  break;
	case GLOBAL_SUM:      ret = new ModelGlobalSumMerge(opts, model_sync);     break;
	case GLOBAL_ADAGRAD:		   ret = new ModelGlobalAdagradMerge(opts, model_sync); break;
	default: KALDI_ERR<< "Unknown MergeFunction type";
	break;
  }
  return ret;
}

void ModelAverageMerge::Merge(int root)
{

	//cblas_Xscal(model_sync_->Dim(), 1.0/opts->num_procs, model_sync_->data_, 1);

	void *srcaddr = (void *) (opts->myid==root ? MPI_IN_PLACE : this->model_sync_->data_);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(srcaddr, (void*)(this->model_sync_->data_),
			this->model_sync_->dim_, MPI_FLOAT, MPI_SUM, root, MPI_COMM_WORLD);

	if (opts->myid == root)
	{
		cblas_Xscal(model_sync_->Dim(), 1.0/opts->num_procs, model_sync_->data_, 1);
	}


	//std::cout<<"Reduce finished!"<<std::endl;

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast((void*)(model_sync_->data_), model_sync_->Dim(), MPI_FLOAT, root, MPI_COMM_WORLD);
	//std::cout<<"Bcast finished!"<<std::endl;
	this->mLeftMerge--;
}



void
ModelGlobalSumMerge::Init()
{
	if (NULL != this->nnet_free_data_)
		return;

	size_t size = 0;
	void *nnet_data_ = NULL;
	void *nnet_free_data_ = NULL;

	this->dim_ = this->model_sync_->Dim();

	size = dim_ * sizeof(BaseFloat)+16;
	CU_SAFE_CALL(cudaHostAlloc((void**) &nnet_free_data_, size, cudaHostAllocPortable)); //cudaHostAllocDefault
	nnet_data_ = (nnet_free_data_ ? (void *)( (((unsigned long)*(&nnet_free_data_)) + 15) & ~0xFUL ) : NULL) ;

	if (NULL != nnet_data_)
	{
		this->nnet_data_ = static_cast<BaseFloat*> (nnet_data_);
		this->nnet_free_data_ = static_cast<BaseFloat*> (nnet_free_data_);

		CU_SAFE_CALL(cudaMemcpy(this->nnet_data_, this->model_sync_->data_, dim_*sizeof(BaseFloat), cudaMemcpyHostToHost));
	}
	else
	{
	    throw std::bad_alloc();
	}
}


void ModelGlobalSumMerge::Merge(int root)
{

	NnetModelSync *model_sync = this->model_sync_;

	float eta = this->mLearningRate;

	cblas_Xaxpy(this->dim_, -1, this->nnet_data_, 1, model_sync->data_, 1);

	void *addr = (void *) (opts->myid==root ? MPI_IN_PLACE : model_sync->data_);

	MPI_Reduce(addr, (void*)(model_sync->data_), model_sync->Dim(), MPI_FLOAT, MPI_SUM, root, MPI_COMM_WORLD);

	if (opts->myid==root)
	{
		//cblas_Xscal(dim_, 1.0/opts->num_procs, model_sync->data_, 1);
		cblas_Xaxpy(this->dim_, eta/opts->num_procs, model_sync->data_, 1, this->nnet_data_, 1);
	}

	//std::cout<<"Adagrad Reduce finished!"<<std::endl;
			//t1 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast((void*)(this->nnet_data_), dim_, MPI_FLOAT, root, MPI_COMM_WORLD);

	//std::memcpy(model_sync->data_, this->nnet_data_, dim_ * sizeof(BaseFloat));
	CU_SAFE_CALL(cudaMemcpy(model_sync->data_, this->nnet_data_, dim_ * sizeof(BaseFloat), cudaMemcpyHostToHost));

	//t2 = MPI_Wtime();
			//c = (t2-t1)*1000;
			//std::cout<<"Bcast finished!"<<std::endl;
			//printf("ModelAdagrad ---- Reduce, Adagrad, Bcast, total time: %.2lf %.2lf %.2lf %.2lf ms.\n",a,b,c,a+b+c);

	this->mLeftMerge--;

}




void ModelGlobalAdagradMerge::AdaGrad(int32 dim, BaseFloat eta, BaseFloat K, const BaseFloat *gradient)
{
	  for(size_t i=0; i<dim; i++)
	  {
	        nnet_data_[i] += (eta/sqrt(K+(gradient[i])*(gradient[i])))*(gradient[i]);
	  }
}


void ModelGlobalAdagradMerge::Merge(int root)
{

	NnetModelSync *model_sync = this->model_sync_;

	float eta = this->mLearningRate;

	cblas_Xaxpy(this->dim_, -1, this->nnet_data_, 1, model_sync->data_, 1);

	void *addr = (void *) (opts->myid==root ? MPI_IN_PLACE : model_sync->data_);

	MPI_Reduce(addr, (void*)(model_sync->data_), model_sync->Dim(), MPI_FLOAT, MPI_SUM, root, MPI_COMM_WORLD);

	if (opts->myid==root)
	{
		//KALDI_VLOG(1) << "ModelGlobalAdagradMerge::AdaGrad" << " eta: " << eta << " dim: " << dim_;
		cblas_Xscal(dim_, 1.0/opts->num_procs, model_sync->data_, 1);
		this->AdaGrad(dim_, eta, 1, model_sync->data_);
	}

	//std::cout<<"Adagrad Reduce finished!"<<std::endl;
			//t1 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast((void*)(this->nnet_data_), dim_, MPI_FLOAT, root, MPI_COMM_WORLD);

	//std::memcpy(model_sync->data_, this->nnet_data_, dim_ * sizeof(BaseFloat));
	CU_SAFE_CALL(cudaMemcpy(model_sync->data_, this->nnet_data_, dim_ * sizeof(BaseFloat), cudaMemcpyHostToHost));

	//t2 = MPI_Wtime();
			//c = (t2-t1)*1000;
			//std::cout<<"Bcast finished!"<<std::endl;
			//printf("ModelAdagrad ---- Reduce, Adagrad, Bcast, total time: %.2lf %.2lf %.2lf %.2lf ms.\n",a,b,c,a+b+c);

	this->mLeftMerge--;

}


} // namespace nnet
} // namespace kaldi

