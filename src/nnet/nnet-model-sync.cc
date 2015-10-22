// nnet/nnet-model-sync.cc

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
#include "nnet/nnet-affine-preconditioned-transform.h"
#include "nnet/nnet-batchnorm-transform.h"
#include "nnet/nnet-convolutional-2d-component-fast.h"

#include "nnet/nnet-model-sync.h"
#include "nnet-model-merge-function.h"

namespace kaldi {
namespace nnet1 {

void
NnetModelSync::Init(Nnet *nnet)
{
	if (NULL != this->free_data_)
		return;

	size_t size = 0;
	void *data = NULL;
	void *free_data = NULL;
	int32 dim = 0;

	dim = this->GetDim(nnet);

	size = dim * sizeof(BaseFloat)+16;
	CU_SAFE_CALL(cudaHostAlloc((void**) &free_data, size, cudaHostAllocPortable)); //cudaHostAllocDefault
	data = (free_data ? (void *)( (((unsigned long)*(&free_data)) + 15) & ~0xFUL ) : NULL) ;

	if (NULL != data)
	{
		this->data_ = static_cast<BaseFloat*> (data);
		this->free_data_ = static_cast<BaseFloat*> (free_data_);
		this->dim_ = dim;
	}
	else
	{
	    throw std::bad_alloc();
	}

}

void
NnetModelSync::MultiMachineInit()
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

void
NnetModelSync::InitMergeFunction()
{
	if (opts_->num_procs > 1 && NULL == p_merge_func_)
	{
		p_merge_func_ = ModelMergeFunction::Factory(opts_, this);
	}
}
void
NnetModelSync::Destory()
{
	if (NULL != this->free_data_)
	{
		CU_SAFE_CALL(cudaFreeHost(this->free_data_));
		this->free_data_ = NULL;
		this->data_ = NULL;
		this->dim_ = 0;
	}

}

int32
NnetModelSync::GetDim(Nnet *nnet)
{
	int32 dim = 0;
	AffineTransform* aff_t;
	BatchNormTransform *norm_t;
	Convolutional2DComponentFast *conv_t;

	for (int32 n = 0; n < nnet->components_.size(); n++)
	{
			if (nnet->components_[n]->IsUpdatable()) {
				switch (nnet->components_[n]->GetType()) {
				case Component::kConvolutional2DComponentFast:
					conv_t = (Convolutional2DComponentFast*)(nnet->components_[n]);
					dim += conv_t->filters_.SizeInBytes()/sizeof(BaseFloat);
					dim += conv_t->bias_.Dim();
					break;
				case Component::kBatchNormTransform:
					norm_t = (BatchNormTransform*)(nnet->components_[n]);
					dim += norm_t->scale_.Dim();
					dim += norm_t->shift_.Dim();
					break;
				case Component::kAffineTransform:
					aff_t = (AffineTransform*)(nnet->components_[n]);
				case Component::kAffinePreconditionedOnlineTransform:
									// get the component
					aff_t = (AffineTransform*)(nnet->components_[n]);
				
					dim += aff_t->linearity_.SizeInBytes()/sizeof(BaseFloat);
					dim += aff_t->bias_.Dim();
					break;
				default:
						KALDI_ERR<< "Unimplemented access to parameters "
						<< "of updatable component "
						<< Component::TypeToMarker(nnet->components_[n]->GetType());
				}
			}
	}
	return dim;
}

void
NnetModelSync::GetWeight(Nnet *nnet)
{
	if (NULL == this->data_)
	{
		this->Init(nnet);
	}

	int32 pos = 0;
	void *host_data_ = (void*)this->data_;
	int32 dst_pitch, src_pitch, width, row, size;
	MatrixDim dim;
	AffineTransform* aff_t;
	BatchNormTransform *norm_t;
	Convolutional2DComponentFast *conv;
	for (int32 n = 0; n < nnet->components_.size(); n++) {
		if (nnet->components_[n]->IsUpdatable()) {
			switch (nnet->components_[n]->GetType()) {
			case Component::kConvolutional2DComponentFast:
				conv = (Convolutional2DComponentFast*)(nnet->components_[n]);

				dim = conv->filters_.Dim();
				src_pitch = dim.stride*sizeof(BaseFloat);
				dst_pitch = src_pitch;
				width = dim.cols*sizeof(BaseFloat);

				CU_SAFE_CALL(cudaMemcpy2D(host_data_+pos, dst_pitch,
						conv->filters_.Data(), src_pitch, width, dim.rows, cudaMemcpyDeviceToHost));
				pos += conv->filters_.SizeInBytes();
				size = conv->bias_.Dim()*sizeof(BaseFloat);
				CU_SAFE_CALL(cudaMemcpy(host_data_+pos, conv->bias_.Data(), size, cudaMemcpyDeviceToHost));

				pos += size;
				break;

			case Component::kBatchNormTransform:
				norm_t = (BatchNormTransform*)(nnet->components_[n]);
				{
					size = norm_t->scale_.Dim()*sizeof(BaseFloat);
					CU_SAFE_CALL(cudaMemcpy(host_data_+pos, norm_t->scale_.Data(), size, cudaMemcpyDeviceToHost));
					pos += size;

					size = norm_t->shift_.Dim()*sizeof(BaseFloat);
					CU_SAFE_CALL(cudaMemcpy(host_data_+pos, norm_t->shift_.Data(), size, cudaMemcpyDeviceToHost));
					pos += size;
				}
				break;
			case Component::kAffineTransform:
				// get the component
				//aff_t = dynamic_cast<AffineTransform*>(nnet->components_[n]);
				aff_t = (AffineTransform*)(nnet->components_[n]);
			case Component::kAffinePreconditionedOnlineTransform:
				// get the component
				//aff_t = dynamic_cast<AffinePreconditionedOnlineTransform*>(nnet->components_[n]);
				aff_t = (AffineTransform*)(nnet->components_[n]);
			{
				dim = aff_t->linearity_.Dim();
				src_pitch = dim.stride*sizeof(BaseFloat);
				dst_pitch = src_pitch;
				width = dim.cols*sizeof(BaseFloat);

				CU_SAFE_CALL(cudaMemcpy2D(host_data_+pos, dst_pitch,
										aff_t->linearity_.Data(), src_pitch, width, dim.rows,
										cudaMemcpyDeviceToHost));

				pos += aff_t->linearity_.SizeInBytes();

				size = aff_t->bias_.Dim()*sizeof(BaseFloat);
				CU_SAFE_CALL(cudaMemcpy(host_data_+pos, aff_t->bias_.Data(), size, cudaMemcpyDeviceToHost));

				pos += size;
			}break;
			default:
				KALDI_ERR<< "Unimplemented access to parameters "
				<< "of updatable component "
				<< Component::TypeToMarker(nnet->components_[n]->GetType());
			}
		}
	}

}

void
NnetModelSync::SetWeight(Nnet *nnet)
{
	KALDI_ASSERT(this->data_ != NULL);

	int32 pos = 0;
	void *host_data_ = (void*)this->data_;
	int32 dst_pitch, src_pitch, width,  size;
	MatrixDim dim;
	AffineTransform* aff_t;
	BatchNormTransform *norm_t;
	Convolutional2DComponentFast *conv;

	for (int32 n = 0; n < nnet->components_.size(); n++) {
		if (nnet->components_[n]->IsUpdatable()) {
			switch (nnet->components_[n]->GetType()) {
			case Component::kConvolutional2DComponentFast:
				conv = (Convolutional2DComponentFast*)(nnet->components_[n]);

				dim = conv->filters_.Dim();
				src_pitch = dim.stride*sizeof(BaseFloat);
				dst_pitch = src_pitch;
				width = dim.cols*sizeof(BaseFloat);

				CU_SAFE_CALL(cudaMemcpy2D(conv->filters_.Data(), dst_pitch,
							host_data_+pos, src_pitch, width, dim.rows, cudaMemcpyHostToDevice));

				pos += conv->filters_.SizeInBytes();

				size = conv->bias_.Dim()*sizeof(BaseFloat);

				CU_SAFE_CALL(cudaMemcpy(conv->filters_.Data(), host_data_+pos, size, cudaMemcpyHostToDevice));

				pos += size;
				break;
			case Component::kBatchNormTransform:
					norm_t = (BatchNormTransform*)(nnet->components_[n]);
					{
						size = norm_t->scale_.Dim()*sizeof(BaseFloat);
						CU_SAFE_CALL(cudaMemcpy(norm_t->scale_.Data(), host_data_+pos, size, cudaMemcpyHostToDevice));
						pos += size;

						size = norm_t->shift_.Dim()*sizeof(BaseFloat);
						CU_SAFE_CALL(cudaMemcpy(norm_t->shift_.Data(), host_data_+pos, size, cudaMemcpyHostToDevice));
						pos += size;
					}
					break;
			case Component::kAffineTransform:
				// get the component
				aff_t = (AffineTransform*)(nnet->components_[n]);

			case Component::kAffinePreconditionedOnlineTransform:
				// get the component
				aff_t = (AffinePreconditionedOnlineTransform*)(nnet->components_[n]);

			{
				dim = aff_t->linearity_.Dim();
				dst_pitch = dim.stride*sizeof(BaseFloat);
				src_pitch = dst_pitch;
				width = dim.cols*sizeof(BaseFloat);


				CU_SAFE_CALL(cudaMemcpy2D(aff_t->linearity_.Data(), dst_pitch,
										host_data_+pos, src_pitch, width, dim.rows,
										cudaMemcpyHostToDevice));

				pos += aff_t->linearity_.SizeInBytes();

				size = aff_t->bias_.Dim()*sizeof(BaseFloat);

				CU_SAFE_CALL(cudaMemcpy(aff_t->bias_.Data(), host_data_+pos, size, cudaMemcpyHostToDevice));

				pos += size;
			}break;
			default:
				KALDI_ERR<< "Unimplemented access to parameters "
				<< "of updatable component "
				<< Component::TypeToMarker(nnet->components_[n]->GetType());
			}
		}
	}

}

/*
 * 'ark,o:copy-feats scp:exp/tri_dnn_mmi/scplist/train.scp ark:- |'
 */

std::string NnetParallelUtil::AddSuffix(std::string filename, int idx)
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

std::string NnetParallelUtil::FAddSuffix(std::string filename, int idx)
{
  char buf[1024];
  char ext[128], fn[128];
  int  len;

  const char *pfn = filename.c_str();
  len = strlen(pfn);
  const char *p1, *p2;

  p2 = strchr(pfn, '.');

  strncpy(fn,pfn, p2-pfn); fn[p2-pfn]='\0';
  strncpy(ext, p2+1, pfn+len-p2); ext[pfn+len-p2]='\0';

  sprintf(buf,"%s.%d.%s",fn,idx,ext);

  return buf;
}

std::string NnetParallelUtil::GetFilename(std::string filename)
{
  char fn[128];

  const char *pfn = filename.c_str();
  const char *p1, *p2;
  p1 = strstr(pfn,"scp:");
  p2 = strchr(p1, ' ');


  strncpy(fn,p1+4, p2-p1-4); fn[p2-p1-4]='\0';

  return fn;
}

int NnetParallelUtil::NumofMerge(std::string fn, int merge_size)
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


} // namespace nnet
} // namespace kaldi
