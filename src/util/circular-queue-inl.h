// util/circular-queue-inl.h

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

#ifndef KALDI_UTIL_CIRCULAR_QUEUE_INL_H_
#define KALDI_UTIL_CIRCULAR_QUEUE_INL_H_


namespace kaldi {

	template<class T>
	CircularQueue<T>::CircularQueue(int size)
	{
		KALDI_ASSERT(size>0);

		buffer_.resize(size);
		rear_ = buffer_.begin();
		front_ = rear_;
		size_ = 0;
	}

	template<class T>
	void CircularQueue<T>::push(T &value)
	{
		auto *it = (front_+1 == buffer_.end()) ? buffer_.begin() : front_+1;

		if (it != rear_)
		{
			*front_ = value;
			front_ = it;
		}
		else // it == rear_ : queue full
			buffer_.insert(front_, value);
		size_++;
	}

	template<class T>
	void CircularQueue<T>::pop()
	{
		if (rear_ != front_)
		{
			rear_++;
			size_--;
		}
	}

	template<class T>
	T CircularQueue<T>::front()
	{
		if (size_ == 0)
			return;

		return *rear_;
	}

	template<class T>
	bool CircularQueue<T>::empty()
	{
		if (size_ == 0)
			KALDI_ASSERT(rear_ == front_);
		return rear_ == front_;
	}

	template<class T>
	void CircularQueue<T>::clear()
	{
		buffer_.clear();
		rear_ = buffer_.begin();
		front_ = rear_;
		size_ = 0;
	}

	template<class T>
	int CircularQueue<T>::size()
	{
		return size_;
	}

	template<class T>
	std::list<T>& CircularQueue<T>::GetList()
	{
		return buffer_;
	}
}

#endif /* UTIL_CIRCULAR_QUEUE_INL_H_ */
