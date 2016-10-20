// util/circular-queue.h

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

#ifndef KALDI_UTIL_CIRCULAR_QUEUE_H_
#define KALDI_UTIL_CIRCULAR_QUEUE_H_

#include <vector>
#include "util/stl-utils.h"

namespace kaldi {

template<class T>
class CircularQueue {
public:
	CircularQueue(int size = 4);

	inline void push(const T &value);

	inline void pop();

	inline T front();

	inline T back();

	inline int size();

	inline bool empty();

	inline void clear();

	inline std::list<T>& GetList();
private:

	std::list<T> buffer_;
	std::list<T>::iterator front_;
	std::list<T>::iterator rear_;
	int	 size_;
};

}

#include "util/circular-queue-inl.h"

#endif /* UTIL_CIRCULAR_QUEUE_H_ */
