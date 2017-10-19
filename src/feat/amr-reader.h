// feat/amr-reader.h

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


/*
// THE AMR FORMAT IS SPECIFIED IN:http://ietfreport.isoc.org/rfc/PDF/rfc3267.pdf
// 
//
//
//
//  RIFF
//  |
//  AMR
//  |    \    \   \
//  fmt_ data ... data
//
//
*/


#ifndef KALDI_FEAT_AMR_READER_H_
#define KALDI_FEAT_AMR_READER_H_

#include <cstring>

#include "base/kaldi-types.h"
#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"


namespace kaldi {


/// This class's purpose is to read in AMR files.
class AmrData {
 public:
  enum ReadDataType { kReadData, kLeaveDataUndefined };

  AmrData(const VectorBase<BaseFloat> &data)
      : data_(data) {}

  AmrData() {}

  /// Read() will throw on error.  It's valid to call Read() more than once--
  /// in this case it will destroy what was there before.
  /// "is" should be opened in binary mode.
  void Read(std::istream &is);

  // This function returns the Amr data-- it's in a matrix
  // becase there may be multiple channels.  In the normal case
  // there's just one channel so Data() will have one row.
  const VectorBase<BaseFloat> &Data() const { return data_; }

  BaseFloat Duration() const { return data_.Dim() / 8000; }

  void CopyFrom(const AmrData &other) {
    data_.CopyFromVec(other.data_);
  }

  BaseFloat SampFreq() const { return 8000; }

  void Clear() {
    data_.Resize(0);
  }

  void Swap(AmrData *other) {
    data_.Swap(&(other->data_));
  }

 private:
  Vector<BaseFloat> data_;

};


// Holder class for .amr files that enables us to read (but not write) .amr
// files. c.f. util/kaldi-holder.h we don't use the KaldiObjectHolder template
// because we don't want to check for the \0B binary header. We could have faked
// it by pretending to read in the Amr data in text mode after failing to find
// the \0B header, but that would have been a little ugly.
class AmrHolder {

 public:
  typedef AmrData T;


  void Copy(const T &t) { t_.CopyFrom(t); }

  static bool IsReadInBinary() { return true; }

  void Clear() { t_.Clear(); }

  const T &Value() { return t_; }

  AmrHolder &operator = (const AmrHolder &other) {
    t_.CopyFrom(other.t_);
    return *this;
  }
  AmrHolder(const AmrHolder &other): t_(other.t_) {}

  AmrHolder() {}

  bool Read(std::istream &is) {
    // We don't look for the binary-mode header here [always binary]
    try {
      t_.Read(is);  // throws exception on failure.
      return true;
    } catch (const std::exception &e) {
      KALDI_WARN << "Exception caught in AmrHolder object (reading). " 
                 << e.what();
      return false;  // write failure.
    }
  }

  void Swap(AmrHolder *other) {
    t_.Swap(&(other->t_));
  }

  bool ExtractRange(const AmrHolder &other, const std::string &range) {
    KALDI_ERR << "ExtractRange is not defined for this type of holder.";
    return false;
  }

 private:
  T t_;
};


}  // namespace kaldi

#endif  // KALDI_FEAT_Amr_READER_H_
