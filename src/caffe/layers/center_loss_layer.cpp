#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void CenterLossLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  LossLayer<Ftype, Btype>::LayerSetUp(bottom, top);
  const int num_output = this->layer_param_.center_loss_param().num_output();  
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.center_loss_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the weight
    vector<int> center_shape(2);
    center_shape[0] = N_;
    center_shape[1] = K_;
    this->blobs_[0] = Blob::create<Ftype>(center_shape);
    // fill the weights
    shared_ptr<Filler<Ftype> > center_filler(GetFiller<Ftype>(
        this->layer_param_.center_loss_param().center_filler()));
    center_filler->Fill(this->blobs_[0].get());

  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  COUNTED_ = false;
}

template <typename Ftype, typename Btype>
void CenterLossLayer<Ftype, Btype>::Reshape(
      const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  M_ = bottom[0]->num();
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  LossLayer<Ftype, Btype>::Reshape(bottom, top);
  distance_.ReshapeLike(*bottom[0]);
  variation_sum_.ReshapeLike(*this->blobs_[0]);
  count_.Reshape(N_);
}

template <typename Ftype, typename Btype>
void CenterLossLayer<Ftype, Btype>::Forward_cpu(
    const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  const Ftype* label = bottom[1]->cpu_data<Ftype>();
  const Ftype* center = this->blobs_[0]->template cpu_data<Ftype>();
  Ftype* distance_data = distance_.template mutable_cpu_data<Ftype>();
  
  // the i-th distance_data
  for (int i = 0; i < M_; i++) {
    const int label_value = static_cast<int>(label[i]);
    // D(i,:) = X(i,:) - C(y(i),:)
    caffe_sub<Ftype>(K_, bottom_data + i * K_, center + label_value * K_, distance_data + i * K_);
  }
  Ftype dot = caffe_cpu_dot<Ftype>(M_ * K_, distance_.template cpu_data<Ftype>(), distance_.template cpu_data<Ftype>());
  Ftype loss = dot / M_ / Ftype(2);
  top[0]->template mutable_cpu_data<Ftype>()[0] = loss;
}

template <typename Ftype, typename Btype>
void CenterLossLayer<Ftype, Btype>::Backward_cpu(
    const vector<Blob*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  // Gradient with respect to centers
  if (this->param_propagate_down_[0]) {
    const Btype* label = bottom[1]->cpu_data<Btype>();
    Btype* center_diff = this->blobs_[0]->template mutable_cpu_diff<Btype>();
    Btype* variation_sum_data = variation_sum_.template mutable_cpu_data<Btype>();
    const Btype* distance_data = distance_.template cpu_data<Btype>();

    // \sum_{y_i==j}
    caffe_set(N_ * K_, (Btype)0., variation_sum_.template mutable_cpu_data<Btype>());
    for (int n = 0; n < N_; n++) {
      int count = 0;
      for (int m = 0; m < M_; m++) {
        const int label_value = static_cast<int>(label[m]);
        if (label_value == n) {
          count++;
          caffe_sub<Btype>(K_, variation_sum_data + n * K_, distance_data + m * K_, variation_sum_data + n * K_);
        }
      }
      caffe_axpy<Btype>(K_, (Btype)1./(count + (Btype)1.), variation_sum_data + n * K_, center_diff + n * K_);
    }
  }
  // Gradient with respect to bottom data 
  if (propagate_down[0]) {
    caffe_copy<Btype>(M_ * K_, distance_.template cpu_data<Btype>(), bottom[0]->template mutable_cpu_diff<Btype>());
    caffe_scal<Btype>(M_ * K_, top[0]->template cpu_diff<Btype>()[0] / M_, bottom[0]->template mutable_cpu_diff<Btype>());
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(CenterLossLayer);
#endif

INSTANTIATE_CLASS_FB(CenterLossLayer);
REGISTER_LAYER_CLASS(CenterLoss);

}  // namespace caffe
