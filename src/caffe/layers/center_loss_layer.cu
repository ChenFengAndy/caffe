#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Compute_distance_data_gpu(int nthreads, const int K, const Dtype* bottom,
	      const Dtype* label, const Dtype* center, Dtype* distance) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int m = index / K;
    int k = index % K;
    const int label_value = static_cast<int>(label[m]);
    // distance(i) = x(i) - c_{y(i)}
    distance[index] = bottom[index] - center[label_value * K + k];
  }
}

template <typename Dtype>
__global__ void Compute_center_diff_gpu(int nthreads, const int M, const int K, 
        const Dtype* label, const Dtype* distance, Dtype* variation_sum, 
        Dtype* center_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int count = 0;
    for (int m = 0; m < M; m++) {
      const int label_value = static_cast<int>(label[m]);
      if (label_value == index) {
        count++;
        for (int k = 0; k < K; k++) {
          variation_sum[index * K + k] -= distance[m * K + k];
        }
      }
    }
    for (int k = 0; k < K; k++) {
      center_diff[index * K + k] = variation_sum[index * K + k] /(count + (Dtype)1.);
    }
  }
}


template <typename Ftype, typename Btype>
void CenterLossLayer<Ftype, Btype>::Forward_gpu(
    const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  int nthreads = M_ * K_;
  Compute_distance_data_gpu<Ftype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, 
                                K_,
                                bottom[0]->gpu_data<Ftype>(),
                                bottom[1]->gpu_data<Ftype>(),
                                this->blobs_[0]->template gpu_data<Ftype>(),
                                distance_.template mutable_gpu_data<Ftype>());
  Ftype dot;
  caffe_gpu_dot<Ftype>(M_ * K_, 
    distance_.template gpu_data<Ftype>(),
    distance_.template gpu_data<Ftype>(),
    &dot);
  Ftype loss = dot / M_ / Ftype(2);
  top[0]->mutable_cpu_data<Ftype>()[0] = loss;
}

template <typename Ftype, typename Btype>
void CenterLossLayer<Ftype, Btype>::Backward_gpu(
    const vector<Blob*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  int nthreads = N_;
  caffe_gpu_set<Btype>(N_ * K_, (Btype)0., variation_sum_.template  mutable_cpu_data<Btype>());
  Compute_center_diff_gpu<Btype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(
      nthreads,
      M_, 
      K_,
      bottom[1]->template gpu_data<Btype>(),
      distance_.template gpu_data<Btype>(), 
      variation_sum_.template mutable_cpu_data<Btype>(),
      this->blobs_[0]->template mutable_gpu_diff<Btype>());

  if (propagate_down[0]) {
    caffe_gpu_scale<Btype>(M_ * K_,
		top[0]->template cpu_diff<Btype>()[0] / M_, 
        distance_.template gpu_data<Btype>(),
        bottom[0]->template mutable_gpu_diff<Btype>());
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(CenterLossLayer);

}  // namespace caffe
