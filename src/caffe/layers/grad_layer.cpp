#include <algorithm>
#include <vector>

#include "caffe/layers/grad_layer.hpp"

namespace caffe {

template <typename Dtype>
void GradLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->shape());
}

template <typename Dtype>
void GradLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
}

template <typename Dtype>
void GradLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  CHECK_EQ(bottom[0]->count(), bottom[1]->count());

  const Dtype* bottom_grad_data = bottom[1]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  caffe_copy(bottom[0]->count(), bottom_grad_data, bottom_diff);
}

INSTANTIATE_CLASS(GradLayer);
REGISTER_LAYER_CLASS(Grad);

}  // namespace caffe
