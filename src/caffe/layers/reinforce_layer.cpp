#include <algorithm>
#include <vector>

#include "caffe/layers/reinforce_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ReinforceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->shape());
}

template <typename Dtype>
void ReinforceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
}

template <typename Dtype>
void ReinforceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  CHECK_EQ(bottom[1]->count(), bottom[2]->count());
  CHECK_EQ(bottom[0]->count(), bottom[3]->count());

  const Dtype* bottom_prob_data = bottom[0]->cpu_data();
  const Dtype* bottom_action_data = bottom[1]->cpu_data();
  const Dtype* bottom_reward_data = bottom[2]->cpu_data();
  const Dtype* bottom_legality_data = bottom[3]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

  int num_actions = bottom[0]->count() / bottom[1]->count();
  int minibatch_size = bottom[1]->count();
  for (int i = 0; i < minibatch_size; ++i) {
    const int action = static_cast<int>(bottom_action_data[i]);
    const Dtype reward = bottom_reward_data[i];

    bottom_diff[action + i * num_actions] = -reward / std::max(Dtype(1e-5), bottom_prob_data[action + i * num_actions]);
  }
  for (int i = 0; i < bottom[0]->count(); ++i) {
    if (bottom_legality_data[i] == 0) {
      bottom_diff[i] += 1;
    }
  }
}

INSTANTIATE_CLASS(ReinforceLayer);
REGISTER_LAYER_CLASS(Reinforce);

}  // namespace caffe
