#include <torch/extension.h>
#include <cuda_fp16.h>
using half_t = half;

void cuda_forward(int B, int T, int H, half_t* w, half_t* q, half_t* k, half_t* v, half_t* z, half_t* a, half_t* y, float* s, float* sa);
void forward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &y, torch::Tensor &s, torch::Tensor &sa) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_forward(B, T, H, 
                (half_t*)w.data_ptr(), 
                (half_t*)q.data_ptr(), 
                (half_t*)k.data_ptr(), 
                (half_t*)v.data_ptr(), 
                (half_t*)z.data_ptr(), 
                (half_t*)a.data_ptr(), 
                (half_t*)y.data_ptr(), 
                (float*)s.data_ptr(), 
                (float*)sa.data_ptr());
}

void cuda_backward(int B, int T, int H, half_t* w, half_t* q, half_t* k, half_t* v, half_t* z, half_t* a, half_t* dy, float* s, float* sa, half_t* dw, half_t* dq, half_t* dk, half_t* dv, half_t* dz, half_t* da);
void backward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &dy,
        torch::Tensor &s, torch::Tensor &sa, torch::Tensor &dw, torch::Tensor &dq, torch::Tensor &dk, torch::Tensor &dv, torch::Tensor &dz, torch::Tensor &da) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_backward(B, T, H, 
                 (half_t*)w.data_ptr(), 
                 (half_t*)q.data_ptr(), 
                 (half_t*)k.data_ptr(), 
                 (half_t*)v.data_ptr(), 
                 (half_t*)z.data_ptr(), 
                 (half_t*)a.data_ptr(), 
                 (half_t*)dy.data_ptr(),
                 (float*)s.data_ptr(), 
                 (float*)sa.data_ptr(), 
                 (half_t*)dw.data_ptr(), 
                 (half_t*)dq.data_ptr(), 
                 (half_t*)dk.data_ptr(), 
                 (half_t*)dv.data_ptr(), 
                 (half_t*)dz.data_ptr(), 
                 (half_t*)da.data_ptr());
}

TORCH_LIBRARY(wind_backstepping, m) {
    m.def("forward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor(a!) y, Tensor(b!) s, Tensor(c!) sa) -> ()");
    m.def("backward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor dy, Tensor s, Tensor sa, Tensor(a!) dw, Tensor(b!) dq, Tensor(c!) dk, Tensor(d!) dv, Tensor(e!) dz, Tensor(f!) da) -> ()");
}

TORCH_LIBRARY_IMPL(wind_backstepping, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
}