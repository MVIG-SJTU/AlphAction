#include "ROIAlign3d.h"
#include "ROIPool3d.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_align_3d_forward",&ROIAlign3d_forward, "ROIAlign3d_forward");
  m.def("roi_align_3d_backward",&ROIAlign3d_backward, "ROIAlign3d_backward");
  m.def("roi_pool_3d_forward", &ROIPool3d_forward, "ROIPool3d_forward");
  m.def("roi_pool_3d_backward", &ROIPool3d_backward, "ROIPool3d_backward");
}
