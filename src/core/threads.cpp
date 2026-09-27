#include "threads.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace kessler {

int configure_threads(int requested) {
#ifdef _OPENMP
    if (requested > 0) omp_set_num_threads(requested);
    return omp_get_max_threads();
#else
    (void)requested;
    return 1;
#endif
}

}  // namespace kessler
