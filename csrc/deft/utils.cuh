#ifndef DEFT_UTILS_CUH_
#define DEFT_UTILS_CUH_
__device__ __forceinline__ float atomicMax(float *addr, float value) {
  float old = *addr, assumed;
  if (old >= value) return old;
  do {
    assumed = old;
    if (assumed > value) break;
    old = atomicCAS((unsigned int *)addr, __float_as_int(assumed),
                    __float_as_int(value));

  } while (old != assumed);
  return old;
}
#endif  // DEFT_UTILS_CUH_
