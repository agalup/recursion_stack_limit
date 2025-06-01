#include <cstdio>
#include <cuda_runtime.h>
#include <cuda.h>

#define CHECK(r, msg)\
{\
    if (r != CUDA_SUCCESS) {\
        const char* s;  cuGetErrorString(r,&s);\
        fprintf(stderr,"[%d] ERROR <%s> %d: %s\n", __LINE__, msg, r, s);\
        std::exit(1);\
    }\
}

// A dummy struct just for demonstration
struct Runtime {
    int dummy;
};

// Forward-declare a device function
__device__ void dev_app_impl(Runtime* rs, int depth, int iteration, int num_threads);

// Define a function-pointer type that matches dev_app_impl
using dev_app_fn_t = void(*)(Runtime*, int, int, int);

// ---------------------------------------------------------------------------
// 1) The INCORRECT pattern: Store the function's address in a __constant__ variable
//    This is what often fails under MPS / compute-sanitizer
// ---------------------------------------------------------------------------
__constant__ dev_app_fn_t dev_symbol_ptr = dev_app_impl;

// ---------------------------------------------------------------------------
// 2) A kernel that expects a pointer to a function pointer in device memory
//    but we are actually passing the address of `dev_symbol_ptr` itself
// ---------------------------------------------------------------------------
__global__
void kernel_call_dev_app(dev_app_fn_t* dev_ptr, Runtime* rs, int depth, int iteration, int num_threads)
{
    // This read may be invalid if dev_ptr doesn't truly point to allocated memory
    dev_app_fn_t fn = *dev_ptr;  // <--- out-of-bounds read likely

    // Call the device function (if the pointer was valid, it would do recursion)
    fn(rs, depth, iteration, num_threads);

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("[device] Thread %d finished kernel_call_dev_app\n", tid);
}

// ---------------------------------------------------------------------------
// 3) The device function that does naive recursion
// ---------------------------------------------------------------------------
__forceinline__
__device__
void dev_app_impl(Runtime* rs, int depth, int iteration, int num_threads)
{
    if (iteration < depth) {
        // Recurse
        dev_app_impl(rs, depth, iteration + 1, num_threads);
    }
    // Just print something
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("[device] tid=%d, iteration=%d, dummy=%d\n", tid, iteration, rs->dummy);
}

// ---------------------------------------------------------------------------
// 4) Host code
// ---------------------------------------------------------------------------
int main()
{
    // A) Retrieve the address of the __constant__ variable `dev_symbol_ptr`.
    //    This is NOT the correct pattern for dynamic function pointers.
    dev_app_fn_t* d_sym_ptr = nullptr;
    cudaGetSymbolAddress(reinterpret_cast<void**>(&d_sym_ptr), dev_symbol_ptr);

    printf("[host] d_sym_ptr = %p\n", (void*)d_sym_ptr);

    // B) Allocate/copy a Runtime struct
    Runtime h_rt;
    h_rt.dummy = 999;
    Runtime* d_rt = nullptr;
    cudaMalloc(&d_rt, sizeof(Runtime));
    cudaMemcpy(d_rt, &h_rt, sizeof(Runtime), cudaMemcpyHostToDevice);

    // C) Increase the device thread stack size to support deeper recursion
    size_t current_limit = 0;
    cudaDeviceGetLimit(&current_limit, cudaLimitStackSize);
    size_t new_limit = current_limit * 8; // grow stack to handle deeper recursion
    cudaDeviceSetLimit(cudaLimitStackSize, new_limit);

    // D) Launch the kernel
    kernel_call_dev_app<<<1,1>>>(d_sym_ptr, d_rt, 18, 0, 1);
    CHECK((CUresult)cudaDeviceSynchronize(), "after kernel_call_dev_app launch");

    cudaFree(d_rt);

    return 0;
}

