#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>
#include <x86intrin.h>

#define NCOEF 10000L
#define NUM_TRIALS 100000

#if defined(__GNUC__) || defined(__clang__)
#define NOINLINE __attribute__((noinline))
#define COMPILER_BARRIER() __asm__ __volatile__("" ::: "memory")
#else
#define NOINLINE
#define COMPILER_BARRIER() ((void)0)
#endif

typedef uint64_t bench_t;

static inline bench_t cycles_begin(void) {
    _mm_lfence();
    bench_t t = __rdtsc();
    _mm_lfence();
    return t;
}

static inline bench_t cycles_end(void) {
    unsigned int aux;
    bench_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static NOINLINE float horner_float(float x, const float *coef, long size) {
    float acc = 0.0f;
    for (long i = 0; i < size; ++i) {
        acc = (acc + coef[i]) * x;
    }
    return acc;
}

static NOINLINE float horner_avx_float(float x, const float *coef, long size) {
    if (size <= 0) return 0.0f;
    if ((size & 7L) != 0) return horner_float(x, coef, size);

    const long groups = size / 8;
    float x2 = x * x;
    float x4 = x2 * x2;
    float x8 = x4 * x4;

    __m256 vx8 = _mm256_set1_ps(x8);
    __m256 y = _mm256_setzero_ps();

    for (long g = 0; g < groups - 1; ++g) {
        __m256 c = _mm256_load_ps(coef + 8 * g);
        y = _mm256_add_ps(y, c);
        y = _mm256_mul_ps(y, vx8);
    }

    __m256 c = _mm256_load_ps(coef + 8 * (groups - 1));
    y = _mm256_add_ps(y, c);

    _Alignas(32) float lane[8];
    _mm256_store_ps(lane, y);

    float result = 0.0f;
    float power = x;
    for (int k = 7; k >= 0; --k) {
        result += lane[k] * power;
        power *= x;
    }
    return result;
}

static void benchmark(const char *name, float (*fn)(float,const float*,long), float x, const float *coef, long size, int trials) {
    volatile float sink = 0.0f;
    for (int i=0;i<10;++i) sink = fn(x,coef,size);
    COMPILER_BARRIER();
    double t0 = now_seconds();
    bench_t c0 = cycles_begin();
    for (int j=0;j<trials;++j) {
        COMPILER_BARRIER();
        sink = fn(x,coef,size);
    }
    bench_t c1 = cycles_end();
    double t1 = now_seconds();
    double dt=t1-t0;
    bench_t cyc=c1-c0;
    printf("%-18s result=% .9e time=%f s cycles=%llu ns/call=%f cycles/call=%f cycles/coef=%f\n",
      name,(double)sink,dt,(unsigned long long)cyc,dt*1e9/trials,(double)cyc/trials,(double)cyc/(trials*size));
}

int main(void) {
    const long size=NCOEF;
    const int trials=NUM_TRIALS;
    const float x=1.0001f;

    float *coef = (float*)_mm_malloc((size_t)size*sizeof(float),32);
    if(!coef){perror("_mm_malloc"); return EXIT_FAILURE;}
    srand(12345u);
    for(long i=0;i<size;++i) coef[i]=(float)(rand()%1000)/1000.0f;

#if defined(__GNUC__) || defined(__clang__)
    if (!__builtin_cpu_supports("avx")) {
        fprintf(stderr, "Esta CPU no soporta AVX.\n");
        _mm_free(coef);
        return EXIT_FAILURE;
    }
    printf("CPU AVX support: yes\n");
#endif

    float rs=horner_float(x,coef,size);
    float rv=horner_avx_float(x,coef,size);
    printf("scalar=% .9e avx=% .9e abs_err=%g rel_err=%g\n",(double)rs,(double)rv,(double)fabsf(rs-rv),(double)(fabsf(rs-rv)/(fabsf(rs)+1e-30f)));
    benchmark("Scalar float",horner_float,x,coef,size,trials);
    benchmark("AVX float",horner_avx_float,x,coef,size,trials);
    _mm_free(coef);
    return 0;
}
