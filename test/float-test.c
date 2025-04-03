#include <stdio.h>
#include <math.h>
#include <stdint.h>

typedef uint32_t u32;

static void kernel_sqrt(float x, float *result) {
    // Handle special cases first
    if (x < 0.0f) {
        *result = 0.0f; // NaN does not work here
        return;
    }
    
    __asm__ __volatile__ (
        "flds %1\n\t"      // Load float x onto FPU stack
        "fsqrt\n\t"        // Compute square root
        "fstps %0\n\t"     // Store the result to *result
        : "=m" (*result)   // output
        : "m" (x)          // input
        : "st"             // clobbered FPU register
    );
}

static void kernel_exp(float x, float *result) {
    // Handle special cases first
    if (x > 88.0f) {
        // HACK: Some very high value.
        *result = 999999.0f;
        // *result = 0x7F800000; // INFINITY
        return;
    }
    if (x < -88.0f) {
        *result = 0.0f;
        return;
    }

    __asm__ __volatile__ (
        "flds %1\n\t"          // ST(0) = x
        "fldl2e\n\t"           // ST(0) = log2(e), ST(1) = x
        "fmulp\n\t"            // ST(0) = x * log2(e)
        
        "fld %%st(0)\n\t"      // ST(0) = x * log2(e), ST(1) = x * log2(e)
        "frndint\n\t"          // ST(0) = n (integer part), ST(1) = x*log2(e)
        "fxch\n\t"             // ST(0) = x*log2(e), ST(1) = n
        "fsub %%st(1), %%st\n\t"  // ST(0) = f = x*log2(e) - n, ST(1) = n
        "f2xm1\n\t"            // ST(0) = 2^f - 1, ST(1) = n
        "fld1\n\t"             // ST(0) = 1, ST(1) = 2^f - 1, ST(2) = n
        "faddp\n\t"            // ST(0) = 2^f, ST(1) = n
        "fscale\n\t"           // ST(0) = 2^f * 2^n = e^x
        "fstp %%st(1)\n\t"     // pop n from stack, leaving just e^x
        "fstps %0\n\t"         // Store result
        : "=m" (*result)
        : "m" (x)
        : "st", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)", 
          "eax", "ecx", "edx", "memory"
    );
}

static void kernel_tanh(float x, float *result) {
    if (x == 0.0f) {
        *result = 0.0f;
        return;
    }
    
    /* For large values, tanh(x) approaches +/-1 */
    if (x > 10.0f) {
        *result = 1.0f;
        return;
    }
    if (x < -10.0f) {
        *result = -1.0f;
        return;
    }

    /* 
     * Calculate tanh using the standard formula:
     * tanh(x) = (e^x - e^-x) / (e^x + e^-x)
     * To avoid potential overflow, we can rewrite this as:
     * tanh(x) = (1 - e^-2x) / (1 + e^-2x)  for x > 0
     * tanh(x) = (e^2x - 1) / (e^2x + 1)    for x < 0
     */
    float res;
    
    if (x > 0.0f) {
        float exp_neg_2x;
        kernel_exp(-2.0f * x, &exp_neg_2x);
        res = (1.0f - exp_neg_2x) / (1.0f + exp_neg_2x);
    } else {
        float exp_2x;
        kernel_exp(2.0f * x, &exp_2x);
        res = (exp_2x - 1.0f) / (exp_2x + 1.0f);
    }
    
    *result = res;
}


void test_sqrt(float x) {
    float actual = sqrtf(x);
    float result;
    kernel_sqrt(x, &result);
    printf("kernel_sqrt(%f) = %f, sqrtf(%f) = %f\n", x, result, x, actual);
}

void test_exp(float x) {
    float actual = expf(x);
    float result;
    kernel_exp(x, &result);
    printf("kernel_exp(%f) = %f, expf(%f) = %f\n", x, result, x, actual);
}

void test_tanh(float x) {
    float actual = tanhf(x);
    float result;
    kernel_tanh(x, &result);
    printf("kernel_tanh(%f) = %f, tanhf(%f) = %f\n", x, result, x, actual);
}

int main() {
    for (float x = -10.0f; x <= 10.0f; x += 0.1f) {
        test_sqrt(x);
    }
    for (float x = -10.0f; x <= 10.0f; x += 0.1f) {
        test_exp(x);
    }
    for (float x = -10.0f; x <= 10.0f; x += 0.1f) {
        test_tanh(x);
    }
    return 0;
}