#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/firmware.h>
#include <linux/math.h>
#include <linux/math64.h>
#include <linux/vmalloc.h>
#include <asm/fpu/api.h>
#include "kllm_gpt2.h"

/* 
 * Implementation of square root for float values 
 * using integer square root and bit manipulation
 */
void kernel_sqrt(float x, float *result)
{
    /* Handle special cases */
    if (x < 0.0f) {
        *result = 0;
        return;  /* Not a number does not exist in kernel */
    }
    if (x == 0.0f || x == 1.0f) {
        *result = x;
        return;
    }
    
    /* Extract the binary representation of the float */
    u32 i = *(u32*)&x;
    
    /* Handle special case for infinity */
    if (i == 0x7f800000) {
        *result = x;
        return;
    }
    
    /* Extract the exponent and mantissa */
    int exp = ((i >> 23) & 0xff) - 127;
    u32 mantissa = i & 0x7fffff;
    
    /* Convert float to a fixed-point integer representation */
    u32 int_val = (mantissa | 0x800000) << 8;
    
    /* Calculate integer square root using kernel's int_sqrt function */
    u32 sqrt_int = int_sqrt(int_val) << 8;
    
    /* Adjust the exponent for the square root */
    int new_exp = (exp >> 1) + 127;
    if (exp & 1) /* Odd exponent */
        sqrt_int = (sqrt_int * 0xb504) >> 16; /* Approximate sqrt(2) = 1.414 */
    
    /* Combine the parts back into a float */
    u32 result_i = (new_exp << 23) | ((sqrt_int >> 8) & 0x7fffff);
    
    /* Convert back to float */
    *result = *(float*)&result_i;
}

/*
 * Implementation of exponential function for float values
 * using integer math and bit manipulation
 */
void kernel_exp(float x, float *result)
{
    /* Handle special cases */
    if (x == 0.0f) {
        *result = 1.0f;
        return;
    }
    if (x == 1.0f) {
        *result = 2.71828182f; /* e */
        return;
    }
    if (x >= 88.0f) {        /* Beyond this, we'd overflow a float */
        *result = 0x7F800000; /* INFINITY */
        return;
    }
    if (x <= -88.0f) {
        *result = 0.0f;
        return;
    }

    /* Constants for Taylor series approximation */
    const float ln2 = 0.693147181f;  /* Natural log of 2 */
    
    /* Break down input into integer and fractional parts */
    int int_part = (int)x;
    float frac_part = x - int_part;
    
    /* 
     * Calculate 2^int_part by directly manipulating the IEEE 754 exponent.
     * This is a much faster approach than repeated multiplication.
     */
    int exp_bias = 127;  /* IEEE 754 single-precision exponent bias */
    u32 int_pow_bits = (u32)(int_part + exp_bias) << 23;
    float int_pow = *(float*)&int_pow_bits;
    
    /* 
     * Calculate e^frac_part using Taylor series approximation:
     * e^x ≈ 1 + x + x²/2! + x³/3! + x⁴/4! + ...
     * We'll use first 8 terms for decent accuracy
     */
    float res = 1.0f;
    float term = 1.0f;
    float frac_pow = frac_part;
    
    for (int i = 1; i <= 8; i++) {
        term *= frac_pow / i;
        res += term;
        
        /* If term is very small, we can stop */
        if (term < 0.0000001f)
            break;
    }
    
    /* Combine integer and fractional ress using e^(a+b) = e^a * e^b */
    res *= int_pow;
    
    *result = res;
}

/*
 * Implementation of natural logarithm for float values
 * using bit manipulation and series approximation
 */
void kernel_log(float x, float *result)
{
    /* Handle special cases */
    if (x <= 0.0f) {
        *result = 0;  /* Not a number for negative inputs */
        return;
    }
    if (x == 1.0f) {
        *result = 0.0f;
        return;
    }
    if (x == 2.71828182f) { /* e */
        *result = 1.0f;
        return;
    }
    
    /* Extract the IEEE 754 components */
    u32 i = *(u32*)&x;
    int exp_biased = (i >> 23) & 0xff;
    int exponent = exp_biased - 127;  /* Unbiased exponent */
    u32 mantissa = i & 0x7fffff;
    
    /* 
     * Calculate log(x) = log(2^exponent * (1.mantissa)) 
     *                  = exponent * log(2) + log(1.mantissa)
     */
    float log2 = 0.693147181f;  /* ln(2) */
    
    /* Normalize x to range [1, 2) */
    float normalized = 1.0f + ((float)mantissa / 0x800000);
    
    /* 
     * Calculate log(normalized) using polynomial approximation.
     * We'll use a minimax polynomial approximation for [1, 2) range.
     * ln(y) ≈ 2 * sum_{k=0}^∞ ((y-1)/(y+1))^(2k+1) / (2k+1)
     */
    float y_minus_1 = normalized - 1.0f;
    float y_plus_1 = normalized + 1.0f;
    float z = y_minus_1 / y_plus_1;
    float z_squared = z * z;
    float z_power = z;
    float log_sum = z;  /* First term */
    
    /* Add more terms of the series */
    for (int k = 1; k < 10; k++) {  /* 10 terms for good precision */
        z_power *= z_squared;
        float term = z_power / (2 * k + 1);
        log_sum += term;
        
        /* If term is very small, we can stop */
        if (term < 0.0000001f)
            break;
    }
    
    /* Multiply by 2 and add the exponent contribution */
    float res = 2.0f * log_sum + exponent * log2;
    
    *result = res;
}


/*
 * Implementation of hyperbolic tangent for float values
 * tanh(x) = (e^x - e^-x) / (e^x + e^-x)
 */
void kernel_tanh(float x, float *result)
{
    /* Handle special cases */
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

// Forward declarations of internal functions
static void fill_in_parameter_sizes(size_t *param_sizes, struct gpt2_config config);
static void fill_in_activation_sizes(size_t *act_sizes, struct gpt2_config config, int B, int T);
static float *malloc_and_point_parameters(struct parameter_tensors *params, size_t *param_sizes);
static float *malloc_and_point_activations(struct activation_tensors *acts, size_t *act_sizes);

// Layer forward pass implementations
static void encoder_forward(float *out, int *inp, float *wte, float *wpe, int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *out_bt = out + b * T * C + t * C;
            int ix = inp[b * T + t];
            float *wte_ix = wte + ix * C;
            float *wpe_t = wpe + t * C;
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

static void layernorm_forward(float *out, float *inp, float *weight, float *bias, int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *x = inp + b * T * C + t * C;
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            float ret;
            kernel_sqrt(v + eps, &ret);
            float s = 1.0f / ret;
            float *out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m));
                float o = n * weight[i] + bias[i];
                out_bt[i] = o;
            }
        }
    }
}

static void matmul_forward(float *out, const float *inp, const float *weight, const float *bias,
                          int B, int T, int C, int OC) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int bt = b * T + t;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                for (int i = 0; i < C; i++) {
                    float weight_val = weight[o*C + i];
                    float inp_val = inp[bt * C + i];
                    float res;
                    val += inp_val * weight_val;
                }
                out[bt * OC + o] = val;
            }
        }
    }
}

static void attention_forward(float *out, float *preatt, float *att, float *inp,
                            int B, int T, int C, int NH) {
    int C3 = C*3;
    int hs = C / NH;
    float ret;
    kernel_sqrt(hs, &ret);
    float scale = 1.0f / ret;

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float *query_t = inp + b * T * C3 + t * C3 + h * hs;
                float *att_bth = att + b*NH*T*T + h*T*T + t*T;

                float maxval = -10000.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float *key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) maxval = val;
                    att_bth[t2] = val;
                }

                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv;
                    kernel_exp(att_bth[t2] - maxval, &expv);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = 1.0f / expsum;

                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        att_bth[t2] = 0.0f;
                    }
                }

                float *out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    float *value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

#define GELU_SCALING_FACTOR 0.7978845608028654f
static void gelu_forward(float *out, float *inp, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float res;
        kernel_tanh(GELU_SCALING_FACTOR * (x + cube), &res);
        out[i] = 0.5f * x * (1.0f + res);
    }
}

static void residual_forward(float *out, float *inp1, float *inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

static void softmax_forward(float *probs, float *logits, int B, int T, int V, int Vp) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *logits_bt = logits + b * T * Vp + t * Vp;
            float *probs_bt = probs + b * T * Vp + t * Vp;

            float maxval = -10000.0f;
            for (int i = 0; i < V; i++) {
                if (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            float sum = 0.0f;
            for (int i = 0; i < V; i++) {
                float res;
                kernel_exp(logits_bt[i] - maxval, &res);
                probs_bt[i] = res;
                sum += probs_bt[i];
            }
            float sum_inv = 1.0f / sum;
            for (int i = 0; i < V; i++) {
                probs_bt[i] *= sum_inv;
            }
            for (int i = V; i < Vp; i++) {
                probs_bt[i] = 0.0f;
            }
        }
    }
}

// Parameter size calculation
static void fill_in_parameter_sizes(size_t *param_sizes, struct gpt2_config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    param_sizes[0] = Vp * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * (3 * C) * C; // qkvw
    param_sizes[5] = L * (3 * C); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C); // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb
}

// Activation size calculation
static void fill_in_activation_sizes(size_t *act_sizes, struct gpt2_config config, int B, int T) {
    size_t C = config.channels;
    size_t NH = config.num_heads;
    size_t L = config.num_layers;
    size_t Vp = config.padded_vocab_size;
    act_sizes[0] = B * T * C; // encoded
    act_sizes[1] = L * B * T * C; // ln1
    act_sizes[2] = L * B * T * 3 * C; // qkv
    act_sizes[3] = L * B * T * C; // atty
    act_sizes[4] = L * B * NH * T * T; // preatt
    act_sizes[5] = L * B * NH * T * T; // att
    act_sizes[6] = L * B * T * C; // attproj
    act_sizes[7] = L * B * T * C; // residual2
    act_sizes[8] = L * B * T * C; // ln2
    act_sizes[9] = L * B * T * 4 * C; // fch
    act_sizes[10] = L * B * T * 4 * C; // fch_gelu
    act_sizes[11] = L * B * T * C; // fcproj
    act_sizes[12] = L * B * T * C; // residual3
    act_sizes[13] = B * T * C; // lnf
    act_sizes[14] = B * T * Vp; // logits
    act_sizes[15] = B * T * Vp; // probs
}

// Memory allocation helpers
static float *malloc_and_point_parameters(struct parameter_tensors *params, size_t *param_sizes) {
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    float *params_memory = vmalloc(num_parameters * sizeof(float));
    if (!params_memory)
        return NULL;

    float **ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    float *params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

static float *malloc_and_point_activations(struct activation_tensors *acts, size_t *act_sizes) {
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += act_sizes[i];
    }
    float *acts_memory = vmalloc(num_activations * sizeof(float));
    if (!acts_memory)
        return NULL;

    float **ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2,
        &acts->ln2, &acts->fch, &acts->fch_gelu, &acts->fcproj,
        &acts->residual3, &acts->lnf, &acts->logits, &acts->probs
    };
    float *acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

// Main model functions
int gpt2_build_from_firmware(struct gpt2_model *model, const struct firmware *model_fw, const struct firmware *tokenizer_fw) {
    if (!model_fw || !model_fw->data || !model_fw->size) {
        pr_err("Invalid model firmware data\n");
        return -EINVAL;
    }

    if (!tokenizer_fw || !tokenizer_fw->data || !tokenizer_fw->size) {
        pr_err("Invalid tokenizer firmware data\n");
        return -EINVAL;
    }

    // Initialize tokenizer first
    tokenizer_init(&model->tokenizer, tokenizer_fw);
    if (!model->tokenizer.init_ok) {
        pr_err("Failed to initialize tokenizer\n");
        return -EINVAL;
    }

    // Read model header
    const int *model_header = (const int *)model_fw->data;
    if (model_header[0] != 20240326) {
        pr_err("Bad magic model file\n");
        return -EINVAL;
    }
    if (model_header[1] != 3) {
        pr_err("Bad version in model file\n");
        return -EINVAL;
    }

    // Read hyperparameters
    model->config.max_seq_len = model_header[2];
    model->config.vocab_size = model_header[3];
    model->config.num_layers = model_header[4];
    model->config.num_heads = model_header[5];
    model->config.channels = model_header[6];
    model->config.padded_vocab_size = model_header[7];

    pr_info("[GPT-2] max_seq_len: %d\n", model->config.max_seq_len);
    pr_info("[GPT-2] vocab_size: %d\n", model->config.vocab_size);
    pr_info("[GPT-2] padded_vocab_size: %d\n", model->config.padded_vocab_size);
    pr_info("[GPT-2] num_layers: %d\n", model->config.num_layers);
    pr_info("[GPT-2] num_heads: %d\n", model->config.num_heads);
    pr_info("[GPT-2] channels: %d\n", model->config.channels);

    // Allocate space for parameters
    fill_in_parameter_sizes(model->param_sizes, model->config);
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    pr_info("[GPT-2] num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters;

    // Allocate and copy parameters
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    if (!model->params_memory) {
        pr_err("Failed to allocate parameter memory\n");
        return -ENOMEM;
    }

    // Copy parameters from firmware
    memcpy(model->params_memory, model_fw->data + sizeof(int) * 256, num_parameters * sizeof(float));

    // Initialize other fields
    model->acts_memory = NULL;
    model->inputs = NULL;
    model->batch_size = 0;
    model->seq_len = 0;

    return 0;
}

void gpt2_forward(struct gpt2_model *model, int *inputs, int *targets, size_t B, size_t T) {
    pr_info("[GPT-2] Beginning forward pass\n");
    if (!model->params_memory) {
        pr_err("Model not initialized properly\n");
        return;
    }

    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    // Validate inputs
    for(int i = 0; i < B * T; i++) {
        if (!(0 <= inputs[i] && inputs[i] < V)) {
            pr_err("Invalid input token: %d\n", inputs[i]);
            return;
        }
    }

    // Allocate activations if needed
    if (!model->acts_memory) {
        model->batch_size = B;
        model->seq_len = T;
        fill_in_activation_sizes(model->act_sizes, model->config, B, T);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        pr_info("[GPT-2] num_activations: %zu\n", num_activations);
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        if (!model->acts_memory) {
            pr_err("Failed to allocate activation memory\n");
            return;
        }
        model->inputs = vmalloc(B * T * sizeof(int));
        if (!model->inputs) {
            pr_err("Failed to allocate input memory\n");
            return;
        }
    } else if (B != model->batch_size || T != model->seq_len) {
        pr_err("Invalid batch size or sequence length\n");
        return;
    }

    // Cache inputs
    memcpy(model->inputs, inputs, B * T * sizeof(int));

    // Forward pass
    struct parameter_tensors params = model->params;
    struct activation_tensors acts = model->acts;

    cond_resched();
    kernel_fpu_begin();
    float *residual;

    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C);
    for (int l = 0; l < L; l++) {
        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        float *l_ln1w = params.ln1w + l * C;
        float *l_ln1b = params.ln1b + l * C;
        float *l_qkvw = params.qkvw + l * 3*C * C;
        float *l_qkvb = params.qkvb + l * 3*C;
        float *l_attprojw = params.attprojw + l * C * C;
        float *l_attprojb = params.attprojb + l * C;
        float *l_ln2w = params.ln2w + l * C;
        float *l_ln2b = params.ln2b + l * C;
        float *l_fcw = params.fcw + l * 4*C * C;
        float *l_fcb = params.fcb + l * 4*C;
        float *l_fcprojw = params.fcprojw + l * C * 4*C;
        float *l_fcprojb = params.fcprojb + l * C;

        float *l_ln1 = acts.ln1 + l * B * T * C;
        float *l_qkv = acts.qkv + l * B * T * 3*C;
        float *l_atty = acts.atty + l * B * T * C;
        float *l_preatt = acts.preatt + l * B * NH * T * T;
        float *l_att = acts.att + l * B * NH * T * T;
        float *l_attproj = acts.attproj + l * B * T * C;
        float *l_residual2 = acts.residual2 + l * B * T * C;
        float *l_ln2 = acts.ln2 + l * B * T * C;
        float *l_fch = acts.fch + l * B * T * 4*C;
        float *l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        float *l_fcproj = acts.fcproj + l * B * T * C;
        float *l_residual3 = acts.residual3 + l * B * T * C;

        kernel_fpu_end();

        cond_resched();
        pr_info("[GPT-2] layer %d: layernorm 1\n", l+1);
        kernel_fpu_begin();
        layernorm_forward(l_ln1, residual, l_ln1w, l_ln1b, B, T, C);
        kernel_fpu_end();

        cond_resched();
        pr_info("[GPT-2] layer %d: matmul 1\n", l+1);
        kernel_fpu_begin();
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        kernel_fpu_end();

        cond_resched();
        pr_info("[GPT-2] layer %d: attention\n", l+1);
        kernel_fpu_begin();
        attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
        kernel_fpu_end();

        cond_resched();
        pr_info("[GPT-2] layer %d: matmul 2\n", l+1);
        kernel_fpu_begin();
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        kernel_fpu_end();

        cond_resched();
        pr_info("[GPT-2] layer %d: residual 1\n", l+1);
        kernel_fpu_begin();
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        kernel_fpu_end();

        cond_resched();
        pr_info("[GPT-2] layer %d: layernorm 2\n", l+1);
        kernel_fpu_begin();
        layernorm_forward(l_ln2, l_residual2, l_ln2w, l_ln2b, B, T, C);
        kernel_fpu_end();

        cond_resched();
        pr_info("[GPT-2] layer %d: matmul 3\n", l+1);
        kernel_fpu_begin();
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        kernel_fpu_end();

        cond_resched();
        pr_info("[GPT-2] layer %d: gelu\n", l+1);
        kernel_fpu_begin();
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        kernel_fpu_end();

        cond_resched();
        pr_info("[GPT-2] layer %d: matmul 4\n", l+1);
        kernel_fpu_begin();
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        kernel_fpu_end();

        cond_resched();
        pr_info("[GPT-2] layer %d: residual 2\n", l+1);
        kernel_fpu_begin();
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
        kernel_fpu_end();

        cond_resched();
        pr_info("[GPT-2] layer %d: layernorm 3\n", l+1);
        kernel_fpu_begin();
    }

    residual = acts.residual3 + (L-1) * B * T * C;
    kernel_fpu_end();

    cond_resched();
    pr_info("[GPT-2] layernorm_forward\n");
    kernel_fpu_begin();
    layernorm_forward(acts.lnf, residual, params.lnfw, params.lnfb, B, T, C);
    kernel_fpu_end();

    cond_resched();
    pr_info("[GPT-2] matmul_forward\n");
    kernel_fpu_begin();
    matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, Vp);
    kernel_fpu_end();

    cond_resched();
    pr_info("[GPT-2] softmax_forward\n");
    kernel_fpu_begin();
    softmax_forward(acts.probs, acts.logits, B, T, V, Vp);
    kernel_fpu_end();
}

void gpt2_free(struct gpt2_model *model) {
    if (model->params_memory)
        vfree(model->params_memory);
    if (model->acts_memory)
        vfree(model->acts_memory);
    if (model->inputs)
        vfree(model->inputs);
    if (model->tokenizer.init_ok)
        tokenizer_free(&model->tokenizer);
}

// Random number generation for sampling
static unsigned int random_u32(uint64_t *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

static void random_f32(uint64_t *state, float *result) {
    *result = (random_u32(state) >> 8) / 16777216.0f;
}

static int sample_mult(float *probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}

// High-level generation function
int gpt2_generate_next_token(struct gpt2_model *model, const char *prompt, char *output) {
    if (!model->params_memory) {
        pr_err("Model not initialized properly\n");
        return -EINVAL;
    }

    if (!model->tokenizer.init_ok) {
        pr_err("Tokenizer not initialized properly\n");
        return -EINVAL;
    }

    // Setup generation parameters
    int B = 1; // batch size 1 for inference
    int T = model->config.max_seq_len; // use full sequence length
    int *gen_tokens = vmalloc(B * T * sizeof(int));
    if (!gen_tokens) {
        pr_err("Failed to allocate generation tokens\n");
        return -ENOMEM;
    }
    memset(gen_tokens, 0, B * T * sizeof(int));
    
    uint64_t rng_state = 1337; // fixed seed for reproducibility

    // Encode the prompt
    size_t num_prompt_tokens;
    tokenizer_encode(&model->tokenizer, prompt, gen_tokens, &num_prompt_tokens);
    if (num_prompt_tokens == 0) {
        pr_err("Failed to encode prompt\n");
        vfree(gen_tokens);
        return -EINVAL;
    }

    // Run the model forward pass
    gpt2_forward(model, gen_tokens, NULL, B, T);
    
    cond_resched();
    kernel_fpu_begin();
    // Get the next token probabilities
    float *probs = model->acts.probs + (num_prompt_tokens-1) * model->config.padded_vocab_size;
    float coin;
    random_f32(&rng_state, &coin);
    int next_token = sample_mult(probs, model->config.vocab_size, coin);
    kernel_fpu_end();
    cond_resched();
    
    // Stop if we hit the end token
    if (next_token == model->tokenizer.eot_token) {
        vfree(gen_tokens);
        return 0; // End of generation
    }
    
    // Decode the token to output
    const char *token_str = tokenizer_decode(&model->tokenizer, next_token);
    if (!token_str) {
        vfree(gen_tokens);
        return -EINVAL;
    }

    size_t token_len = strlen(token_str);
    memcpy(output, token_str, token_len);
    output[token_len] = '\0';
    
    // Cleanup
    vfree(gen_tokens);

    return strlen(token_str);
}

// Helper function to find the longest matching token at a given position
static uint32_t find_longest_token(struct tokenizer *tokenizer, const char *text, size_t text_len, size_t pos) {
    uint32_t best_token = tokenizer->vocab_size; // Use vocab_size as sentinel value
    size_t best_length = 0;
    
    // Try each token in the vocabulary
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        size_t token_len = tokenizer->token_lengths[i];
        // Skip if token is longer than remaining text
        if (pos + token_len > text_len) continue;
        
        // Compare token with text at current position
        if (strncmp(tokenizer->token_table[i], text + pos, token_len) == 0) {
            if (token_len > best_length) {
                best_length = token_len;
                best_token = i;
            }
        }
    }
    
    return best_token;
}

// Tokenizer implementation
void tokenizer_init(struct tokenizer *tokenizer, const struct firmware *fw) {
    if (!fw || !fw->data || !fw->size) {
        pr_err("Invalid firmware data\n");
        tokenizer->init_ok = 0;
        return;
    }

    // Read in the header
    const uint32_t *header = (const uint32_t *)fw->data;
    if (header[0] != 20240328) {
        pr_err("Bad magic tokenizer file\n");
        tokenizer->init_ok = 0;
        return;
    }
    int version = header[1];
    tokenizer->vocab_size = header[2];
    if (version == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        if (tokenizer->vocab_size != 50257) {
            pr_err("Bad vocab size for version 1 tokenizer\n");
            tokenizer->init_ok = 0;
            return;
        }
        tokenizer->eot_token = 50256;
    } else if (version == 2) {
        tokenizer->eot_token = header[3];
    } else {
        pr_err("Bad version in tokenizer file: %d\n", version);
        tokenizer->init_ok = 0;
        return;
    }

    // Read in all the tokens
    const unsigned char *data = (const unsigned char *)(fw->data + sizeof(uint32_t) * 256);
    size_t data_pos = 0;
    
    // Allocate memory for token table and lengths
    tokenizer->token_table = vmalloc(tokenizer->vocab_size * sizeof(char *));
    tokenizer->token_lengths = vmalloc(tokenizer->vocab_size * sizeof(uint32_t));
    if (!tokenizer->token_table || !tokenizer->token_lengths) {
        pr_err("Failed to allocate tokenizer memory\n");
        tokenizer->init_ok = 0;
        return;
    }
    
    tokenizer->max_token_length = 0;
    tokenizer->byte_token_start = 0;
    
    // First pass: find where byte tokens start
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        unsigned char length = data[data_pos++];
        if (length == 1) {
            tokenizer->byte_token_start = i;
            break;
        }
        data_pos += length;
    }
    
    // Reset data position
    data_pos = 0;
    
    // Second pass: read all tokens
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        unsigned char length = data[data_pos++];
        if (length == 0) {
            pr_err("Invalid token length 0\n");
            tokenizer->init_ok = 0;
            return;
        }
        
        char *token_bytes = vmalloc(length + 1);
        if (!token_bytes) {
            pr_err("Failed to allocate token memory\n");
            tokenizer->init_ok = 0;
            return;
        }
        
        memcpy(token_bytes, data + data_pos, length);
        token_bytes[length] = '\0';
        tokenizer->token_table[i] = token_bytes;
        tokenizer->token_lengths[i] = length;
        
        if (length > tokenizer->max_token_length) {
            tokenizer->max_token_length = length;
        }
        
        data_pos += length;
    }
    
    tokenizer->init_ok = 1;
}

const char *tokenizer_decode(struct tokenizer *tokenizer, uint32_t token_id) {
    if (tokenizer->init_ok == 0) {
        return NULL;
    }
    if (token_id < tokenizer->vocab_size) {
        return tokenizer->token_table[token_id];
    } else {
        pr_err("Invalid token id %u!\n", token_id);
        return NULL;
    }
}

void tokenizer_encode(struct tokenizer *tokenizer, const char *text, uint32_t *tokens, size_t *num_tokens) {
    if (tokenizer->init_ok == 0) {
        *num_tokens = 0;
        return;
    }
    
    size_t text_len = strlen(text);
    size_t pos = 0;
    size_t token_count = 0;
    const size_t max_tokens = 1024; // Reasonable limit for input text
    
    while (pos < text_len && token_count < max_tokens) {
        uint32_t token = find_longest_token(tokenizer, text, text_len, pos);
        
        if (token == tokenizer->vocab_size) {
            // No matching token found, encode as byte token
            unsigned char byte = text[pos];
            token = tokenizer->byte_token_start + byte;
            if (token >= tokenizer->vocab_size) {
                // If byte token is out of range, skip the character
                pos++;
                continue;
            }
        }
        
        tokens[token_count++] = token;
        pos += tokenizer->token_lengths[token];
    }
    
    *num_tokens = token_count;
}

void tokenizer_free(struct tokenizer *tokenizer) {
    if (tokenizer->init_ok) {
        for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            vfree(tokenizer->token_table[i]);
        }
        vfree(tokenizer->token_table);
        vfree(tokenizer->token_lengths);
    }
} 