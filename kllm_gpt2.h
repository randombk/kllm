#ifndef KLLM_GPT2_H
#define KLLM_GPT2_H

#include <linux/types.h>
#include <linux/firmware.h>

// GPT-2 model configuration
struct gpt2_config {
    int max_seq_len;    // max sequence length, e.g. 1024
    int vocab_size;     // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers;     // number of layers, e.g. 12
    int num_heads;      // number of heads in attention, e.g. 12
    int channels;       // number of channels, e.g. 768
};

// Parameter tensors for the model
#define NUM_PARAMETER_TENSORS 16
struct parameter_tensors {
    float *wte;         // (V, C)
    float *wpe;         // (maxT, C)
    float *ln1w;        // (L, C)
    float *ln1b;        // (L, C)
    float *qkvw;        // (L, 3*C, C)
    float *qkvb;        // (L, 3*C)
    float *attprojw;    // (L, C, C)
    float *attprojb;    // (L, C)
    float *ln2w;        // (L, C)
    float *ln2b;        // (L, C)
    float *fcw;         // (L, 4*C, C)
    float *fcb;         // (L, 4*C)
    float *fcprojw;     // (L, C, 4*C)
    float *fcprojb;     // (L, C)
    float *lnfw;        // (C)
    float *lnfb;        // (C)
};

// Activation tensors for the model
#define NUM_ACTIVATION_TENSORS 16
struct activation_tensors {
    float *encoded;     // (B, T, C)
    float *ln1;         // (L, B, T, C)
    float *qkv;         // (L, B, T, 3*C)
    float *atty;        // (L, B, T, C)
    float *preatt;      // (L, B, NH, T, T)
    float *att;         // (L, B, NH, T, T)
    float *attproj;     // (L, B, T, C)
    float *residual2;   // (L, B, T, C)
    float *ln2;         // (L, B, T, C)
    float *fch;         // (L, B, T, 4*C)
    float *fch_gelu;    // (L, B, T, 4*C)
    float *fcproj;      // (L, B, T, C)
    float *residual3;   // (L, B, T, C)
    float *lnf;         // (B, T, C)
    float *logits;      // (B, T, V)
    float *probs;       // (B, T, V)
};

// Tokenizer structure
struct tokenizer {
    uint32_t vocab_size;
    char **token_table;
    int init_ok;
    int eot_token; // <|endoftext|> token id
    uint32_t *token_lengths; // length of each token
    uint32_t max_token_length; // maximum token length
    uint32_t byte_token_start; // index where byte tokens start
};

// Main GPT-2 model structure
struct gpt2_model {
    struct gpt2_config config;
    struct parameter_tensors params;
    struct activation_tensors acts;
    float *params_memory;
    float *acts_memory;
    int *inputs;
    size_t num_parameters;
    size_t num_activations;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    int seq_len;
    struct tokenizer tokenizer; // Store initialized tokenizer
};

// Tokenizer function declarations
void tokenizer_init(struct tokenizer *tokenizer, const struct firmware *fw);
const char *tokenizer_decode(struct tokenizer *tokenizer, uint32_t token_id);
void tokenizer_encode(struct tokenizer *tokenizer, const char *text, uint32_t *tokens, size_t *num_tokens);
void tokenizer_free(struct tokenizer *tokenizer);

// Function declarations
int gpt2_build_from_firmware(struct gpt2_model *model, const struct firmware *model_fw, const struct firmware *tokenizer_fw);
void gpt2_forward(struct gpt2_model *model, int *inputs, int *targets, size_t T);
void gpt2_free(struct gpt2_model *model);
int gpt2_generate_next_token(struct gpt2_model *model, const char *prompt, char *output);

#endif /* KLLM_GPT2_H */ 