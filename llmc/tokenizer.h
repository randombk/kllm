/*
Defines the GPT-2 Tokenizer.
Supports both encoding (text -> tokens) and decoding (tokens -> text).
*/

#include <stdint.h>
#include <ctype.h>
#include <assert.h>
#include <string.h>
// our own utilities
// defines fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "utils.h"

// ----------------------------------------------------------------------------

typedef struct {
    uint32_t vocab_size;
    char **token_table;
    int init_ok;
    int eot_token; // <|endoftext|> token id
    // for encoding
    uint32_t *token_lengths; // length of each token
    uint32_t max_token_length; // maximum token length
    uint32_t byte_token_start; // index where byte tokens start
} Tokenizer;

void safe_printf(const char *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // weird byte, don't print it
        }
    }
    printf("%s", piece);
}

void tokenizer_init(Tokenizer *tokenizer, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        // try to be more helpful as we just added this feature, erase later
        printf("---\n");
        printf("WARNING: Failed to open the tokenizer file %s\n", filename);
        printf("The Tokenizer is a new feature added April 14 2024.\n");
        printf("Re-run `python train_gpt2.py` to write it\n");
        printf("---\n");
        tokenizer->init_ok = 0;
        return;
    }
    // read in the header
    uint32_t header[256];
    freadCheck(header, sizeof(uint32_t), 256, file);
    assert(header[0] == 20240328);
    int version = header[1];
    tokenizer->vocab_size = header[2];
    if (version == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        assert(tokenizer->vocab_size == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } else if (version == 2) {
        tokenizer->eot_token = header[3];
    } else {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, version);
        exit(EXIT_FAILURE);
    }
    // read in all the tokens
    unsigned char length;
    tokenizer->token_table = (char **)mallocCheck(tokenizer->vocab_size * sizeof(char *));
    tokenizer->token_lengths = (uint32_t *)mallocCheck(tokenizer->vocab_size * sizeof(uint32_t));
    tokenizer->max_token_length = 0;
    tokenizer->byte_token_start = 0;
    
    // First pass: find where byte tokens start
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        freadCheck(&length, sizeof(unsigned char), 1, file);
        if (length == 1) {
            tokenizer->byte_token_start = i;
            break;
        }
        // Skip the token bytes for now
        fseek(file, length, SEEK_CUR);
    }
    // Reset file position
    fseek(file, sizeof(uint32_t) * 256, SEEK_SET);
    
    // Second pass: read all tokens
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        freadCheck(&length, sizeof(unsigned char), 1, file);
        assert(length > 0); // every token should be at least one character
        char *token_bytes = (char *)mallocCheck(length + 1);
        freadCheck(token_bytes, sizeof(char), length, file);
        token_bytes[length] = '\0';  // Add null terminator for printing
        tokenizer->token_table[i] = token_bytes;
        tokenizer->token_lengths[i] = length;
        if (length > tokenizer->max_token_length) {
            tokenizer->max_token_length = length;
        }
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    if (tokenizer->init_ok == 0) {
        return NULL;
    }
    if (token_id < tokenizer->vocab_size) {
        return tokenizer->token_table[token_id];
    } else {
        printf("invalid token id %u!\n", token_id);
        return NULL;
    }
}

// Helper function to find the longest matching token at a given position
uint32_t find_longest_token(Tokenizer *tokenizer, const char *text, size_t text_len, size_t pos) {
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

// Encode text into tokens
void tokenizer_encode(Tokenizer *tokenizer, const char *text, uint32_t *tokens, size_t *num_tokens) {
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

void tokenizer_free(Tokenizer *tokenizer) {
    if (tokenizer->init_ok) {
        for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            free(tokenizer->token_table[i]);
        }
        free(tokenizer->token_table);
        free(tokenizer->token_lengths);
    }
}
