#ifndef LLAMA_H
#define LLAMA_H

#include "llama_mock.h"

#endif

struct llama_sampler * llama_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root);
