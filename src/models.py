"""
#####################################################################################################################

    List of available models

    since 2025-03-14 model names may contain a directive that specialize their usage, it follows the
    proper name of the model, separated by the symbol '+'

#####################################################################################################################
"""

models                  = (                     # available models (first one is the default)
        "no-model",
        "gpt-3.5-turbo-instruct",
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-vision-preview",
        "gpt-4o-2024-05-13",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4o-mini+blank_img",                # variant that includes a blank image for text only news
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "facebook/chameleon-7b",
        "Qwen/Qwen2-VL-2B-Instruct",
        "Qwen/Qwen2-VL-7B-Instruct",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "claude-3-haiku-20240307",              # the less expensive
        "claude-3-5-haiku-20241022",            # cheap
        "claude-3-5-sonnet-20240620",           # medium cost
        "claude-3-7-sonnet-20250219",           # medium cost, the most recent one
        "claude-3-opus-20240229",               # high cost
)
models_endpoint         = {                     # which endpoint should be used for a model
        "no-model"                          : "chat",
        "gpt-3.5-turbo-instruct"            : "cmpl",
        "gpt-3.5-turbo"                     : "chat",
        "gpt-4"                             : "chat",
        "gpt-4-vision-preview"              : "chat",
        "gpt-4o-2024-05-13"                 : "chat",
        "gpt-4o"                            : "chat",
        "gpt-4o-mini"                       : "chat",
        "gpt-4o-mini+blank_img"             : "chat",
        "llava-hf/llava-v1.6-mistral-7b-hf" : "chat",
        "facebook/chameleon-7b"             : "cmpl",
        "Qwen/Qwen2-VL-2B-Instruct"         : "chat",
        "Qwen/Qwen2-VL-7B-Instruct"         : "chat",
        "google/gemma-3-4b-it"              : "chat",
        "google/gemma-3-12b-it"             : "chat",
        "claude-3-haiku-20240307"           : "chat",
        "claude-3-5-haiku-20241022"         : "chat",
        "claude-3-5-sonnet-20240620"        : "chat",
        "claude-3-7-sonnet-20250219"        : "chat",
        "claude-3-opus-20240229"            : "chat",
}
models_interface        = {                     # which interface should be used for a model
        "no-model"                          : "none",
        "gpt-3.5-turbo-instruct"            : "openai",
        "gpt-3.5-turbo"                     : "openai",
        "gpt-4"                             : "openai",
        "gpt-4-vision-preview"              : "openai",
        "gpt-4o-2024-05-13"                 : "openai",
        "gpt-4o"                            : "openai",
        "gpt-4o-mini"                       : "openai",
        "gpt-4o-mini+blank_img"             : "openai",
        "llava-hf/llava-v1.6-mistral-7b-hf" : "hf",
        "facebook/chameleon-7b"             : "hf",
        "Qwen/Qwen2-VL-2B-Instruct"         : "hf",
        "Qwen/Qwen2-VL-7B-Instruct"         : "hf",
        "google/gemma-3-4b-it"              : "hf",
        "google/gemma-3-12b-it"             : "hf",
        "claude-3-haiku-20240307"           : "anthro",
        "claude-3-5-haiku-20241022"         : "anthro",
        "claude-3-5-sonnet-20240620"        : "anthro",
        "claude-3-7-sonnet-20250219"        : "anthro",
        "claude-3-opus-20240229"            : "anthro",
}
models_short_name       = {                     # short name identifying a model, as used in statistics
        "gpt-3.5-turbo"                     : "gpt35",
        "gpt-4"                             : "gpt4",
        "gpt-4-vision-preview"              : "gpt4v",
        "gpt-4o"                            : "gpt4o",
        "gpt-4o-mini"                       : "gpt4om",
        "gpt-4o-mini+blank_img"             : "gpt4omb",
        "llava-hf/llava-v1.6-mistral-7b-hf" : "ll167b",
        "facebook/chameleon-7b"             : "cham7b",
        "Qwen/Qwen2-VL-2B-Instruct"         : "qwen2b",
        "Qwen/Qwen2-VL-7B-Instruct"         : "qwen7b",
        "google/gemma-3-4b-it"              : "gem4b",
        "google/gemma-3-12b-it"             : "gem12b",
        "claude-3-haiku-20240307"           : "cl3h",
        "claude-3-5-haiku-20241022"         : "cl3.5h",
        "claude-3-5-sonnet-20240620"        : "cl3.5s",
        "claude-3-7-sonnet-20250219"        : "cl3.5s",
        "claude-3-opus-20240229"            : "cl3o",
}
