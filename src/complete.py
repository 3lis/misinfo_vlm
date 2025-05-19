"""
#####################################################################################################################

    Module to handle model completions

#####################################################################################################################
"""

import  os
import  sys
import  time
import  random
import  gc
import  platform
from    PIL         import Image

key_file                = "../data/.key.txt"    # file with the current OpenAI API access key
hf_file                 = "../data/.hf.txt"     # file with the current huggingface access key
anthro_file             = "../data/.anth.txt"   # file with the current anthropic access key

native_res              = ( 672, 672 )          # image resolution for LLaVA-NeXT, Qwen2-VL-7B should be multiple of 28
llava_next_n_max        = 50                    # maximum number of returns for LLaVA-NeXT (due to GPU memory)
qwen2_vl_n_max          = 1                     # NOTE: Qwen2-VL-7B provide inconsisten results with more than 1!!

client                  = None                  # the language model client object
cnfg                    = None                  # parameter obj assigned by main_exec.py
delay                   = 120                   # delay in seconds after OpenAI/anthropic internal errors

# ===================================================================================================================
#
#   - set_hf_llava_next
#   - set_hf_chameleon
#   - set_hf_qwen
#   - set_hf_gemma
#   - set_hf
#   - set_openai
#   - set_anthro
#
# ===================================================================================================================

def set_hf_llava_next():
    """
    Return the LlavaNext client
    """
    global  torch
    from    transformers    import LlavaNextForConditionalGeneration, LlavaNextProcessor
    import  torch

    model           = LlavaNextForConditionalGeneration.from_pretrained(
            cnfg.model,
            torch_dtype=torch.float16,
            device_map="auto"
            )
    processor       = LlavaNextProcessor.from_pretrained( cnfg.model )
    client          = { "model": model, "processor": processor }
    return client


def set_hf_chameleon():
    """
    Return the Chameleon client
    """
    global  torch
    from    transformers    import ChameleonForConditionalGeneration, ChameleonProcessor
    import  torch

    model           = ChameleonForConditionalGeneration.from_pretrained(
            cnfg.model,
            torch_dtype         = torch.float16,
            repetition_penalty  = cnfg.repetition_penalty,
            device_map          = "auto"
            )

    processor       = ChameleonProcessor.from_pretrained( cnfg.model )
    processor.tokenizer.padding_side = "left"
    client          = { "model": model, "processor": processor }
    return client


def set_hf_qwen():
    """
    Return the Qwen client
    """
    global  torch
    from    transformers    import Qwen2VLForConditionalGeneration, AutoProcessor
    import  torch

    model           = Qwen2VLForConditionalGeneration.from_pretrained(
            cnfg.model,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",    # should install FlashAttention-2 and see if works
            device_map="auto"
            )
    processor       = AutoProcessor.from_pretrained( cnfg.model )
    client          = { "model": model, "processor": processor }
    return client


def set_hf_gemma():
    """
    Return the gemma client
    """
    global  torch
    from    transformers    import Gemma3ForConditionalGeneration, AutoProcessor
    import  torch

    model           = Gemma3ForConditionalGeneration.from_pretrained(
            cnfg.model,
# this patch was suggested in https://github.com/google-deepmind/gemma/issues/169
# and has the effect to suppress the "p.attn_bias_ptr is not correctly aligned"
# but to cause CUDA out of memory
#           attn_implementation="eager",
            device_map="auto"
            ).eval()
    processor       = AutoProcessor.from_pretrained( cnfg.model )
    processor.tokenizer.padding_side = "left"
    client          = { "model": model, "processor": processor }
    return client


def set_hf():
    """
    Parse the hugginface key and return the client
        NOTE: should be the first function to call before all others that use hugginface models
        NOTE: the client has two items: the model and the prompt processor
    """
    from    huggingface_hub import login

    key             = open( hf_file, 'r' ).read().rstrip()
    login( token=key )

    if "llava-v1.6" in cnfg.model:
        return set_hf_llava_next()
    if "gemma" in cnfg.model:
        return set_hf_gemma()
    if "chameleon" in cnfg.model:
        return set_hf_chameleon()
    if "Qwen" in cnfg.model:
        return set_hf_qwen()

    return None


def set_openai():
    """
    Parse the OpenAI key and return the client
        NOTE: should be the first function to call before all others that use openai
    """
    from    openai          import OpenAI
    key             = open( key_file, 'r' ).read().rstrip()
    client          = OpenAI( api_key=key )
    return client


def set_anthro():
    """
    Parse the anthropic key and return the client
        NOTE: should be the first function to call before all others that use anthropic
    """
    import anthropic
    key             = open( anthro_file, 'r' ).read().rstrip()
    client          = anthropic.Anthropic( api_key=key )
    return client


# ===================================================================================================================
#
#   - complete_anthro
#   - complete_openai
#   - complete_llava
#   - complete_chameleon
#   - complete_qwen
#   - complete_gemma
#   - complete_hf
#
#   - do_complete
#
# ===================================================================================================================

def complete_anthro( prompt ):
    """
    Feed a prompt to an anthropic model and get the list of completions returned.

    params:
        prompt      [str] or [list] the prompt for completion-mode models,
                    or the messages for chat-mode models

    return:         [list] with completions [str]
    """
    global client
    if cnfg.DEBUG:  return [ "test_only" ]

    if client is None:              # check if anthropic has already a client, otherwise set it
        client  = set_anthro()
 
    # arguments for completion calls
    cargs   = {
            "messages"          : prompt,
            "model"             : cnfg.model,
            "max_tokens"        : cnfg.max_tokens,
            "top_p"             : cnfg.top_p,
            "temperature"       : cnfg.temperature,
    }

    # there have been anthropic._exceptions.OverloadedError errors, this is a very simple workaround
    # if the problem will repeat more often, and is not solved, the solution is exponential backoff
    try:
        res     = client.messages.create( **cargs )
    except Exception as e:              # catch EVERY exception to ensure compatibility with OpenAI/anthropic versions
        if cnfg.VERBOSE:
            print( f"catched error {e}, sleeping {delay} seconds and trying again" )
        time.sleep( delay )             # just sleep for a while and then try again
        res     = client.messages.create( **cargs )
    return res.content[ 0 ].text


def complete_none():
    """
    dummy completion for degugging purposes

    return:         [list] with completions [str]
    """
    likerts = [ random.choice( range( 1, 6 ) ) for i in range( cnfg.n_returns ) ]
    returns = [ f"my answer: L{l}" for l in likerts ]
    return returns


def complete_openai( prompt ):
    """
    Feed a prompt to an OpenAI model and get the list of completions returned.
    This function works for both completion-mode models and chat-mode models.

    params:
        prompt      [str] or [list] the prompt for completion-mode models,
                    or the messages for chat-mode models

    return:         [list] with completions [str]
    """
    global client
    if cnfg.DEBUG:  return [ "test_only" ]

    if client is None:              # check if openai has already a client, otherwise set it
        client  = set_openai()
    user    = os.getlogin() + '@' + platform.node()

    if cnfg.mode == "cmpl":
        assert isinstance( prompt, str ), "ERROR: for completion-mode models, the prompt should be a string"
        res     = client.completions.create(
            model                   = cnfg.model,
            prompt                  = prompt,
            max_tokens              = cnfg.max_tokens,
            n                       = cnfg.n_returns,
            top_p                   = cnfg.top_p,
            temperature             = cnfg.temperature,
            stop                    = None,
            user                    = user
        )
        return [ t.text for t in res.choices ]

    if cnfg.mode == "chat":
        assert isinstance( prompt, list ), "ERROR: for chat-mode models, the prompt should be a list"
        # NOTE: for gpt-4o stop=None raises Error code: 400! do not use it
        res     = client.chat.completions.create(
            model                   = cnfg.model,
            messages                = prompt,
            max_tokens              = cnfg.max_tokens,
            n                       = cnfg.n_returns,
            top_p                   = cnfg.top_p,
            temperature             = cnfg.temperature,
            user                    = user
        )
        return [ t.message.content for t in res.choices ]

    return None


def complete_llava( model, processor, prompt, image ):
    """
    Feed a prompt to a Llava model and get the list of completions returned.

        NOTE: currently there is a bug in llava-next when doing inference without an image:
        https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/discussions/36
        As a temporary workaround, if no image is requested, a blank image of the same size is loaded.

    params:
        prompt      [str] or [list] the prompt for completion-mode models,
                    or the messages for chat-mode models
        image       [PIL.JpegImagePlugin.JpegImageFile] or None in case of no image

    return:         [list] with completions [str]
    """
    text        = processor.apply_chat_template( prompt, add_generation_prompt=True )

    # dummy image as workaround for llava-next bug (see comment above)
    if image is None:
        image   = Image.new( mode='L', size=native_res, color="black" )
    else:
        image   = image.resize( native_res )

    inputs      = processor(
            images          = image,
            text            = text,
            return_tensors  = "pt"
    ).to( model.device, torch.float16 )

    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    out         = model.generate(
            **inputs,
            max_new_tokens          = cnfg.max_tokens,
            do_sample               = True,                     # NOTE: the default is greedy!
            num_return_sequences    = cnfg.n_returns,
            top_p                   = cnfg.top_p,
            temperature             = cnfg.temperature,
    )
    res         = processor.batch_decode( out, skip_special_tokens=True )

    # NOTE that the prompt is included in the completion, there is no parameter like return_full_text in pipeline
    # that can avoid this issue in model.generate. Therefore, here are workarounds that are model dependent.
    # CHECK when new models are added
    end_input   = "[/INST]"
    completions = [ r.split( end_input )[ -1 ].strip() for r in res ]

    return completions


def complete_chameleon( model, processor, prompt, image ):
    """
    Feed a prompt to a Chameleon model and get the list of completions returned.

    params:
        prompt      [str] or [list] the prompt for completion-mode models,
                    or the messages for chat-mode models
        image       [PIL.JpegImagePlugin.JpegImageFile] or None in case of no image
        model       [transformers.models...] client model
        processor   transformers.models...] client input processor

    return:         [list] with completions [str]
    """
    if image is None:
        text        = prompt
    else:
        text        = prompt + "<image>"
        image       = image.resize( native_res )

    inputs      = processor(
            images          = image,
            text            = text,
            return_tensors  = "pt"
    ).to( model.device, torch.float16 )

    out         = model.generate(
            **inputs,
            max_new_tokens          = cnfg.max_tokens,
            do_sample               = True,                     # NOTE: the default is greedy!
            num_return_sequences    = cnfg.n_returns,
            top_p                   = cnfg.top_p,
            temperature             = cnfg.temperature,
    )
    res         = processor.batch_decode( out, skip_special_tokens=True )

    # NOTE that the prompt is included in the completion, there is no parameter like return_full_text in pipeline
    # that can avoid this issue in model.generate. Therefore, here are workarounds that are model dependent.
    # CHECK when new models are added
    completions = [ r.replace( prompt, "" ) for r in res ]

    return completions


def complete_qwen( model, processor, prompt, image ):
    """
    Feed a prompt to a Qwen model and get the list of completions returned.

    params:
        prompt      [str] or [list] the prompt for completion-mode models,
                    or the messages for chat-mode models

    return:         [list] with completions [str]
    """
    if image is not None:
        image   = image.resize( native_res )
    text        = processor.apply_chat_template(
                    prompt,
                    tokenize                = False,
                    add_generation_prompt   = True
    )
    inputs      = processor(
            text            = text,
            images          = image,
            padding         = True,
            return_tensors  = "pt"
    ).to( model.device, torch.float16 )

    out         = model.generate(
            **inputs,
            max_new_tokens          = cnfg.max_tokens,
            do_sample               = True,                     # NOTE: the default is greedy!
            num_return_sequences    = 1,                        # NOTE: more than 1 produces garbage!
            top_p                   = cnfg.top_p,
            temperature             = cnfg.temperature,
    )
    res         = processor.batch_decode( out, skip_special_tokens=True )

    # NOTE that the prompt is included in the completion, there is no parameter like return_full_text in pipeline
    # that can avoid this issue in model.generate. Therefore, here are workarounds that are model dependent.
    # CHECK when new models are added
    end_input   = "\nassistant\n"
    completions = [ r.split( end_input )[ -1 ].strip() for r in res ]

    return completions


def complete_gemma( model, processor, prompt ):
    """
    Feed a prompt to a gemma model and get the list of completions returned.

    params:
        prompt      [str] or [list] the prompt for completion-mode models,
                    or the messages for chat-mode models

    return:         [list] with completions [str]
    """
    global client
    if cnfg.model == "google/gemma-3-4b-it":
        dtype   = torch.bfloat16
    else:
        dtype   = torch.float16
    inputs      = processor.apply_chat_template(
                    prompt,
                    tokenize                = True,
                    add_generation_prompt   = True,
                    return_dict             = True,
# this patch was suggested in https://github.com/google-deepmind/gemma/issues/169
# but has no effect
#                   do_pan_and_scan         = True,
# same for the following one, from further posts in https://github.com/google-deepmind/gemma/issues/169
# have no effect
#                   pad_to_multiple_of      = 8,
# this seems to have effect on gemma-3-4b, and just a mitigation of the error on gemma-3-12b
                    padding                 = "longest",
                    return_tensors          = "pt"
    ).to( model.device, dtype=dtype )

    input_len   = inputs[ "input_ids" ].shape[ -1 ]

    with torch.inference_mode():
        # there have been "RuntimeError: p.attn_bias_ptr is not correctly aligned" errors,
        # the problem is not solved, so in case just report the error in completions
        # maybe a solution could be to delete the model and re-initialize in case of exception
        try:
            out         = model.generate(
                    **inputs,
                    max_new_tokens          = cnfg.max_tokens,
                    do_sample               = True,
                    num_return_sequences    = cnfg.n_returns,
                    top_p                   = cnfg.top_p,
                    temperature             = cnfg.temperature,
            )
        except Exception as e:
            if cnfg.VERBOSE:
                print( f"catched error {e}, trying to reinstall the model" )
            out         = None

# as emergency action for the RuntimeError try to delete the model and reinstall it again
    if out is None:
        del model
        del processor
        gc.collect()
        torch.cuda.empty_cache()
        client      = set_hf_gemma()
        model       = client[ "model" ]
        processor   = client[ "processor" ]
        with torch.inference_mode():
            try:
                out         = model.generate(
                        **inputs,
                        max_new_tokens          = cnfg.max_tokens,
                        do_sample               = True,
                        num_return_sequences    = cnfg.n_returns,
                        top_p                   = cnfg.top_p,
                        temperature             = cnfg.temperature,
                )
            except Exception as e:
                if cnfg.VERBOSE:
                    print( f"catched error {e} for the second time" )
                completions = [ "error" for i in range( cnfg.n_returns ) ]
                return completions

    out = [ o[ input_len: ] for o in out ]
    completions = processor.batch_decode( out, skip_special_tokens=True )

    return completions


def complete_hf( prompt, image ):
    """
    Feed a prompt to a HuggingFace model and get the list of completions returned.

    params:
        prompt      [str] or [list] the prompt for completion-mode models,
                    or the messages for chat-mode models
        image       [PIL.JpegImagePlugin.JpegImageFile] or None in case of no image

    return:         [list] with completions [str]
    """
    global client
    if cnfg.DEBUG:  return [ "test_only" ]

    if client is None:              # check if hf has already a client, otherwise set it
        client  = set_hf()

    model       = client[ "model" ]
    processor   = client[ "processor" ]

    if "llava-v1.6" in cnfg.model:
        return complete_llava( model, processor, prompt, image )
    if "gemma" in cnfg.model:
        return complete_gemma( model, processor, prompt )
    if "chameleon" in cnfg.model:
        return complete_chameleon( model, processor, prompt, image )
    if "Qwen" in cnfg.model:
        return complete_qwen( model, processor, prompt, image )
#       return complete_qwen_base64( model, processor, prompt )

    print( f"WARNING: model '{cnfg.model}' not currently supported.")
    return None


def do_complete( prompt, image=None ):
    """
    Feed a prompt to any model and get the list of completions returned.

    params:
        prompt      [str] or [list] the prompt for completion models,
                    or the messages for chat completion models
        image       [PIL.JpegImagePlugin.JpegImageFile] or None, for OpenAI and Qwen
                    the image is embedded in the propmt

    return:         [list] with completions [str]
    """
    match cnfg.interface:

        case 'openai':
            return complete_openai( prompt )

        case 'none':
            return complete_none()

        case 'anthro':
            # anthropic allows only one return per completion
            completions     = []
            for i in range( cnfg.n_returns ):
                completions.append( complete_anthro( prompt ) )
            return completions

        case 'hf':
            if "Qwen" in cnfg.model:
                n_max   = qwen2_vl_n_max
            else:
                n_max   = llava_next_n_max
            if cnfg.n_returns <= n_max:
                return complete_hf( prompt, image=image )
            # to avoid problems of GPU memory, do separate reps
            reps            = cnfg.n_returns // n_max
            reminder        = cnfg.n_returns % n_max
            saved           = cnfg.n_returns                    # make a copy of the original number of results
            completions     = []
            cnfg.n_returns  = n_max
            for i in range( reps ):
                completions += ( complete_hf( prompt, image=image ) )
            if reminder:
                cnfg.n_returns  = reminder
                completions += ( complete_hf( prompt, image=image ) )
            cnfg.n_returns  = saved                             # restore the original number of results
            return completions

        case _:
            print( f"WARNING: model interface '{cnfg.interface}' not supported" )
            return None
