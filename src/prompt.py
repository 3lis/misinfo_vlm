"""
#####################################################################################################################

    Module to compose prompts

#####################################################################################################################
"""


import  os
import  sys
import  string
import  base64
import  json
from    PIL         import Image
from    io          import BytesIO


dir_json                = "../data"                 # directory with all input data
dir_imgs                = "../imgs"                 # directory with news images
f_dialog                = "dialogs.json"            # filename of the preliminary dialogs
f_demo                  = None                      # filename of demographic data, set in main_exec.py
f_news                  = None                      # filename of the news tests, set in main_exec.py
detail                  = "high"                    # parameter for OpenAI image, overwritten by cnfg
native_res              = ( 672, 672 )              # resolution of blank image
insert_blank            = False                     # directive to insert a blank image in case of text only
DEBUG                   = False                     # local debugging


# ===================================================================================================================
#
#   Utilities to read data
#   - list_news
#   - image_pil
#   - image_b64
#   - get_dialog
#   - get_news
#
# ===================================================================================================================

def list_news( n=None ):
    """
    Return the list with all ID of the news found in the JSON dataset

    params:
        n:      [int] number of news to return
    return:     [list] with news IDs
    """
    fname   = os.path.join( dir_json, f_news )
    with open( fname, 'r' ) as f:
        data    = json.load( f )
    ids     = [ d[ 'id' ] for d in data ]

    # select only first n/2 false and n/2 true news
    if n is not None:
        nn      = n // 2
        ct, cf  = ( 0, 0 )
        t_ids   = []
        f_ids   = []

        for i in ids:
            if i.startswith( "t" ) and ( ct < nn ):
                t_ids.append( i )
                ct  += 1
            if i.startswith( "f" ) and ( cf < nn ):
                f_ids.append( i )
                cf  += 1
            if ( ct == nn ) and ( cf == nn ):
                break
        ids     = f_ids + t_ids

    return ids


def list_profiles():
    """
    Return the list with all ID of the profiling dialogs in the JSON dataset

    return:     [list] with profiling dialogs IDs
    """
    fname   = os.path.join( dir_json, f_dialog )
    with open( fname, 'r' ) as f:
        data    = json.load( f )
    ids     = [ d[ 'id' ] for d in data ]

    p_ids   = [ i for i in ids if i.startswith( "p_" ) ]

    return ids


def image_pil( i ):
    """
    Return an image as PIL object, as requested in LlaVa prompts

    params:
        i       [int] id of the news linked to the image

    return:     [PIL.JpegImagePlugin.JpegImageFile]
    """
    fname   = os.path.join( dir_json, f_news )
    with open( fname, 'r' ) as f:
        data    = json.load( f )
    ids     = [ d[ 'id' ] for d in data ]

    try:
        idx     = ids.index( i )
    except Exception as e:
        print( f"ERROR: non existing news with ID={i} in image_pil()" )
        raise e

    img_name    = data[ idx ][ "image" ]
    fname       = os.path.join( dir_imgs, img_name )
    img         = Image.open( fname )
    return img


def image_b64( fname ):
    """
    Return an image as b64encoded string, as requested in OpenAi prompts

    params:
        fname   [str] name of the file, without path

    return:     [bytes] the b64encoded image
    """
    fname       = os.path.join( dir_imgs, fname )
    with open( fname, 'rb' ) as f:
        img_str     = base64.b64encode( f.read() ).decode( "utf-8" )

    return img_str


def blank_b64():
    """
    generate a blank image as b64encoded string, as requested in OpenAi prompts

    return:     [bytes] the b64encoded image
    """
    image       = Image.new( mode='L', size=native_res, color="black" )
    buffer      = BytesIO()
    image.save( buffer, format="JPEG" )
    img_str     = base64.b64encode( buffer.getvalue() ).decode( "utf-8" )

    return img_str


def get_dialog( id_dialog, with_img, demographics=None ):
    """
    Return a single dialog from the JSON dataset, optionally including demographic details.

    params:
        id_dialog   [str] ID of the dialog
        with_img    [bool] the news contains an image
        demographics [dict] demographic details, or None

    return:     [str] the dialog content (optionally including demographics)
                [bool] flag that the chat mode is chat-completion
    """
    # print(f"DEBUG: Calling get_dialog() with i={i}, with_img={with_img}, include_demographics={include_demographics}, demographics={demographics}")

    if not len( id_dialog ):
        return ''

    fname = os.path.join(dir_json, f_dialog)

    with open(fname, 'r') as f:
        data = json.load(f)
    ids = [ d['id'] for d in data ]

    try:
        idx = ids.index( id_dialog )
    except Exception as e:
        print( f"ERROR: non-existing dialog '{i}' in get_dialog()" )
        raise e

    if with_img and "content_img" in data[ idx ]:
        text    = data[ idx ][ "content_img" ]
    else:
        text    = data[ idx ][ "content" ]

    let_compl   = "completion_mode" in data[ idx ].keys()

    # If demographics are included, process them
    if demographics:
        # Load valid demographic attributes
        dfile = os.path.join( dir_json, f_demo )
        with open(dfile, 'r') as f:
            valid_demographics = json.load(f)

        # Validate demographics against available options
        valid_keys = valid_demographics.keys()
        for key, value in demographics.items():
            if key not in valid_keys or value not in valid_demographics[key]:
                raise ValueError(f"Invalid demographic value: {key} = {value}")

        # Retrieve and format content_dems
        try:
            content_dems = next(item['content'] for item in data if item['id'] == 'content_dems')
        except StopIteration:
            raise ValueError(f"ERROR: Missing 'content_dems' entry in {f_demo}")

        # Dynamically format demographics into content_dems
        content_dems_filled = content_dems.format(**demographics)

        # Insert demographic content into the main text
        text = text.replace("{content_dems}", content_dems_filled)
    # otherwise, remove the {content_dems} slot
    else:
        text = text.replace( "{content_dems}", '' )

    return text, let_compl


def get_news( news, source=False, more=False ):
    """
    Return the formatted textual description of a news

    params:
        news    [dict] as extracted from the JSON file
        source  [bool] add info about the source of the news
        more    [bool] add more available info about the news, like number of share/followers

    return:     [str] text content of the news
    """
    c           = news[ "content" ]
    s           = news[ "source" ]
    m           = news[ "more" ]
    text        = "\n"

    if source and len( s ) > 0:
        text    += f"The news comes from {s}"
    if more and len( m ) > 0:
        text    += f" {m}"
    if source or more:
        text    += "\n"
    if "headline" in news:
        text    += f"{news[ 'headline' ]} "
    text    += f"{c}\n"

    return text



# ===================================================================================================================
#
#   Functions composing prompts
#   - prune_prompt
#   - compose_prompt
#   - format_prompt
#
# ===================================================================================================================

def prune_prompt( prompt ):
    """
    Prune the b64-encoded images and other unwanted structures from the prompt result.
    This is used to save the textual part only in the log.

    params:
        prompt  [list] the prompt

    return:     [list] the pruned prompt
    """
    pruned      = []

    for p in prompt:
        if isinstance( p[ "content" ], str ):
            pruned.append( p )
        else:
            role    = p[ "role" ]
            if "text" in p[ "content" ][ 0 ]:
                text    = p[ "content" ][ 0 ][ "text" ]
                # text    = text.lstrip()                     # remove annoying leading whitespace characters
                pruned.append( { "role": role, "content": text } )

    return pruned


def compose_prompt(
        news_id,
        pre="",
        post="",
        with_img=True,
        source=False,
        more=False,
        demographics=None):
    """
    Compose the text of a prompt processing one news

    params:
        news        [str] id of the news
        pre         [str] or [list of str] optional ids of text before the news content
        post        [str] or [list of str] optional ids of text after the news content
        with_img    [bool] the news contains an image
        source      [bool] add info about the source of the news
        more        [bool] add more available info about the news, like number of share/followers
        demographics [dict] demographics data, or None

    return:         [str] the prompt
                    [str] image name or "" if not with_img
                    [bool] flag that the chat mode is chat-completion
    """
    fname   = os.path.join( dir_json, f_news )
    with open( fname, 'r' ) as f:
        data    = json.load( f )
    ids     = [ d[ 'id' ] for d in data ]

    try:
        idx     = ids.index( news_id )
    except Exception as e:
        print( f"ERROR: non existing news with ID {news_id} in compose_prompt()" )
        raise e

    full_text           = ""
    news                = data[ idx ]
    text                = get_news( news, source=source, more=more )
    fimage              = news[ "image" ] if with_img else ''

    if isinstance( pre, list ):
        for p in pre:
            t,let_compl = get_dialog(p, with_img, demographics=demographics)
            full_text   += f"{t} "

    elif isinstance( pre, str ):
        t,let_compl = get_dialog(pre, with_img, demographics=demographics)
        full_text   += f"{t} "

    full_text   += f"\n{text}\n"

    if isinstance( post, list ):
        for p in post:
            t,let_compl = get_dialog(p, with_img, demographics=demographics)
            full_text   += f"{t} "

    elif isinstance( post, str ):
        t,let_compl = get_dialog(post, with_img, demographics=demographics)
        full_text   += f"{t} "

    return full_text, fimage, let_compl


def format_prompt(
        news_id,
        interface,
        mode="chat",
        pre="",
        post="",
        with_img=True,
        source=False,
        more=False,
        demographics=None):
    """
    Format the prompt for the language model.
    For OpenAI interface, the image is passed within the prompt.
    For HF interface, the image is passed separately in complete.py, but not in the case of gemma

    params:
        news        [str] id of the news
        interface   [str] "openai" or "hf" or "qwen" or "gemma"
        mode        [str] "cmpl" or "chat"
        pre         [str] or [list of str] optional ids of text before the news content
        post        [str] or [list of str] optional ids of text after the news content
        with_img    [bool] the news contains an image
        source      [bool] add info about the source of the news
        more        [bool] add more available info about the news, like number of share/followers
        demographics [dic] demographic data, or None

    return:         [list] the prompt
                    [str] image name or "" if not with_img
    """
    full_text, fimage, let_compl  = compose_prompt(
                            news_id,
                            pre         = pre,
                            post        = post,
                            with_img    = with_img,
                            source      = source,
                            more        = more,
                            demographics=demographics,
    )

#   if DEBUG:   full_text = "describe the content of this image"

    match interface:
        case "openai" | "none":
            # OpenAI with image included as string in the prompt
            if with_img or insert_blank:
                if with_img:
                    image               = image_b64( fimage )
                else:
                    image               = blank_b64()
                img_content         = {
                        "type":         "image_url",
                        "image_url" :   {
                            "url":      f"data:image/jpeg;base64,{image}",
                            "detail":   detail
                        }
                    }
                prompt      = [ {
                    "role":     "user",
                    "content":  [
                        { "type": "text", "text": full_text },
                        img_content
                    ]
                } ]

            # OpenAI without image
            else:
                prompt      = [ { "role": "user", "content": full_text } ]

        case "anthro":
            # anthropic with image included as string in the prompt
            if with_img or insert_blank:
                if with_img:
                    image               = image_b64( fimage )
                else:
                    image               = blank_b64()
                img_content         = {
                        "type":         "image",
                        "source" :   {
                            "type":         "base64",
                            "media_type":   "image/jpeg",
                            "data":         image,
                        }
                    }
                prompt      = [ {
                    "role":     "user",
                    "content":  [
                        { "type": "text", "text": full_text },
                        img_content
                    ]
                } ]

            # anthropic without image
            else:
                prompt      = [ { "role": "user", "content": full_text } ]

        # HuggingFace Qwen with or without image (the image is handled in complete.py)
        case "qwen":
            if with_img:
                img_content = { "type": "image" }
                prompt      = [ {
                    "role":     "user",
                    "content":  [
                        { "type": "text", "text": full_text },
                        img_content
                    ]
                } ]

            # Qwen without image
            else:
                prompt      = [ {
                    "role":     "user",
                    "content":  [ { "type": "text", "text": full_text } ]
                } ]

        # HuggingFace gemma with or without image
        case "gemma":
            if with_img:
                # the image is specified with its file name
                fname       = os.path.join( dir_imgs, fimage )
                prompt      = [ {
                    "role":     "user",
                    "content":  [
                        { "type": "image", "image": fname },
                        { "type": "text", "text": full_text }
                    ]
                } ]

            # gemma without image
            else:
                prompt      = [ {
                    "role":     "user",
                    "content":  [ { "type": "text", "text": full_text } ]
                } ]

        # HuggingFace llava-next with or without image (the image is handled in complete.py)
        case "hf":
            # NOTE currently llava-next has a bug that does not allow inference without an image
            # therefore "chat" mode has just one option "with image"
            if mode == "chat":
                img_content = { "type": "image" }
                prompt      = [ {
                    "role":     "user",
                    "content":  [
                        { "type": "text", "text": full_text },
                        img_content
                    ]
                } ]

            # for "cmpl" mode, the image is handled in complete.py
            elif mode == "cmpl":
                prompt      = full_text

            else:
                print( f"ERROR: mode '{mode}' not supported" )
                sys.exit()

        case _:
            print( f"ERROR: model interface '{interface}' not supported" )
            sys.exit()

    # add an assistant role with empty content to trigger completion of the previous sentence
    # NOTE: this trigger DOES NOT WORK with Llava and Calude families of models
    if let_compl and interface not in [ "hf", "anthro" ]:
        prompt.append( { "role": "assistant", "content": [ {"type": "text", "text": ""} ] } )

    return prompt, fimage
