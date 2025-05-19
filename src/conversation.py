"""
#####################################################################################################################

    Module to handle conversation dynamics

#####################################################################################################################
"""

import  sys
import  re
import  copy
import  numpy           as np
import  pickle

import  prompt          as prmpt                # this module composes the prompts
import  complete        as cmplt                # this module performs LLM completions

cnfg                    = None                  # parameter obj assigned by main_exec.py

# ===================================================================================================================
#
#   - save_backup
#   - load_backup
#   - check_reply_bool
#   - check_reply_likert
#   - check_reply
#   - ask_news
#
# ===================================================================================================================

def save_backup( back_noi, back_img ):
    """
    save the current processes stories on temporary backup
    input:
        back_noi    [tuple] no-image backup or None
        back_img    [tuple] with-image backup or None
    """
    with open( cnfg.back_file, 'wb' ) as f:
        pickle.dump( ( back_noi, back_img ), f )


def load_backup():
    """
    load the processes stories saved on temporary backup
    return:
        [tuple] with [list] of processed stories, [list] with scores of processed stories
    """
    with open( cnfg.back_file, 'rb' ) as f:
        s   = pickle.load( f )
    return s


def check_reply_bool( completion ):
    """
    Check the model answers in response to yes/no questions.
    NOTE: revised for 3rd person chat mode, and multiple occurrence of yes/no patterns

    params:
        completion  [list] of completion text

    return:         [dict] with three [np.array] of booleans for yes/no/unk replies
    """
    no_patterns = [
        "<decision>no",
        "would recommend not reposting",
        "would recommend not sharing",
        "would be advisable to not repost",
        "would be advisable to not share",
        " would not repost",
        " would not post",
        " would not share",
        " would not recommend reposting",
        " would not recommend sharing",
        " should not repost",
        " should not share",
        " would not feel comfortable sharing",
        " do not want to repost",
        " do not want to share",
        " would not want to repost",
        " would not want to share",
    ]
    yes_patterns    = [
        "<decision>yes",
        " want to repost",
        " want to share",
        " would recommend doing so",
        " would decide to repost",
        " would decide to share",
        " would want to repost",
        " would want to share",
        " would feel inclined to repost",
        " would feel inclined to share",
        " would likely share",
        " would like to repost",
        " would like to share",
        " might consider sharing it",
        "it would be reasonable to share",
    ]

    values      = 'yes', 'no', 'unk'
    nc          = len( completion )
    res         = dict()
    for v in values:
        res[ v ]    = np.full( nc, False )

    for i, c in enumerate( completion ):
        c       = c.lower()

        # first, count <yes>/<no> [atterns
        y_count = c.count( "<yes>" )
        n_count = c.count( "<no>" )
        # then, add positive/negative patterns
        for p in yes_patterns:
            y_count     += c.count( p )
        for p in no_patterns:
            n_count     += c.count( p )

        # scan all possible cases
        if not y_count:
            if n_count:
                res[ "no" ][ i ]    = True
                continue
            else:
                print( f"WARNING: no clear reply to YES/NO in completion {i}" )
                res[ "unk" ][ i ]   = True
                continue
        if not n_count:
            res[ "yes" ][ i ]       = True
            continue
        if y_count > n_count:
            res[ "yes" ][ i ]       = True
            continue
        if y_count < n_count:
            res[ "no" ][ i ]        = True
            continue
        print( f"WARNING: {y_count} YES and {n_count} NO in completion {i}" )
        res[ "unk" ][ i ]   = True

    return res


def check_reply_likert( completion, agreement=True ):
    """
    Check the model answers in response to questions with Likter scale.

    params:
        completion  [list] of completion text
        agreement   [bool] request measure of degree of agreement between completions

    return:         [np.array] of fraction of replies in columns [0..5], where 0 is for undefined reply,
                               with appended degree of agreement, when requested
    """
    nc          = len( completion )
    res         = np.zeros( 6 )         # columns [0..5], where 0 is for undefined reply
    # Likert scale replies as required by the prompt (1 lowest, 5 highest)
    replies     = [ "L1", "L2", "L3", "L4", "L5" ]
    # the pattern includes are strange cases found so far: brackets, quotes, spaces, dashes, punctuation
    pattern     = r'[\(\[\{\"\']?\s*L\s*[-]?\s*(1|2|3|4|5)\s*[\)\]\}\"\']?'

    """ original version that issues easily "too many Likert-scale values found"
    for i, c in enumerate( completion ):
        value   = 0
        found   = [ v for v in replies if v in c ]
        if len( found ) == 1:   # check if completion contains one of "replies"
            value   = int( found[ 0 ][ -1 ] )
        elif len( found ) == 0: # zero matches or more than one
            print( f"WARNING: no Likert-scale value found in completion #{i}" )
        else:                   # more than one match
            print( f"WARNING: too many Likert-scale values found in completion #{i}" )
        res[ value ] += 1
    """

    # retrieve all possible matches with Likter scores, and if more than one take the last one
    for i, c in enumerate( completion ):
        value   = 0
        matches = re.findall( pattern, c, flags=re.IGNORECASE )
        if len( matches ) == 1: # completion contains one of "replies", easier case
            value   = int( matches[ 0 ][ -1 ] )
        elif len( matches ) == 0: # zero matches or more than one
            print( f"WARNING: no Likert-scale value found in completion #{i}" )
        else:                   # more than one match
            nm      = len( matches )
            print( f"WARNING: {nm} Likert-scale values found in completion #{i}, taking the last one" )
            value   = int( matches[ -1 ][ -1 ] )
        res[ value ] += 1

    # measure the degree of agreement between completions using Pi of Fleiss' kappa
    # see statsmodels/stats/inter_rater.py, https://en.wikipedia.org/wiki/Fleiss%27_kappa
    # NOTE that statsmodels.stats.inter_rater.fleiss_kappa() cannot be used directly,
    # because the degenerate case of one subject only does not work
    kappa   = ( ( res * res ).sum() - nc ) / ( nc * ( nc - 1. ) )

    res     /= nc
    if agreement:
        res = np.append( res, kappa )

    return res


def check_reply( completion, agreement=True ):
    """
    Check the model answers in response to yes/no or likert-scale questions.

    params:
        completion  [list] of completion text
        agreement   [bool] request measure of degree of agreement between completions

    return:         one or three [np.array] for replies
    """

    if cnfg.likert_scale:
        return check_reply_likert( completion, agreement=agreement )
    return check_reply_bool( completion )


def ask_news( with_img=True, demographics=None, agreement=False, backup=(None,None) ):
    """
    Prepare the prompts and obtain the model completions

    params:
        with_img    [bool] whether the prompts include image and text
        demographics[dict] demographic details, or None
        agreement   [bool] include the agreement score
        backup      [tuple] possible previous backed data, with no-image first, and with-image second

    return:
        [tuple] of:
                    prompts     [list] of all prompt conversations
                    completions [list] the list of completions
                    scores      [list] of the yes/not answers
    """
    prompts         = []            # initialize the list of prompts
    completions     = []            # initialize the list of completions
    scores          = dict()        # initialize the yes/not replies
    img_names       = []            # initialize the list of image names
    done_news       = []            # initialize the processed news
    todo_news       = copy.deepcopy( cnfg.news_ids )

    back_noi, back_img  = backup
    if with_img:
        if back_img is not None:
            if cnfg.VERBOSE:
                print( "recovering from aborted executions with images\n" )
            prompts, completions, scores, img_names, done_news  = back_img
            todo_news       = list( set( todo_news ) - set( done_news ) )
    else:
        if back_noi is not None:
            if cnfg.VERBOSE:
                print( "recovering from aborted executions without images\n" )
            prompts, completions, scores, img_names, done_news  = back_noi
            todo_news       = list( set( todo_news ) - set( done_news ) )

    for n in todo_news:
        if cnfg.VERBOSE:
            i_mode      = "img + txt" if with_img else "only text"
            print( f"====> Processing news {n} {i_mode} <====" )

        # note that cnfg.interface is not enough to instruct prompt formation, several models
        # have different prompt formats even if under the same cnfg.interface
        if "Qwen" in cnfg.model:
            interface       = "qwen"
        elif "gemma" in cnfg.model:
            interface       = "gemma"
        else:
            interface       = cnfg.interface

        pr, name        = prmpt.format_prompt(
                            n,
                            interface,
                            mode        = cnfg.mode,
                            pre         = cnfg.dialogs_pre,
                            post        = cnfg.dialogs_post,
                            with_img    = with_img,
                            source      = cnfg.info_source,
                            more        = cnfg.info_more,
                            demographics= demographics,
        )

        # using OpenAI
        if cnfg.interface == "openai":
            completion  = cmplt.do_complete( pr )
            pr          = prmpt.prune_prompt( pr ) # remove the textual version of the image from the prompt
        # using HuggingFace
        else:
            image       = prmpt.image_pil( n ) if with_img else None
            completion  = cmplt.do_complete( pr, image=image )

        res             = check_reply( completion, agreement=agreement )
        scores[ n ]     = res
        prompts.append( pr )
        completions.append( completion )
        img_names.append( name )
        done_news.append( n )

        if with_img:
            back_img    = prompts, completions, scores, img_names, done_news
            save_backup( back_noi, back_img )
        else:
            back_noi    = prompts, completions, scores, img_names, done_news
            save_backup( back_noi, None )

    return prompts, completions, scores, img_names


def check_news_text():
    """
    Prompt the model to check text content of news.
    Prepare the prompts and obtain model completions.

    return:
        [tuple] of:
                    prompts     [list] of all prompt conversations
                    completions [list] the list of completions
                    scores      [list] of the yes/not answers
    """
    prompts         = []        # initialize the list of prompts
    completions     = []        # initialize the list of completions
    answer          = dict()    # initialize the yes/not replies

    for n in cnfg.news_ids:
        if cnfg.VERBOSE:
            print( f"====> Processing news {n} <====" )

        pr, _           = prmpt.compose_prompt( n, pre=cnfg.dialogs_pre, source=cnfg.info_source, more=cnfg.info_more, )   # discard the image name
        prompt          = [ { "role": "user", "content": pr } ]
        completion      = cmplt.do_complete( prompt )
        answer[ n ]     = completion[ 0 ].strip()
        prompts.append( prompt )
        completions.append( completion )

    return prompts, completions, answer
