"""
#####################################################################################################################

    Main file to execute the program

    For help
        $ python main_exec.py -h

#####################################################################################################################
"""

import  os
import  sys
import  copy
import  shutil
import  itertools
import  time
import  numpy           as np

import  load_cnfg                               # this module sets program parameters
import  prompt          as prmpt                # this module composes the prompts
import  complete        as cmplt                # this module performs LLM completions
import  conversation    as conv                 # this module handles conversations with the LLM
import  save_res                                # this module saves results

# this module lists the available LLMs
from    models          import models, models_endpoint, models_interface

# execution directives
DO_NOTHING              = False                 # for interactive usage

frmt_response           = "%y-%m-%d_%H-%M-%S"   # datetime format for filenames
dir_res                 = '../res'              # folder of results
dir_json                = '../data'             # folder of json data
back_file               = "../data/.back.pkl"   # file with temporary backup

cnfg                    = None                  # object containing the execution configuration (see load_cnfg.py)

# globals pointing to current execution folders and files
# NOTE they will be validated in init_dirs()
exec_dir                = None
exec_src                = None
exec_data               = None
exec_log                = None
exec_pkl                = None
exec_csv                = None
base_exec_src           = 'src'
base_exec_data          = 'data'
base_exec_log           = 'log.txt'
base_exec_pkl           = 'res.pkl'
base_exec_csv           = 'res.csv'


# ===================================================================================================================
#
#   Utilities to set up execution
#   - init_dirs
#   - init_cnfg
#   - archive
#
# ===================================================================================================================

def init_dirs():
    """
    Set paths and create directories where to save the current execution
    """
    global exec_dir, exec_src, exec_data        # dirs
    global exec_log, exec_pkl, exec_csv         # files

    now_time        = time.strftime( frmt_response )        # string used for composing file names of results
    exec_dir        = os.path.join( dir_res, now_time )
    while os.path.isdir( exec_dir ):
        if cnfg.VERBOSE:
            print( f"WARNING: a folder with the timestamp {exec_dir} already exists." )
            print( "Creating a folder with a timestamp a second ahead.\n" )
        sec         = int( exec_dir[ -2: ] )
        sec         += 1
        exec_dir    = f"{exec_dir[ :-2 ]}{sec:02d}"

    exec_src        = os.path.join( exec_dir, base_exec_src )
    exec_data       = os.path.join( exec_dir, base_exec_data )

    os.makedirs( exec_dir )
    os.makedirs( exec_src )
    os.makedirs( exec_data )
    exec_log        = os.path.join( exec_dir, base_exec_log )
    exec_pkl        = os.path.join( exec_dir, base_exec_pkl )
    exec_csv        = os.path.join( exec_dir, base_exec_csv )


def init_cnfg():
    """
    Set execution parameters received from command line and python configuration file
    NOTE Execute this function before init_dirs()
    """
    global cnfg

    cnfg            = load_cnfg.Config()                    # instantiate the configuration object

    # load parameters from command line
    line_kwargs     = load_cnfg.read_args()                 # read the arguments in the command line
    cnfg.load_from_line( line_kwargs )                      # and parse their value into the configuration obj

    if cnfg.MODEL is not None and cnfg.MODEL < 0:
        print( "ID    model                                 interface" )
        for i, m in enumerate( models ):
            f   = models_interface[ m ]
            if len( m ) > 40:
                m   = m[ : 26 ] + "<...>" + m[ -9 : ]
            print( f"{i:>2d}   {m:<43}{f:<8}" )
        sys.exit()

    # load parameters from configuration file
    if cnfg.CONFIG is not None:
        exec( "import " + cnfg.CONFIG )                     # exec the import statement
        file_kwargs     = eval( cnfg.CONFIG + ".kwargs" )   # assign the content to a variable
        cnfg.load_from_file( file_kwargs )                  # read the configuration file

    else:                                                   # default configuration
        cnfg.model_id           = 0                         # use the defaul model
        cnfg.n_returns          = 1                         # just one response
        cnfg.max_tokens         = 50                        # afew tokens
        cnfg.repetition_penalty = 1.1                       # value found with little experimentation
        cnfg.top_p              = 1                         # set a reasonable default
        cnfg.temperature        = 0.3                       # set a reasonable default
        cnfg.dialogs_pre        = ""                        # set a reasonable default
        cnfg.dialogs_post       = ""                        # set a reasonable default
        cnfg.news_ids           = []                        # set a reasonable default

    if not hasattr( cnfg, 'experiment' ):
        cnfg.experiment         = None                      # whether experiment uses images or not, or other

    # overwrite command line arguments
    if cnfg.MAXTOKENS is not None:      cnfg.max_tokens = cnfg.MAXTOKENS
    if cnfg.MODEL is not None:          cnfg.model_id   = cnfg.MODEL
    if cnfg.NRETURNS is not None:       cnfg.n_returns  = cnfg.NRETURNS

    if cnfg.experiment == "check_news":                     # when checking news just one completion is required
        cnfg.n_returns  = 1
        if not len( cnfg.dialogs_pre ):                     # set default dialog, if not already set in configuration
            cnfg.dialogs_pre    = "check_text"

    # if a model is used, from its index derive the complete model name and usage mode
    if hasattr( cnfg, 'model_id' ):
        assert cnfg.model_id < len( models ), f"error: model # {cnfg.model_id} not available"
        cnfg.model          = models[ cnfg.model_id ]
        cnfg.mode           = models_endpoint[ cnfg.model ]
        cnfg.interface      = models_interface[ cnfg.model ]

    # if a model is used, check if the name contains a directive, that follows the model name with a "+"
    if hasattr( cnfg, 'model' ):
        if '+' in cnfg.model:
            name, directive     = cnfg.model.split( '+' )
            cnfg.model          = name                      # restore the proper name of the model
            cnfg.directive      = directive                 # make the directive visible in log.txt
            # now manage the directive
            if directive == "blank_img":                    # include a blank image for text only news
                prmpt.insert_blank  = True                  # inform the prompt module

    # this variabile in the configuration triggers multi-execution over multiple options in dialogs_pre
    if hasattr( cnfg, 'multi_dialogs_pre' ):
        assert isinstance( cnfg.multi_dialogs_pre, list ), "error in configuration: multi_dialogs_pre is not a list"
        assert not hasattr( cnfg, 'multi_demography' ), "error: cannot do multiple runs for dialogs_pre and demography"
        cnfg.multi_exec = "dialogs_pre"
    else:
        cnfg.multi_exec = "single"

    # this variabile in the configuration triggers multi-execution over multiple options in demographics
    if hasattr( cnfg, 'multi_demography' ):
        assert isinstance( cnfg.multi_demography, dict ), "error in configuration: multi_demography is not a dict"
        cnfg.multi_exec = "demography"

    # export information from config
    if hasattr( cnfg, 'f_dialog' ):     prmpt.f_dialog  = cnfg.f_dialog
    if hasattr( cnfg, 'f_demo' ):       prmpt.f_demo    = cnfg.f_demo
    if hasattr( cnfg, 'detail' ):       prmpt.detail    = cnfg.detail
    prmpt.f_news        = cnfg.f_news

    if not len( cnfg.news_ids ):
        if cnfg.news_amount is None:
            # use all news in file if not specified otherwise
            cnfg.news_ids   = prmpt.list_news()
        else:
            # use the first N news in file
            cnfg.news_ids   = prmpt.list_news( cnfg.news_amount )

    # verify backward compatibility of dialog titles
    if hasattr( cnfg, 'dialogs_pre' ):
        if isinstance( cnfg.dialogs_pre, list ):
            for i, d in enumerate( cnfg.dialogs_pre ):
                if "profile_" in d:                         # since 2025-03-13
                    cnfg.dialogs_pre[ i ]   = d.replace( "profile_", "p_" )
        else:
            if "profile_" in cnfg.dialogs_pre:
                cnfg.dialogs_pre   = cnfg.dialogs_pre.replace( "profile_", "p_" )

    # set automatically likert_scale if a preliminary dialog requires it
    if isinstance( cnfg.dialogs_post, list ):
        for d in cnfg.dialogs_post:
            if "likert" in d:
                cnfg.likert_scale   = True
                if not hasattr( cnfg, 'agreement' ):
                    cnfg.agreement          = True          # set agreement measure by default for Likert
    elif isinstance( cnfg.dialogs_post, str ):
        if "likert" in cnfg.dialogs_post:
            cnfg.likert_scale   = True
            if not hasattr( cnfg, 'agreement' ):
                cnfg.agreement          = True              # set agreement measure by default for Likert

    if not hasattr( cnfg, 'agreement' ):
        cnfg.agreement          = False                     # set no agreement measure, if not set otherwise

    if cnfg.RECOVER:                                        # manage recovery backup
        assert os.path.isfile( back_file ), \
            "can't recover execution: backup file not found"
    else:
        if os.path.exists( back_file ):
            os.remove( back_file )
    cnfg.back_file      = back_file

    # pass global parameters to other modules
    cmplt.cnfg          = cnfg
    conv.cnfg           = cnfg
    save_res.cnfg       = cnfg


def archive():
    """
    Save a copy of current python source and json data files in the execution folder
    """
    jfiles  = (
                "demo_small.json",
                "dialogs_asst.json",
                "dialogs_user.json",
                "news_200.json",
                "trait.json",
    )

    pfiles  = [
                "clean_data.py",
                "complete.py",
                "conversation.py",
                "infstat.py",
                "load_cnfg.py",
                "main_exec.py",
                "models.py",
                "plot.py",
                "prompt.py",
                "save_res.py",
                "scan_res.py",
    ]

    if cnfg.CONFIG is not None:
        pfiles.append( cnfg.CONFIG + ".py" )

    for pfile in pfiles:
        try:
            shutil.copy( pfile, exec_src )
        except:
            print( f"NOTE: no file named {pfile} to copy")
    for jfile in jfiles:
        jfile   = os.path.join( dir_json, jfile )
        try:
            shutil.copy( jfile, exec_data )
        except:
            print( f"NOTE: no file named {jfile} to copy")


# ===================================================================================================================
#
#   Main function
#   - do_exec
#   - multi_exec
#
# ===================================================================================================================

def do_exec():
    """
    Execute the program in one of the available modality (with image, without, or both) and save the results.

    return:     True if execution is succesful
    """
    fstream         = open( exec_log, 'w', encoding="utf-8" )   # open the log file

    match cnfg.experiment:
        case "news_noimage":
            pr, compl, res, names           = conv.ask_news(
                    with_img        = False,
                    demographics    = cnfg.demographics,
                    agreement       = cnfg.agreement
                )

        case "news_image":
            pr, compl, res, names           = conv.ask_news(
                    with_img        = True,
                    demographics    = cnfg.demographics,
                    agreement       = cnfg.agreement
                )

        case "both":
            back_noi                = None
            back_img                = None
            if cnfg.RECOVER:
                back_noi, back_img  = conv.load_backup()
            if back_img is None:
                pr_noi, com_noi, res_noi, n_n   = conv.ask_news(
                        with_img        = False,
                        demographics    = cnfg.demographics,
                        agreement       = cnfg.agreement,
                        backup          = ( back_noi, False )
                    )
                back_noi                = pr_noi, com_noi, res_noi, n_n, copy.deepcopy( cnfg.news_ids )
            pr_img, com_img, res_img, n_i   = conv.ask_news(
                    with_img        = True,
                    demographics    = cnfg.demographics,
                    agreement       = cnfg.agreement,
                    backup          = ( back_noi, back_img )
                )
            pr                              = pr_img + pr_noi
            compl                           = com_img + com_noi
            names                           = n_i + n_n
            res                             = { "with_img": res_img, "no_img": res_noi }

        case "check_news":
            pr, compl, res                  = conv.check_news_text()
            save_res.write_check_news( fstream, pr, compl, res, mode=cnfg.mode )
            fstream.close()
            return True

        case _:
            print( f"ERROR: experiment '{cnfg.experiment}' not implemented" )
            return None

    save_res.write_all(
            fstream,
            pr,
            compl,
            res,
            names,
            exec_csv,
            exec_pkl,
            mode        = cnfg.mode,
            likert      = cnfg.likert_scale,
            agreement   = cnfg.agreement
            )
    fstream.close()
    return True


def multi_dialogs_pre():
    """
    Execute the program multiple times with variations in dialogs_pre
    it is necessary to validate in the configuration file a list with name "multi_dialogs_pre"
    that contains an arbitrary number of dialog turns, one of which is not a string with a dialog title
    but another list, and all the titles inside it are used in an execution
    if this list is empty, all the exixting options with title starting with "p_" will be used, see
    list_profiles() in prompt.py
    """

    pre_var     = [ pre for pre in cnfg.multi_dialogs_pre if isinstance( pre, list ) ]
    assert len( pre_var ) == 1, "ERROR: there should be only one multivariation in multi_dialogs_pre"
    pre_var     = pre_var[ 0 ]
    if not len( pre_var ):                                      # the empty list means use all possible options
        pre_var = prmpt.list_profiles()                         # that are read from json file
    n_pre       = len( cnfg.multi_dialogs_pre )                 # number of dialog turns
    idx_var     = cnfg.multi_dialogs_pre.index( pre_var )       # index of the dialg turn with multiple options
    n_pre_var   = len( pre_var )                                # number of options for the varied dialog turn

    # generates on-the-fly cnfg.dialogs_pre with one option at turn
    for i, option in enumerate( pre_var ):
        if cnfg.VERBOSE:
            print( f"\n** run {i+1} of {n_pre_var} multiple executions, using option {option} of dialogs_pre **\n" )
        dialogs_pre             = copy.deepcopy( cnfg.multi_dialogs_pre )
        dialogs_pre[ idx_var ]  = option
        cnfg.dialogs_pre        = dialogs_pre                   # NOTE: no need to export cnfg again to cmplt
        if i:                                                   # skip init_dirs() and archive() the first run
            init_dirs()                                         # create a new folder for results
            archive()                                           # archive the results
        do_exec()                                               # execute the main program


def multi_demography():
    """
    Execute the program multiple times with variations in demographics
    it is necessary to validate in the configuration file a directory with name "multi_demography"
    that contains for each key a list with an arbitrary number of options
    """
    # generates all combinations of the options listed for each demogaphics key
    keys        = cnfg.multi_demography.keys()
    values      = cnfg.multi_demography.values()
    all_values  = itertools.product( *values )                  # cartesian product of all lists in multi_demography
    demo_comb   = [ dict( zip(keys,v) ) for v in all_values ]   # all combinations
    n_comb      = len( demo_comb )

    # generates on-the-fly cnfg.demographics with one option at turn
    for i, demo in enumerate( demo_comb ):
        if cnfg.VERBOSE:
            print( f"\n** run {i+1} of {n_comb} multiple executions with variations in demographics **\n" )
        cnfg.demographics       = demo
        if i:                                                   # skip init_dirs() and archive() the first run
            init_dirs()                                         # create a new folder for results
            archive()                                           # archive the results
        do_exec()                                               # execute the main program


# ===================================================================================================================
#
#   MAIN
#
# ===================================================================================================================

if __name__ == '__main__':
    if DO_NOTHING:
        print( "Program instructed to DO_NOTHING" )

    else:
        init_cnfg()
        init_dirs()
        if cnfg.experiment is not None:
            if cnfg.DEBUG:
                print( "Program running in DEBUG mode, not archiving" )
            else:
                archive()
            match cnfg.multi_exec:
                case "single":
                    do_exec()
                case "dialogs_pre":
                    multi_dialogs_pre()
                case "demography":
                    multi_demography()
                case _:
                    print( "execution mode {cnfg.multi_exec} not implemented" )
