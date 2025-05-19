"""
#####################################################################################################################

    Program for generating statistics over a range of executions in ../res

    for textual statistics the specifications for which analyses to perform are in head of do_stat()

#####################################################################################################################
"""

import  os
import  sys
import  re
import  json
import  time
import  shutil
import  numpy   as np
import  pandas  as pd
from    statsmodels.formula.api     import ols, mixedlm
from    statsmodels.stats.anova     import anova_lm
from    scipy.stats                 import wilcoxon, norm, shapiro, kstest
from    models                      import models_short_name
import  plot

DO_NOTHING          = False                 # for interactive use
DO_MPLOTS           = False                 # generate multiple plots
DO_SPLOTS           = False                 # generate single plots
DO_LKPLOT           = False                 # generate Likert scale plots
DO_RADAR            = True                  # generate radar plots
DO_STATS            = False                 # generate statistics

# specification of the executions to analyze
# if res_range is empty all executions found in ../res are analyzed
# if res_range has only one entry, it is the first execution to process, followed by all the others
# if res_range has two entries, these are the boundaries of the executions to analyze
# if res_range has one list, than all and onle the entries in the inner list are analyzed
res_range           = [ "25-04-06_11-45-18", "25-04-06_19-41-24" ]  # early 200 news, gpt profiles
res_range           = [ "25-04-07_19-29-58", "25-04-11_21-01-37" ]  # corrected 200 news, profiles gpt, claude, llava
res_range           = [ "25-04-12_18-21-27", "25-04-13_03-44-53" ]  # 200 news, demography gpt
res_range           = [ "25-04-12_05-23-54", "25-04-12_18-06-20" ]  # 200 news, profiles, qwen only
res_range           = [["25-04-08_03-58-28", "25-04-21_17-51-45"]]  # gpt void normal and blank_img
res_range           = [ [
    "25-04-08_00-06-12", "25-04-08_01-05-48", "25-04-08_02-04-13", "25-04-08_03-03-27",
    "25-04-22_09-38-55", "25-04-22_09-38-56", "25-04-22_10-29-19", "25-04-22_11-18-26",
]]  # gpt compare dialogs_post for p_neur, p_mach, p_narc, p_psyc
res_range           = [["25-04-08_03-58-28", "25-04-22_16-32-34"]]  # gpt void normal and ask_share_noexplain_likert5
res_range           = [["25-04-22_18-20-00"]]                       # gpt void boolean (no likert)
res_range           = [ [
    "25-04-08_00-06-12", "25-04-08_01-05-48", "25-04-08_02-04-13", "25-04-08_03-03-27",
    "25-04-22_09-38-55", "25-04-22_09-38-56", "25-04-22_10-29-19", "25-04-22_11-18-26",
    "25-04-08_22-53-01", "25-04-09_02-03-06", "25-04-09_05-16-23", "25-04-09_08-21-55",
    "25-04-23_16-16-55", "25-04-23_19-14-09", "25-04-23_22-09-40", "25-04-24_00-53-46",
]]  # gpt and claude compare dialogs_post for p_neur, p_mach, p_narc, p_psyc
res_range           = [ "25-04-07_19-29-58", "25-04-12_18-06-20" ]  # 200 news, profiles gpt, claude, llava, qwen
res_range           = [ "25-04-07_19-29-58", "25-04-21_12-05-58" ]  # all valid executions gpt, calude, llava, qwen
res_range           = [ "25-04-12_18-21-27", "25-04-21_12-05-58" ]  # demography gpt, calude, llava, qwen

res                 = "../res"                  # results directory
dir_json            = "../data"                 # directory with all input data
dir_stat            = "../stat"                 # output directory

# filename of demographic data
# NOTE that there are different data formats, therefore the file name
# is strictly related with its internal format
f_demo              = "demographics.json"       # demographic data in use up to Mar 31st 2025
f_demo              = "demo_small.json"         # essential demographic data
f_news              = "news_200.json"           # file with all news
log                 = "log.txt"                 # filename of execution logs
f_bstat             = "bstat.txt"               # filename of output basic statistics
f_hstat             = "hstat.txt"               # filename of output higher statistics
f_plot              = "pl"                      # filename prefix of output plots
frmt_statdir        = "%y-%m-%d-%H-%M"          # datetime format for output directory
columns_indepv      = (
    "predia",          # the titles of preliminary dialogs
    "postdia",         # the titles of final part of the dialogs
    "profile",         # the profile induced to the model
    "model",           # the model used
    "blank_img",       # a blank images is associated to the news in text mode
    "race",            # race categories as in the demographics file
    "age",             # age categories as in the demographics file
    "news",            # the news code
    "tag1",            # the news tag
    "tag2",            # the news tag
    "tag3",            # the news tag
    "tag4",            # the news tag
    "tagi",            # the news tag
    "value",           # true/false
)
if f_demo == "demo_small.json":                 # demographic columns are bizarre
    columns_indepv  += ( "sex", "party" )
else:
    columns_indepv  += ( "gender", "politic", "edu" )

columns_bool        = (
    "yes_img",        # fraction of YES answer for text+image
    "not_img",        # fraction of NO answer for text+image
    "unk_img",        # fraction of missing answer for text+image
    "yes_txt",        # fraction of YES answer for text only
    "not_txt",        # fraction of NO answer for text only
    "unk_txt",        # fraction of missing answer for text only
)
columns_likert      = (
    "unk_img",        # fraction of missing answer for text+image
    "lk1_img",        # Likert scale interval 1 for text+image
    "lk2_img",        # Likert scale interval 2 for text+image
    "lk3_img",        # Likert scale interval 3 for text+image
    "lk4_img",        # Likert scale interval 4 for text+image
    "lk5_img",        # Likert scale interval 5 for text+image
    "unk_txt",        # fraction of missing answer for text only
    "lk1_txt",        # Likert scale interval 1 for text only
    "lk2_txt",        # Likert scale interval 2 for text only
    "lk3_txt",        # Likert scale interval 3 for text only
    "lk4_txt",        # Likert scale interval 4 for text only
    "lk5_txt",        # Likert scale interval 5 for text only
)
columns_likert_agr  = (
    "unk_img",        # fraction of missing answer for text+image
    "lk1_img",        # Likert scale interval 1 for text+image
    "lk2_img",        # Likert scale interval 2 for text+image
    "lk3_img",        # Likert scale interval 3 for text+image
    "lk4_img",        # Likert scale interval 4 for text+image
    "lk5_img",        # Likert scale interval 5 for text+image
    "agr_img",        # agreement measure for text+image
    "unk_txt",        # fraction of missing answer for text only
    "lk1_txt",        # Likert scale interval 1 for text only
    "lk2_txt",        # Likert scale interval 2 for text only
    "lk3_txt",        # Likert scale interval 3 for text only
    "lk4_txt",        # Likert scale interval 4 for text only
    "lk5_txt",        # Likert scale interval 5 for text only
    "agr_txt",        # agreement measure for text only
)

shortcuts   = {
    'ask_img':                  'ask_im',
    'ask_share':                'ask_sh',
    'ask_share_likert5':        'ask_lk',
    'ask_share_strict':         'ask_ss',
    'ask_share_strict_likert5': 'ask_sl',
    'ask_share_noexplain':      'ask_ne',
    'ask_share_noexplain_likert5': 'ask_nl',
    'ask_dsc':                  'ask_ds',
    'ask_user_dsc':             'ask_ud',
    'ask_user_dsc_void':        'ask_uv',
    'intro_profile':            'int_pr',
    'p_conspirator':            'p_cnsp',
    'p_moderate':               'p_mdrt',
    'p_rational':               'p_rtnl',
    'p_open':                   'p_open',
    'p_cons':                   'p_cons',
    'p_extr':                   'p_extr',
    'p_agre':                   'p_agre',
    'p_neur':                   'p_neur',
    'p_mach':                   'p_mach',
    'p_narc':                   'p_narc',
    'p_psyc':                   'p_psyc',
    'p_void':                   'p_none',
    'context':                  'context',
    'context_strict':           'ctx_st',
    'reason_3steps':            'rea_3s',
    'reason_base':              'rea_bs',
    'reason_share':             'rea_sh',
    'reason_share_likert5':     'rea_sl',
    'reason_share_xml':         'rea_sx',
    'reason_share_delimit':     'rea_sd',
}

all_tags    = [
    "politics",
    "health",
    "economy",
    "law",
    "environment",
    "foreign",
    "society",
    "technology"
]

# the desired order of the "profile" categorial data
ocean_dark  = [ "open", "cons", "extr", "agre", "neur", "narc", "mach", "psyc", "void" ]

demography          = None                      # dictionary with demographic categories
news_tags           = None                      # dictionary with news tags


def read_demo():
    """
    Read the demographic categories in the json file
    """
    global demography
    dfile               = os.path.join( dir_json, f_demo )
    with open( dfile, 'r' ) as f:
        demography      = json.load( f )


def read_tags():
    """
    Read the tags associated with each news in the json file
    """
    global news_tags
    def_tag             = "unknown"
    news_tags           = dict()
    dfile               = os.path.join( dir_json, f_news )
    with open( dfile, 'r' ) as f:
        news            = json.load( f )
    for n in news:
        t_dict      = {
            "tag1" : def_tag,
            "tag2" : def_tag,
            "tag3" : def_tag,
            "tag4" : def_tag,
            "tagi" : n[ "tags_img" ][ 0 ]
        }
        tgs             = n[ "tags" ]
        for i, t in enumerate( tgs ):
            assert t in all_tags, f"non existing tag {t}"
            k           = f"tag{i+1}"
            t_dict[ k ] = t
        news_tags[ n[ "id" ] ]  = t_dict


def count_tags():
    """
    count how many news per tag
    """
    global news_tags
    if news_tags is None:
        print( "WARNING: you should execute read_tags() first" )
        return None

    counts      = dict()
    all_count   = { 't': 0, 'f': 0, 'a': 0 }
    for tag in all_tags:
        counts[ tag ]   = { 't': 0, 'f': 0, 'a': 0 }
        for t_id, a_news in news_tags.items():
            value   = t_id[ 0 ]     # should be 't' or 'f'
            for i in range( 1, 5 ):
                k           = f"tag{i}"
                if tag == a_news[ k ]:
                    counts[ tag ][ value ]   += 1
                    counts[ tag ][ 'a' ]   += 1
                    continue

    for t_id in news_tags.keys():
        value   = t_id[ 0 ]     # should be 't' or 'f'
        all_count[ value ]  += 1
        all_count[ 'a' ]    += 1

    print( "tag             true  false all" )
    print( "-------------------------------" )
    for tag in all_tags:
        t_count = counts[ tag ][ 't' ]
        f_count = counts[ tag ][ 'f' ]
        a_count = counts[ tag ][ 'a' ]
        print( f"{tag:<16}{t_count:3d}   {f_count:3d}   {a_count:3d}" )
    t_count = all_count[ 't' ]
    f_count = all_count[ 'f' ]
    a_count = all_count[ 'a' ]

    print( "-------------------------------" )
    print( f"total          {t_count:4d}  {f_count:4d}  {a_count:4d}" )
    print( "-------------------------------" )

    return counts


def get_demo( lines ):
    """
    Retrieve demographics info used in the statistics
    """
    gender          = "unspec"
    race            = "unspec"
    edu             = "unspec"
    age             = "unspec"
    politic         = "unspec"

    for l in lines:
        if "political_affiliation" in l:
            politic     = l.lower().split()[ -1 ]
        if "gender" in l:
            if "male" in l:
                gender  = 'M'
            if "female" in l:
                gender  = 'F'
        if "race" in l:
            for i, d in enumerate( demography[ "race" ] ):
                if d in l:
                    race    = f"type{i}"
        if "education" in l:
            for i, d in enumerate( demography[ "education" ] ):
                if d in l:
                    edu     = f"level{i}"
        if "age" in l:
            for i, d in enumerate( demography[ "age" ] ):
                if d in l:
                    age     = f"bin{i}"

    return gender, race, edu, age, politic


def get_demo_small( lines ):
    """
    Retrieve essential demographics info used in the statistics
    """
    gender          = "unspec"
    race            = "unspec"
    edu             = "unspec"
    age             = "unspec"
    politic         = "unspec"

    for l in lines:
        if "party" in l:
            politic     = l.lower().split()[ -1 ]
        if "sex" in l:
            if "male" in l:
                gender  = 'M'
            if "female" in l:
                gender  = 'F'
        if "race" in l:
            race        = l.lower().split()[ -1 ]
        if "age" in l:
            age         = l.lower().split()[ -1 ]

    return gender, race, edu, age, politic


def get_predialog( line ):
    """
    Retrieve the pre-dialog settings
    """
    predia      = "unspec"
    profile     = "unspec"
    dialogs     = re.sub( r'[\W]+', ' ', line ).split()[ 1 : ]
    if len( dialogs ) < 2:
        print( f"not enough elements in {dialogs}" )
        return predia, profile
    if "profile_" in dialogs[ 1 ]:
        profile = dialogs[ 1 ].replace( "profile_", "" )
    if "p_" in dialogs[ 1 ]:
        profile = dialogs[ 1 ].replace( "p_", "" )
    if len( dialogs ) > 2:
        if dialogs[ 2 ] in shortcuts.keys():
            predia  = shortcuts[ dialogs[ 2 ] ]

    return predia, profile


def get_postdialog( line ):
    """
    Retrieve the post-dialog settings
    """
    postdia     = "unspec"
    dialogs     = re.sub( r'[\W]+', ' ', line ).split()[ 1 : ]
    if not len( dialogs ):
        print( f"not enough elements in {line}" )
        return postdia
    if dialogs[ 0 ] in shortcuts.keys():
        postdia = shortcuts[ dialogs[ 0 ] ]
    if len( dialogs ) > 1:
        if dialogs[ -1 ] in shortcuts.keys():
            postdia  += shortcuts[ dialogs[ -1 ] ][ -3 : ]

    return postdia


def get_info( lines ):
    """
    Retrieve all info used in the statistics

    the initial part of the log file shuold contain the general information,
    then from the line with "News" on, there are the values
    note that this function is strictly dependent on the format of the log file
    in the part with the values, the function detects if values are in boolean of Likert scale
    again, this automatic detection works only if the number of elements in the values rows
    are strict: 7 for bollean output, 13 for Likert output
    """
    predia      = "unspec"
    postdia     = "unspec"
    profile     = "unspec"
    model       = "unspec"
    gender      = "unspec"
    race        = "unspec"
    edu         = "unspec"
    age         = "unspec"
    politic     = "unspec"
    blank_img   = False
    news        = []            # the news identifiers
    value       = []            # whether the news is classified as true or false
    tag1        = []            # news tags
    tag2        = []            # news tags
    tag3        = []            # news tags
    tag4        = []            # news tags
    tagi        = []            # news tags
    dep_vars    = []            # where dependent variables will be stored

    # scan for general information
    for i, l in enumerate( lines ):
        items   = l.split()
        if not len( items ):
            continue
        item1   = items[ 0 ]
        if "News" == item1:
            break
        if "model" == item1:
            m_fullname      = items[ -1 ]
            model           = models_short_name[ m_fullname ]
        if "directive" == item1:
            if "blank_img" == items[ -1 ]:
                blank_img   = True
        if "dialogs_pre" == item1:
            predia, profile = get_predialog( l )
        if "dialogs_post" == item1:
            postdia         = get_postdialog( l )
        if "demographics" in item1 and not "None" in l:     # care that item1 is "demographics:"!
            demo_lines      = [ l ]
            n   = 1
            l   = lines[ i+n ]
            while l[ : 2 ] == '  ':     # assume that details of demography are indented
                demo_lines.append( l )
                n   += 1
                l   = lines[ i+n ]
                if n > 6:               # stop if too many indented lines are found (6 assumed)
                    print( "invalid demographic format" )
                    return None
                if f_demo == "demo_small.json":
                    gender, race, edu, age, politic = get_demo_small( demo_lines )
                else:
                    gender, race, edu, age, politic = get_demo( demo_lines )

    n       = len( lines )
    i       += 1
    first   = True
    # scan for values of dependent variables
    while True:
        l       = lines[ i ]
        v       = l.split()
        n_var   = len( v )
        if not n_var:
            continue
        if "f_mn" == v[ 0 ]:
            break
        if "t_mn" == v[ 0 ]:
            break
        if "mean" == v[ 0 ]:
            break
        if first:               # self-detect boolean/Likert/agreement in the first row of values
            first_n_var = n_var
            if n_var == 7:
                likert  = False
            elif n_var >= 13:
                likert  = True
                agree   = n_var == 15
            else:
                print( f"invalid lenght of data: {v}" )
                return None
            dep_vars    = [ [] for d in range( n_var - 1 ) ]
            first       = False
        else:                   # chek for consistency in the remaining rows
            if n_var != first_n_var:
                print( f"invalid lenght of data: {v}" )
                return None

        news_id     = v[ 0 ]
        news.append( news_id[ 1 : ] )
        if news_id[ 0 ] == 'f':
            value.append( 'false' )
        else:
            value.append( 'true' )
        tags    = news_tags[ news_id ]
        tag1.append( tags[ 'tag1' ] )
        tag2.append( tags[ 'tag2' ] )
        tag3.append( tags[ 'tag3' ] )
        tag4.append( tags[ 'tag4' ] )
        tagi.append( tags[ 'tagi' ] )

        for d in range( n_var - 1 ):
            dep_vars[ d ].append( float( v[ d + 1 ] ) )

        i       += 1
        if i == n:
            print( "missing end of news results" )
            return None

    # organize all data as numpy array inside a dictionary
    news        = np.array( news )
    value       = np.array( value )
    tag1        = np.array( tag1 )
    tag2        = np.array( tag2 )
    tag3        = np.array( tag3 )
    tag4        = np.array( tag4 )
    tagi        = np.array( tagi )
    predia      = np.full( news.shape, predia )
    postdia     = np.full( news.shape, postdia )
    profile     = np.full( news.shape, profile )
    model       = np.full( news.shape, model )
    blank_img   = np.full( news.shape, blank_img )
    gender      = np.full( news.shape, gender )
    race        = np.full( news.shape, race )
    edu         = np.full( news.shape, edu )
    age         = np.full( news.shape, age )
    politic     = np.full( news.shape, politic )

    # insert the independent variables
    data            = {
        "predia":          predia,
        "postdia":         postdia,
        "profile":         profile,
        "model":           model,
        "blank_img":       blank_img,
        "race":            race,
        "age":             age,
        "news":            news,
        "value":           value,
        "tag1":            tag1,
        "tag2":            tag2,
        "tag3":            tag3,
        "tag4":            tag4,
        "tagi":            tagi,
    }
    # adjust column names depending on the type of demographic data
    if f_demo == "demo_small.json":
        data[ "sex" ]   = gender
        data[ "party" ] = politic
    else:
        data["gender"]  = gender
        data["politic"] = politic
        data["edu"]     = edu
    if likert:
        if agree:
            col_depvars = columns_likert_agr
        else:
            col_depvars = columns_likert
    else:
        col_depvars = columns_bool

    # insert the dependent variables
    for i, c in enumerate( col_depvars ):
        data[ c ]   = np.array( dep_vars[ i ] )

    return data


def select_data():
    """
    build the list of results to collect for statistics

    return:             [list] with directories in ../res
    """

    list_res    = sorted( os.listdir( res ) )
    if not len( res_range ):
        return list_res

    if isinstance( res_range[ 0 ], list ):
        return res_range[ 0 ]

    if len( res_range ) == 1:
        first   = res_range[ 0 ]
        assert first in list_res, f"first specified result {first} not found"
        i_first     = list_res.index( first )
        return list_res[ i_first : ]

    if len( res_range ) == 2:
        first   = res_range[ 0 ]
        last    = res_range[ -1 ]
        assert first in list_res, f"first specified result {first} not found"
        assert last in list_res, f"last specified result {last} not found"
        i_first     = list_res.index( first )
        i_last      = list_res.index( last )
        return list_res[ i_first : i_last+1 ]

    print( "if you want to specify single results to be collected, include them in a list inside res_range\n" )
    return []


def collect_data():
    """
    Scan the results, collecting all data
    the function detects automatically  if values are in boolean of Likert scale
    by checking the name "lk1_img" in the data kyes, therefore this name should be strictly
    present in case of Likert scale
    same for the agreement measue field, should be strictly named "agr_img: in the data kyes

    return:             [pandas.core.frame.DataFrame] the data in pandas DataFrame
    """

    list_res    = select_data()
    n_res       = len( list_res )
    print( f"scanning for {n_res} execution results\n" )
    arrays  = dict()

    first   = True
    for f in list_res:                          # scan all selected results
        fname   = os.path.join( res, f, log )
        if not os.path.isfile( fname ):
            print( f"{f}  is not a file" )
            continue
        with open( fname, 'r' ) as fd:
            lines   = fd.readlines()
        if not len( lines ):
            print( f"{f}  has no lines" )
            continue
        data        = get_info( lines )         # get data for one execution
        if data is None:
            print( f"{f}  no info found" )
            continue
        if first:                               # check for Likert and agreement, and establish the appropriate columns
            if "lk1_img" in data.keys():
                if "agr_img" in data.keys():
                    columns = columns_indepv + columns_likert_agr
                else:
                    columns = columns_indepv + columns_likert
            else:
                columns = columns_indepv + columns_bool
            for c in columns:
                arrays[ c ] = []
            first   = False
        for c in columns:                       # accumulate data
            arrays[ c ].append( data[ c ] )
        n_rec       = len( data[ columns[ 0 ] ] )
        print( f"{f}  done with {n_rec} records" )

    for c in columns:
        v           = arrays[ c ]
        arrays[ c ] = np.concatenate( v )

    df          = pd.DataFrame( arrays )

    # establish the desired order for "profile" categorial data
    df.profile  = pd.Categorical( df.profile, categories=ocean_dark, ordered=True )
    return df


def likert_to_bool( df, half_neutral=True ):
    """
    Convert a dataframe with Likert values into boolean scores
    Use the standard column naming for boolean results

    params:
        df              [pandas.core.frame.DataFrame] the data in pandas DataFrame
        half_neutral    [bool] sum half of Lk3 responses as YES

    return:             df modified in place
    """
    old_columns     = [
            "lk1_img",
            "lk2_img",
            "lk3_img",
            "lk4_img",
            "lk5_img",
            "lk1_txt",
            "lk2_txt",
            "lk3_txt",
            "lk4_txt",
            "lk5_txt"
    ]
    # fraction of YES answer for text+image and text only
    if half_neutral:
        df[ columns_bool[ 0 ] ] = df[ 'lk4_img' ] + df[ 'lk5_img' ] + 0.5 * df[ 'lk3_img' ]
        df[ columns_bool[ 3 ] ] = df[ 'lk4_txt' ] + df[ 'lk5_txt' ] + 0.5 * df[ 'lk3_txt' ]
        df[ columns_bool[ 2 ] ] = df[ 'unk_img' ]
        df[ columns_bool[ 5 ] ] = df[ 'unk_txt' ]
    else:
        df[ columns_bool[ 0 ] ] = df[ 'lk4_img' ] + df[ 'lk5_img' ] # fraction of YES answer for text+image
        df[ columns_bool[ 3 ] ] = df[ 'lk4_txt' ] + df[ 'lk5_txt' ] # fraction of YES answer for text only
        df[ columns_bool[ 2 ] ] = df[ 'lk3_img' ] + df[ 'unk_img' ] # fraction of neutral or missing answer for text+image
        df[ columns_bool[ 5 ] ] = df[ 'lk3_txt' ] + df[ 'unk_txt' ] # fraction of neutral or missing answer for text only
    df[ columns_bool[ 1 ] ] = df[ 'lk1_img' ] + df[ 'lk2_img' ] # fraction of NO answer for text+image
    df[ columns_bool[ 4 ] ] = df[ 'lk1_txt' ] + df[ 'lk2_txt' ] # fraction of NO answer for text only

    df = df.drop( columns=old_columns )

    return df


def unify_demo( df, demo_cat ):
    """
    Convert a dataframe with Likert values into boolean scores
    Use the standard column naming for boolean results

    params:
        df              [pandas.core.frame.DataFrame] the data in pandas DataFrame
        demo_cat        [tuple] with the demographics categories to unify

    return:             df modified in place
    """
    assert len( demo_cat ) == 3, "unify_demo() expects exactly 3 demographics categories"

    c1, c2, c3      = demo_cat
#   df[ 'demo' ]    = df[ c1 ].str[ 0 ].lower() + '/' + df[ c2 ].str[ 0 ].lower() + '/' + df[ c3 ].str[ 0 ].lower()
    df[ 'demo' ]    = df.apply(
            lambda row: f"{row[ c1 ][0].lower()}/{row[ c2 ][0].lower()}/{row[ c3 ][0].lower()}",
            axis=1 )

    return df


def print_means( f, df, dft, groups, scores=['yes_img','yes_txt'] ):
    """
    print means and std of the main scores grouped as requested
    NOTE the use of multiple dataframes, in order to avoid possible side effects of multiple records for same news
    in melted dafarames, like for tags or unified yes output

    params:
        df              [pandas.core.frame.DataFrame] the data in pandas DataFrame
        dft             [pandas.core.frame.DataFrame] as df, unified by tags
        groups          [list] of lists with independent variables column names
        scores          [list] of dependent variables column names

    """
    models      = df[ "model" ].unique().tolist()
    dfy         = unify_yes( df )                   # dataframe melted for unified "yes" output
    dfty        = unify_yes( dft )                  # dataframe melted for unified "tag" and "yes" output

    for group in groups:
        if "tag" in group:                          # use melted dataframes when needed
            d   = dft
            dy  = dfty
        else:
            d   = df                                # otherwise, the original one
            dy  = dfy
        res     = d.groupby( group, observed=True )[ scores ].agg( [ "mean", "std" ] )
        f.write( f" {group} ".center( 80, '=' ) + '\n\n' )
        f.write( res.to_string() + '\n\n' )
        f.write( 80 * "=" + "\n\n" )
        if len( models ) > 1:
            res     = d.groupby( [ "model" ] + group, observed=True )[ scores ].agg( [ "mean", "std" ] )
            f.write( f" {group} by models ".center( 80, '=' ) + '\n\n' )
            f.write( res.to_string() + '\n\n' )
            f.write( 80 * "=" + "\n\n" )
        if "value" in group and "yes_img" in scores:
            res     = dy.groupby( group, observed=True )[ "yes" ].agg( [ "mean", "std" ] )
            f.write( f" {group} for YES ".center( 80, '=' ) + '\n\n' )
            f.write( res.to_string() + '\n\n' )
            f.write( 80 * "=" + "\n\n" )
            if len( models ) > 1:
                res     = dy.groupby( [ "model" ] + group, observed=True )[ "yes" ].agg( [ "mean", "std" ] )
                f.write( f" {group} for YES by models ".center( 80, '=' ) + '\n\n' )
                f.write( res.to_string() + '\n\n' )
                f.write( 80 * "=" + "\n\n" )


def unify_tags( df ):
    """
    unify the 4 tags columns using the melt() function of Pandas, into the new column "tag"
    uses the melt() function of Pandas to select tag values across all possible "tagX" columns
    get rid of duplicated tags with "unknown" content
    NOTE: if the news dataset contains news without tags, their records will be removed as well

    params:
        df              [pandas.core.frame.DataFrame] the data in pandas DataFrame

    return:             [pandas.core.frame.DataFrame]
    """
    tag_cols        = [ "tag1", "tag2", "tag3", "tag4" ]

    all_cols    = list( df.columns )
    keep_cols   = list( set( all_cols ) - set( tag_cols ) )
    dft         = df.melt( id_vars=keep_cols, value_vars=tag_cols, var_name='tag_column', value_name='tag' )
    dft         = dft.drop( dft[ dft.tag == "unknown" ].index )

    return dft


def unify_yes( df ):
    """
    unify the two "yes" columns using the melt() function of Pandas, into the new column "yes"
    uses the melt() function of Pandas

    params:
        df              [pandas.core.frame.DataFrame] the data in pandas DataFrame

    return:             [pandas.core.frame.DataFrame]
    """
    yes_cols        = [ 'yes_img', 'yes_txt' ]

    all_cols    = list( df.columns )
    keep_cols   = list( set( all_cols ) - set( yes_cols ) )
    dfy         = df.melt( id_vars=keep_cols, value_vars=yes_cols, var_name='yes_column', value_name='yes' )

    return dfy


def means_tags( df, no_value=False ):
    """
    Compute means of the main scores grouped by news tags and other variables independently.
    params:
        df              [pandas.core.frame.DataFrame] the data in pandas DataFrame
        no_value        [bool] do not groupo by "value"

    return:             [tuple] means for profile, age, gender, race, edu, politic (all models), and same by models
    """
    if "yes_img" in df.columns:
        scores      = [ 'yes_img', 'yes_txt' ]
    else:
        scores      = [
            "lk1_img",
            "lk2_img",
            "lk3_img",
            "lk4_img",
            "lk5_img",
            "lk1_txt",
            "lk2_txt",
            "lk3_txt",
            "lk4_txt",
            "lk5_txt"
        ]
    dft         = unify_tags( df )
    if no_value:
        mt          = dft.groupby( [ 'tag' ] )[ scores ].mean().reset_index()
        mmt         = dft.groupby( [ 'model', 'tag' ] )[ scores ].mean().reset_index()
    else:
        mt          = dft.groupby( [ 'value', 'tag' ] )[ scores ].mean().reset_index()
        mmt         = dft.groupby( [ 'model', 'value', 'tag' ] )[ scores ].mean().reset_index()

    return mt, mmt


def pearson( df, x, y ):
    """
    compute Pearson's coefficient
    Construct the regression formula in the 'R'-style used in the ols function of statsmodels
    the p value derives from Student t applied to
        t = r * sqrt( ( n-2 ) / ( 1 - r^2 ) )
    take into account if the x column is numeric or not (y should be)

    params:
        df              [pandas.core.frame.DataFrame] the data in pandas DataFrame
        x               [str] one of the independent variables
        y               [str] one of the dependent variables

    return:             [tuple] r, p-value
    """
    numeric     = df[ x ].dtype.kind in 'biuf'  # bool/int/uint/float
    if numeric:
        formula     = f"{y} ~ {x}"
    else:
        formula     = f"{y} ~ C({x})"
    model       = ols( formula, df ).fit()
    r2          = model.rsquared
    r2          = max( r2, 0.0 )                # for tiny numbers may be even negative
    r           = np.sqrt( r2 )
    if numeric:                                 # then use a sign for r
        sign        = np.sign( model.params[ x ] )
        r           *= sign
    p           = model.pvalues[ x ]

    return r, p


def anova_1( df, x, y ):
    """
    compute one-way anova of one independent categorial variable x against the result y
    Construct the regression formula in the 'R'-style used in the ols function of statsmodels,
    then apply the anova_lm to the regression model returned by ols
    See:
        https://www.statsmodels.org/dev/generated/statsmodels.formula.api.ols.html
        https://www.statsmodels.org/dev/generated/statsmodels.stats.anova.anova_lm.html
    Moreover, compute the effect size by eta^2 of F, then squared, assuming that degree-of-freedom_Hyp=1
        r = sqrt( (F * df1 ) / ( F * df1 + df2 ) )

    params:
        df              [pandas.core.frame.DataFrame] the data in pandas DataFrame
        x               [str] one of the independent variables
        y               [str] one of the dependent variables

    return:             [tuple] F, df2, r, p-value
    """
    n           = len( df ) - 2
    formula     = f"{y} ~ C({x})"
    model       = ols( formula, df ).fit()
    a           = anova_lm( model )
    va          = a.values
#   print( va )
    f           = va[ 0 ][ 3 ]                      # this is F
    p           = va[ 0 ][ 4 ]                      # this is p-value
    r           = np.sqrt( f / ( f + n ) )          # the "r" effect size

    return f, n, r, p


def shapiro_wilk( df ):
    """
    test normality distribution of the difference yes_img - yes_txt
    case with N < 5000
    see:
        https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test
    params:
        df              [pandas.core.frame.DataFrame] the data in pandas DataFrame

    return:             [float] p_value
    """
    # the column names of the two dependent variables to compare
    yes_img     = "yes_img"
    yes_txt     = "yes_txt"

    assert yes_img in df.columns, "column {yes_img} not found in the dataframe"

    yes_diff    = df[ yes_img ] - df[ yes_txt ]
    _, p_value  = shapiro( yes_diff )

    return p_value


def kolmogorov_smirnov( df ):
    """
    test normality distribution of the difference yes_img - yes_txt
    case with N > 5000
    see:
        https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    params:
        df              [pandas.core.frame.DataFrame] the data in pandas DataFrame

    return:             [float] p_value
    """
    # the column names of the two dependent variables to compare
    yes_img     = "yes_img"
    yes_txt     = "yes_txt"

    assert yes_img in df.columns, "column {yes_img} not found in the dataframe"

    yes_diff    = df[ yes_img ] - df[ yes_txt ]
    yes_mean    = yes_diff.mean()
    yes_std     = yes_diff.std()
    res         = kstest( yes_diff, 'norm', args=( yes_mean, yes_std ) )

    return res.pvalue


def normality_test( df ):
    """
    test normality distribution of the difference yes_img - yes_txt
    params:
        df              [pandas.core.frame.DataFrame] the data in pandas DataFrame

    return:             [float] p_value
    """
    if len( df ) > 500:
        return kolmogorov_smirnov( df )

    return shapiro_wilk( df )


def wilcoxon_stat( df ):
    """
    Compute the Wilcoxon test p-value and the effect size using the rank-biserial correlation
    for the image/no-image scores
    the effect size formula is r = Z/sqrt(N)
    where Z is the z-score equivalent of the Wilcoxon test, and N is the number of non-zero differences
    see:
        https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
    params:
        df              [pandas.core.frame.DataFrame] the data in pandas DataFrame

    return:             [tuple] size_effect, p_value
    """
    # the column names of the two dependent variables to compare
    yes_img     = "yes_img"
    yes_txt     = "yes_txt"

    assert yes_img in df.columns, "column {yes_img} not found in the dataframe"

    s, p_value  = wilcoxon( df[ yes_img ], df[ yes_txt ] )

    yes_diff    = df[ yes_img ] - df[ yes_txt ]
    non_0_diff  = yes_diff[ yes_diff != 0 ]         # drop zero differences
    non_0_sqr   = np.sqrt( len( non_0_diff ) )      # this is the denominator in the formula for r
    if not norm:
        return 0, p_value

    # execute Wilcoxon again, but with correction=False to get the z-statistic
    res         = wilcoxon( df[ yes_img ], df[ yes_txt ], correction=False )
    # compute now the approximate normal z from the p-value, assuming two-tailed
    z_approx    = norm.isf( res.pvalue / 2 )
    size_effect = z_approx / non_0_sqr

    return size_effect, p_value


def mixedmod_stat( df, full_output=False ):
    """
    Compute the linear mixed-effects model for the within-subjects factor ("yes_img", "yes_txt" )
    and the between-subjects factor "value" (true, false)
    params:
        df              [pandas.core.frame.DataFrame] the data in pandas DataFrame
        full_output     [bool] print the complete model report

    return:             [tuple] the amount of interaction true/false -> yes_img/yes_txt, and its p-value
    """
    # the column names of the two dependent variables to compare, NOTE: this order is important
    scores              = [ "yes_txt", "yes_img" ]
    # the column names of the two independent variables
    id_vars             = [ 'news', 'value' ]

    assert "yes_img" in df.columns, "column {yes_img} not found in the dataframe"

    # there are now two new columns: "yes" that is either "yes_txt" or "yes_img", and "modality" for txt or img
    df_long             = df.melt( id_vars=id_vars, value_vars=scores, var_name='modality', value_name='yes' )
    # set the order for modality and value so to have as interaction modality[T.yes_img]:value[T.false]
    df_long.modality    = pd.Categorical( df_long.modality, categories=scores, ordered=True )
    df_long.value       = pd.Categorical( df_long.value, categories=[ "true", "false" ], ordered=True )
    model               = mixedlm( "yes ~ modality * value", data=df_long, groups=df_long[ "news" ] )
    result              = model.fit()

    if full_output:
        print( result.summary() )

    params              = result.params
    pvalues             = result.pvalues

    # the order of data is:
    # [ Intercept, modality[T.yes_img], value[T.true], modality[T.yes_img]:value[T.false] Group_Var ]
    interact            = params.iloc[ 3 ]
    p_val               = pvalues.iloc[ 3 ]

    return interact, p_val


def do_radar_demo( df ):
    """
    do radar plots for demographics
    produces 4 series of plots, for all combinations of the current categories grouped by 3
    plot names contain a code for the combination:
        sar:     "sex", "age",  "race"
        sap:     "sex", "age",  "party"
        srp:     "sex", "race", "party"
        arp:     "age", "race", "party"
    """
    scores  = [ 'yes_img', 'yes_txt' ]

    cats    = [
        [ "sex", "age",  "race" ],
        [ "sex", "age",  "party" ],
        [ "sex", "race", "party" ],
        [ "age", "race", "party" ]
    ]

    dfc     = unify_demo( df, cats[ 0 ] )
    # repeat [ "sex", "age", "race" ] for democratic only
    cat_code    = "sarD"
    dfcd    = dfc[ dfc[ 'party' ] == 'democratic' ]
    mt      = dfcd.groupby( [ 'value', 'demo' ], observed=True )[ scores ].mean().reset_index()
    mmt     = dfcd.groupby( [ 'model', 'value', 'demo' ], observed=True )[ scores ].mean().reset_index()
    fname   = os.path.join( dir_stat, f_plot ) + "_demo_" + cat_code + "_ft"
    plot.plot_models_radar( mt, mmt, "demo", group="value", fname=fname )
    mt      = dfcd.groupby( [ 'demo' ], observed=True )[ scores ].mean().reset_index()
    mmt     = dfcd.groupby( [ 'model', 'demo' ], observed=True )[ scores ].mean().reset_index()
    fname   = os.path.join( dir_stat, f_plot ) + "_demo_" + cat_code
    plot.plot_models_radar( mt, mmt, "demo", group=None, fname=fname )

    # repeat [ "age", "race", "party" ] for republican only
    cat_code    = "sarR"
    dfcr    = dfc[ dfc[ 'party' ] == 'republican' ]
    mt      = dfcr.groupby( [ 'value', 'demo' ], observed=True )[ scores ].mean().reset_index()
    mmt     = dfcr.groupby( [ 'model', 'value', 'demo' ], observed=True )[ scores ].mean().reset_index()
    fname   = os.path.join( dir_stat, f_plot ) + "_demo_" + cat_code + "_ft"
    plot.plot_models_radar( mt, mmt, "demo", group="value", fname=fname )
    mt      = dfcr.groupby( [ 'demo' ], observed=True )[ scores ].mean().reset_index()
    mmt     = dfcr.groupby( [ 'model', 'demo' ], observed=True )[ scores ].mean().reset_index()
    fname   = os.path.join( dir_stat, f_plot ) + "_demo_" + cat_code
    plot.plot_models_radar( mt, mmt, "demo", group=None, fname=fname )

    for cat in cats:
        cat_code = ''.join( word[ 0 ] for word in cat )
        dfc     = unify_demo( df, cat )
        mt      = dfc.groupby( [ 'value', 'demo' ], observed=True )[ scores ].mean().reset_index()
        mmt     = dfc.groupby( [ 'model', 'value', 'demo' ], observed=True )[ scores ].mean().reset_index()
        fname   = os.path.join( dir_stat, f_plot ) + "_demo_" + cat_code + "_ft"
        plot.plot_models_radar( mt, mmt, "demo", group="value", fname=fname )
        mt      = dfc.groupby( [ 'demo' ], observed=True )[ scores ].mean().reset_index()
        mmt     = dfc.groupby( [ 'model', 'demo' ], observed=True )[ scores ].mean().reset_index()
        fname   = os.path.join( dir_stat, f_plot ) + "_demo_" + cat_code
        plot.plot_models_radar( mt, mmt, "demo", group=None, fname=fname )

    # repeat [ "age", "race", "party" ] for female only
    cat_code    = "arpF"
    dfcf    = dfc[ dfc[ 'sex' ] == 'F' ]
    mt      = dfcf.groupby( [ 'value', 'demo' ], observed=True )[ scores ].mean().reset_index()
    mmt     = dfcf.groupby( [ 'model', 'value', 'demo' ], observed=True )[ scores ].mean().reset_index()
    fname   = os.path.join( dir_stat, f_plot ) + "_demo_" + cat_code + "_ft"
    plot.plot_models_radar( mt, mmt, "demo", group="value", fname=fname )
    mt      = dfcf.groupby( [ 'demo' ], observed=True )[ scores ].mean().reset_index()
    mmt     = dfcf.groupby( [ 'model', 'demo' ], observed=True )[ scores ].mean().reset_index()
    fname   = os.path.join( dir_stat, f_plot ) + "_demo_" + cat_code
    plot.plot_models_radar( mt, mmt, "demo", group=None, fname=fname )

    # repeat [ "age", "race", "party" ] for male only
    cat_code    = "arpM"
    dfcm    = dfc[ dfc[ 'sex' ] == 'M' ]
    mt      = dfcm.groupby( [ 'value', 'demo' ], observed=True )[ scores ].mean().reset_index()
    mmt     = dfcm.groupby( [ 'model', 'value', 'demo' ], observed=True )[ scores ].mean().reset_index()
    fname   = os.path.join( dir_stat, f_plot ) + "_demo_" + cat_code + "_ft"
    plot.plot_models_radar( mt, mmt, "demo", group="value", fname=fname )
    mt      = dfcm.groupby( [ 'demo' ], observed=True )[ scores ].mean().reset_index()
    mmt     = dfcm.groupby( [ 'model', 'demo' ], observed=True )[ scores ].mean().reset_index()
    fname   = os.path.join( dir_stat, f_plot ) + "_demo_" + cat_code
    plot.plot_models_radar( mt, mmt, "demo", group=None, fname=fname )



def do_radar_plots( df ):
    """
    do radar plots
    """
    scores  = [ 'yes_img', 'yes_txt' ]
    if "lk1_img" in df.columns:
        df  = likert_to_bool( df )

    if len( df[ "profile" ].unique() ) > 1:
        mt      = df.groupby( [ 'value', 'profile' ], observed=True )[ scores ].mean().reset_index()
        mmt     = df.groupby( [ 'model', 'value', 'profile' ], observed=True )[ scores ].mean().reset_index()
        fname   = os.path.join( dir_stat, f_plot ) + "_pro_ft"
        plot.plot_models_radar( mt, mmt, "profile", group="value", fname=fname )
        mt      = df.groupby( [ 'profile' ], observed=True )[ scores ].mean().reset_index()
        mmt     = df.groupby( [ 'model', 'profile' ], observed=True )[ scores ].mean().reset_index()
        fname   = os.path.join( dir_stat, f_plot ) + "_pro"
        plot.plot_models_radar( mt, mmt, "profile", group=None, fname=fname )
    elif len( df[ "age" ].unique() ) > 1:
        do_radar_demo( df )

    mt, mmt = means_tags( df )  # grouped by tag
    fname   = os.path.join( dir_stat, f_plot ) + "_tag_ft"
    plot.plot_models_radar( mt, mmt, "tag", group="value", fname=fname )
    mt, mmt = means_tags( df, no_value=True )  # grouped by tag
    fname   = os.path.join( dir_stat, f_plot ) + "_tag"
    plot.plot_models_radar( mt, mmt, "tag", group=None, fname=fname )


def do_single_plots( df ):
    """
    Do single plots
    """
    if "yes_img" in df.columns:
        values      = [ 'yes_img', 'yes_txt' ]
    else:
        values      = [
            "unk_img",
            "lk1_img",
            "lk2_img",
            "lk3_img",
            "lk4_img",
            "lk5_img",
            "unk_txt",
            "lk1_txt",
            "lk2_txt",
            "lk3_txt",
            "lk4_txt",
            "lk5_txt"
        ]
    fname       = os.path.join( dir_stat, f_plot )
    plot.plot_models_single( df, "profile", group="value", values=values, fname=fname )


def do_multiple_plots( df ):
    """
    Do various plots
    """
    if "yes_img" in df.columns:
        values      = [ 'yes_img', 'yes_txt' ]
    else:
        values      = [
            "unk_img",
            "lk1_img",
            "lk2_img",
            "lk3_img",
            "lk4_img",
            "lk5_img",
            "unk_txt",
            "lk1_txt",
            "lk2_txt",
            "lk3_txt",
            "lk4_txt",
            "lk5_txt"
        ]
    fname       = os.path.join( dir_stat, f_plot )
    plot.plot_models( df, groups=[ "value", "age" ], values=values, fname=fname )
    plot.plot_models( df, groups=[ "value", "race" ], values=values, fname=fname )
    plot.plot_models( df, groups=[ "value", "profile" ], values=values, fname=fname )
    plot.plot_models( df, groups=[ "value", "predia" ], values=values, fname=fname )
    plot.plot_models( df, groups=[ "value", "postdia" ], values=values, fname=fname )
    if f_demo == "demo_small.json":
        plot.plot_models( df, groups=[ "value", "sex" ], values=values, fname=fname )
        plot.plot_models( df, groups=[ "value", "party" ], values=values, fname=fname )
    else:
        plot.plot_models( df, groups=[ "value", "edu" ], values=values, fname=fname )
        plot.plot_models( df, groups=[ "value", "gender" ], values=values, fname=fname )
        plot.plot_models( df, groups=[ "value", "politic" ], values=values, fname=fname )
    unk = [ "unk_img", "unk_txt" ]
    plot.plot_models( df, groups=[ "value", "predia" ], values=unk, fname=fname+"_unk" )
    plot.plot_models( df, groups=[ "value", "postdia" ], values=unk, fname=fname+"_unk" )



def print_profile_corr( f, df ):
    """
    Print correlations for single profiles against "void"

    params:
        f       [TextIOWrapper] text stream of the output file
        df      [pandas.core.frame.DataFrame] the data in pandas DataFrame
    """
    profiles        = df.profile.unique().tolist()
    profiles.remove( 'void' )
    f.write( 80 * "=" + "\n" )
    f.write( " profile correlations ".center( 80, ' ' ) + '\n' )
    f.write( 80 * "=" + "\n\n" )
    f.write( f"                        yes_img             yes_txt       yes\n" )
    f.write( f"model  profile         r    p-value     r   p-value     r    p-value\n" )
    f.write( "________________________________________________________________________________________________\n" )
    m               = "all"
    # cycle over all possible profile values, excluding "void"
    for x in profiles:
        # filter records keeping only those with one profile (x) and profile "void"
        dfp             = df[ df.profile.isin( [ "void", x ] )  ].copy()
        # add a numeric colomn which is 1 when profile==x, otherwise 0
        dfp[ "x" ]      = dfp[ 'profile' ].str.contains(x, case=False, na=False ).astype(int)
        dfpy            = unify_yes( dfp )
        ri, pi          = pearson( dfp, "x", "yes_img" )
        rt, pt          = pearson( dfp, "x", "yes_txt" )
        ry, py          = pearson( dfpy, "x", "yes" )
        f.write( f"{m:<8} {x:<10}{ri:6.3f}  {pi:5.4f}   {rt:6.3f}  {pt:5.4f}   {ry:6.3f}  {py:5.4f}\n" )
    models      = df[ "model" ].unique().tolist()
    # do the same as above, separated by models
    if len( models ) > 1:
        for m in models:
            dfm     = df[ (df[ 'model' ]==m ) ]
            for x in profiles:
                dfp             = dfm[ dfm.profile.isin( [ "void", x ] )  ].copy()
                dfp[ "x" ]      = dfp[ 'profile' ].str.contains(x, case=False, na=False ).astype(int)
                dfpy            = unify_yes( dfp )
                ri, pi          = pearson( dfp, "x", "yes_img" )
                rt, pt          = pearson( dfp, "x", "yes_txt" )
                ry, py          = pearson( dfpy, "x", "yes" )
                f.write( f"{m:<8} {x:<10}{ri:6.3f}  {pi:5.4f}   {rt:6.3f}  {pt:5.4f}   {ry:6.3f}  {py:5.4f}\n" )
    f.write( "________________________________________________________________________________________________\n\n" )




def print_anova_1( f, df, dft, groups ):
    """
    Print one-way anova
    NOTE the use of multiple dataframes, in order to avoid possible side effects of multiple records for same news
    in melted dafarames, like for tags or unified yes output

    params:
        df              [pandas.core.frame.DataFrame] the data in pandas DataFrame
        dft             [pandas.core.frame.DataFrame] as df, unified by tags
    """
    dfy             = unify_yes( df )
    dfty            = unify_yes( dft )
    f.write( 80 * "=" + "\n" )
    f.write( " one-way anova ".center( 80, ' ' ) + '\n' )
    f.write( 80 * "=" + "\n\n" )
    f.write( f"                              yes_img                    yes_txt                     yes\n" )
    f.write( f"model  ind variable        F        r      p-value     F        r      p-value     F        r      p-value\n" )
    f.write( "___________________________________________________________________________________________________________\n" )
    m               = "all"
    for x in groups:
        if x == "tag":                              # use melted dataframes when needed
            d           = dft
            dy          = dfty
        else:
            d           = df                                # otherwise, the original one
            dy          = dfy
        fi, ni, ri, pi  = anova_1( d, x, "yes_img" )
        ft, nt, rt, pt  = anova_1( d, x, "yes_txt" )
        fy, ny, ry, py  = anova_1( dy, x, "yes" )
        f.write( f"{m:<8} {x:<10}{fi:7.1f}({ni:5d}) {ri:4.3f}  {pi:5.4f} {ft:7.1f}({nt:5d}) {rt:4.3f}  {pt:5.4f} {fy:7.1f}({ny:5d}) {ry:4.3f}  {py:5.4f}\n" )
    models      = df[ "model" ].unique().tolist()
    if len( models ) > 1:
        for m in models:
            dfm     = df[ (df[ 'model' ]==m ) ]
            dfmt    = dft[ (dft[ 'model' ]==m ) ]
            dfmy    = dfy[ (dfy[ 'model' ]==m ) ]
            dfmty   = dfty[ (dfty[ 'model' ]==m ) ]
            for x in groups:
                if x == "tag":
                    d   = dfmt
                    dy  = dfmty
                else:
                    d   = dfm
                    dy  = dfmy
                fi, ni, ri, pi  = anova_1( d, x, "yes_img" )
                ft, nt, rt, pt  = anova_1( d, x, "yes_txt" )
                fy, ny, ry, py  = anova_1( dy, x, "yes" )
                f.write( f"{m:<8} {x:<10}{fi:7.1f}({ni:5d}) {ri:4.3f}  {pi:5.4f} {ft:7.1f}({nt:5d}) {rt:4.3f}  {pt:5.4f} {fy:7.1f}({ny:5d}) {ry:4.3f}  {py:5.4f}\n" )
    f.write( "___________________________________________________________________________________________________________\n" )


def print_wilcoxon( f, df, group="tag" ):
    """
    print higher-level statistics: Wilcoxon
        f       [TextIOWrapper] text stream of the output file
        df      [pandas.core.frame.DataFrame] the data in pandas DataFrame
        group   [str] one of the independent variables, or "" for no grouping
    """
    if len( group ):
        topics      = df[ group ].unique().tolist()
    else:
        topics      = []

    f.write( 80 * "=" + "\n" )
    f.write( f" Wilcoxon test for {group}".center( 80, ' ' ) + '\n' )
    f.write( 80 * "=" + "\n\n" )
    f.write( f"model       {group:<10}   effect  p-value\n" )
    f.write( "_________________________________________\n" )
    size_effect, p_value    = wilcoxon_stat( df )
    t   = "all"
    m   = "all"
    f.write( f"{m:<12} {t:<12} {size_effect:4.3f}  {p_value:5.4f}\n" )
    for t in topics:
        size_effect, p_value    = wilcoxon_stat( df[ df[ group ] == t ] )
        f.write( f"{m:<12} {t:<12} {size_effect:4.3f}  {p_value:5.4f}\n" )
    models      = df[ "model" ].unique().tolist()
    if len( models ) > 1:
        for m in models:
            dfm = df[ (df[ 'model' ]==m ) ]
            size_effect, p_value    = wilcoxon_stat( dfm )
            t   = "all"
            f.write( f"{m:<12} {t:<12} {size_effect:4.3f}  {p_value:5.4f}\n" )
            for t in topics:
                size_effect, p_value    = wilcoxon_stat( dfm[ dfm[ group ] == t ] )
                f.write( f"{m:<12} {t:<12} {size_effect:4.3f}  {p_value:5.4f}\n" )
    f.write( "_________________________________________\n\n" )


def print_mixedmod( f, df, group="tag" ):
    """
    print higher-level statistics: mixed model
        f       [TextIOWrapper] text stream of the output file
        df      [pandas.core.frame.DataFrame] the data in pandas DataFrame
        group   [str] one of the independent variables, or "" for no grouping
    """
    if len( group ):
        topics      = df[ group ].unique().tolist()
    else:
        topics      = []
    f.write( 80 * "=" + "\n" )
    f.write( f" linear mixed model for false/true img/txt for {group}".center( 80, ' ' ) + '\n' )
    f.write( 80 * "=" + "\n\n" )
    f.write( f"model       {group:<10} interaction p-value\n" )
    f.write( "_________________________________________\n" )
    t   = "all"
    m   = "all"
    inter, p    = mixedmod_stat( df )
    f.write( f"{m:<12} {t:<12} {inter:4.3f}  {p:5.4f}\n" )
    for t in topics:
        inter, p    = mixedmod_stat( df[ df[ group ] == t ] )
        f.write( f"{m:<12} {t:<12} {inter:4.3f}  {p:5.4f}\n" )
    models      = df[ "model" ].unique().tolist()
    if len( models ) > 1:
        for m in models:
            t   = "all"
            dfm = df[ (df[ 'model' ]==m ) ]
            inter, p    = mixedmod_stat( dfm )
            f.write( f"{m:<12} {t:<12} {inter:4.3f}  {p:5.4f}\n" )
            for t in topics:
                inter, p    = mixedmod_stat( dfm[ dfm[ group ] == t ] )
                f.write( f"{m:<12} {t:<12} {inter:4.3f}  {p:5.4f}\n" )
    f.write( "_________________________________________\n\n" )


def do_stat( df ):
    """
    do all statistics and write it on file
    which statistics, both at basic and high level, are set with lists detailing
    the group of columns to be analyzed

    """
# general statistics for profile executions
    # for basic statistics test, NOTE that in this case entries should be lists
    # this is the group for "yes..." dependent variables
    title       = ""
    yes_groups  = [
        [ "value" ],
        [ "tagi" ],
        [ "value", "tagi" ],
        [ "tag" ],
        [ "value", "tag" ],
    ]
    # this is the group for agreement dependent variables, the same is used for UNK
    agr_groups  = [
        [ "value" ],
        [ "profile" ],
        [ "value", "profile" ],
        [ "tag" ],
        [ "value", "tag" ],
    ]
    an1_groups  = [ "value", "tag", "tagi", "profile" ] # for anova 1
    wil_groups  = [ "value", "tag", "profile" ]         # for Wilcoxon test
    mml_groups  = [ "tag", "profile" ]                  # for Mixed Model test
    do_profile  = True                                  # do profile correlation

# settings specific for comparing blank_img, without higher statistics
    yes_groups  = [ [ "blank_img" ] ]
    an1_groups  = [ "blank_img" ]
    agr_groups  = []
    wil_groups  = []
    mml_groups  = []
    do_profile  = False                                 # do profile correlation

# settings specific for comparing reason_3steps + reason_share_likert5 with ask_share_noexplain_likert5
    yes_groups  = [ [ "postdia" ] ]
    agr_groups  = [ [ "postdia" ] ]
    an1_groups  = [ "postdia" ]
    wil_groups  = []
    mml_groups  = []
    do_profile  = False                                 # do profile correlation

# settings specific for boolean, no comparison...
    yes_groups  = [ [ "value" ], [ "value", "tag" ] ]
    agr_groups  = []
    an1_groups  = [ "value" ]
    wil_groups  = [ "value" ]
    mml_groups  = [ "" ]
    do_profile  = False                                 # do profile correlation

# settings specific for comparing ask_user_dsc and ask_dsc, without higher statistics
    yes_groups  = [ [ "postdia" ], [ "postdia", "profile" ], [ "postdia", "value", "profile" ] ]
    agr_groups  = [ [ "postdia" ], [ "postdia", "profile" ], [ "postdia", "value" ] ]
    an1_groups  = [ "postdia" ]
    wil_groups  = []
    mml_groups  = []
    do_profile  = False                                 # do profile correlation

# general statistics for demographic executions
    title       = "demography for all models"
    yes_groups  = [
        [ "sex" ],
        [ "age" ],
        [ "party" ],
        [ "race" ],
        [ "sex", "party" ],
        [ "sex", "party", "age" ],
        [ "sex", "party", "race" ],
        [ "value", "sex" ],
        [ "value", "age" ],
        [ "value", "party" ],
        [ "value", "race" ],
        [ "tag", "sex" ],
        [ "tag", "age" ],
        [ "tag", "party" ],
        [ "tag", "race" ],
    ]
    agr_groups  = [
        [ "sex" ],
        [ "age" ],
        [ "party" ],
        [ "race" ],
    ]
    an1_groups  = [ "sex", "age", "party", "race" ]     # for anova 1
    wil_groups  = [ "sex", "age", "party", "race" ]     # for Wilcoxon test
    mml_groups  = [ "sex", "age", "party", "race" ]     # for Mixed Model test
    do_profile  = False                                 # do profile correlation

# general statistics for all executions (profiles and demographics) - only common groups can be analyzed
    title       = "global statistics on all executions"
    yes_groups  = [ [ "value" ], [ "tag" ], [ "value", "tag" ], ]
    agr_groups  = [ [ "value" ], [ "tag" ], ]
    an1_groups  = [ "value", "tag", "tagi" ]            # for anova 1
    wil_groups  = [ "value", "tag" ]                    # for Wilcoxon test
    mml_groups  = [ "tag", "" ]                         # for Mixed Model test
    do_profile  = False                                 # do profile correlation

# statistics specific for profiles
    title       = "statistics for profiling (no Wilcoxon)"
    yes_groups  = [ [ "value" ], [ "profile" ], [ "value", "profile" ], ]
    agr_groups  = [ [ "value" ], [ "profile" ], ]
    an1_groups  = []                                    # for anova 1
    wil_groups  = []                                    # for Wilcoxon test
    mml_groups  = []                                    # for Mixed Model test
    do_profile  = True                                  # do profile correlation

    scores      = [ 'yes_img', 'yes_txt' ]              # assume statistics for boolean results only
    if not "yes_img" in df.columns:
        df      = likert_to_bool( df )
    dft         = unify_tags( df )                      # unify the topics tags

    # do basic statistics first
    fname       = os.path.join( dir_stat, f_bstat )
    f           = open( fname, 'w' )
    f.write( 80 * "=" + "\n\n" )
    f.write( 80 * "+" + "\n" )
    f.write( "basic statistics for YES scores".center( 80, ' ' ) + '\n' )
    f.write( title.center( 80, ' ' ) + '\n' )
    f.write( 80 * "+" + "\n\n" )
    print_means( f, df, dft, yes_groups, scores=['yes_img','yes_txt'] )
    if len( agr_groups ):
        f.write( 80 * "+" + "\n" )
        f.write( "basic statistics for agreement scores".center( 80, ' ' ) + '\n' )
        f.write( 80 * "+" + "\n\n" )
        print_means( f, df, dft, agr_groups, scores=['agr_img','agr_txt'] )
        f.write( 80 * "+" + "\n" )
        f.write( "basic statistics for unresponses scores".center( 80, ' ' ) + '\n' )
        f.write( 80 * "+" + "\n\n" )
        print_means( f, df, dft, agr_groups, scores=['unk_img','unk_txt'] )

    if do_profile:
        print_profile_corr( f, df )
    if len( an1_groups ):
        print_anova_1( f, df, dft, an1_groups )

    f.close()

    if ( not len( wil_groups ) ) and ( not len( mml_groups ) ):
        return True

    # now do higher statistics
    fname       = os.path.join( dir_stat, f_hstat )
    f           = open( fname, 'w' )
    f.write( 80 * "+" + "\n" )
    f.write( "higher statistics".center( 80, ' ' ) + '\n' )
    f.write( title.center( 80, ' ' ) + '\n' )
    f.write( 80 * "+" + "\n\n" )
    p_value     = normality_test( df )
    f.write( f"the preliminary test for normality of the distribution gives p-value={p_value}\n" )
    f.write( 80 * "+" + "\n\n" )
    for group in wil_groups:
        if group == "tag":
            print_wilcoxon( f, dft, group=group )
        else:
            print_wilcoxon( f, df, group=group )
    for group in mml_groups:
        if group == "tag":
            print_mixedmod( f, dft, group=group )
        else:
            print_mixedmod( f, df, group=group )

    f.close()

    return True



# ===================================================================================================================
#
#   MAIN
#
# ===================================================================================================================
if __name__ == '__main__':
    if DO_NOTHING:
        print( "program instructed to do nothing" )
    else:
        # first, it is necessary to read the demographics from json
        read_demo()
        # ...and the news tags from json
        read_tags()
        df          = collect_data()        # all data in pandas DataFrame
        now_time    = time.strftime( frmt_statdir )
        dir_stat    = os.path.join( dir_stat, now_time )
        if not os.path.isdir( dir_stat ):
            os.makedirs( dir_stat )
        # save a copy of this file and the plotting script
        shutil.copy( "infstat.py", dir_stat )
        shutil.copy( "plot.py", dir_stat )
        if DO_MPLOTS:
            do_multiple_plots( df )
        if DO_LKPLOT:
            fname       = os.path.join( dir_stat, f_plot )
            plot.plot_models_likert( df, "profile", group="value", fname=fname )
        if DO_SPLOTS:
            do_single_plots( df )
        if DO_STATS:
            do_stat( df )
        if DO_RADAR:
            do_radar_plots( df )
