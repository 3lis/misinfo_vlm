"""
#####################################################################################################################

    Print a summary of executions in res/

#####################################################################################################################
"""

import  os
import  sys
import  re

res         = "../res"
log         = "log.txt"
short_scan  = True                          # do not report numerical results, just settings

versions    = [ "25-02-02_11-42-08" ]
shortcuts   = {
    'ask_img':                  'ask_im',
    'ask_share':                'ask_sh',
    'ask_share_likert5':        'ask_lk',
    'ask_share_strict':         'ask_ss',
    'ask_share_strict_likert5': 'ask_sl',
    'ask_share_noexplain':      'ask_ne',
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
    'p_void':                   'p_void',
    'context':                  'ctx___',
    'context_strict':           'ctx_st',
    'reason_3steps':            'rea_3s',
    'reason_base':              'rea_bs',
    'reason_share':             'rea_sh',
    'reason_share_xml':         'rea_sx',
    'reason_share_delimit':     'rea_sd',
}


def get_version ( folder ):
    """
    retrieve the version of the logs based on the folder date
    """
    for v, f in enumerate( versions ):
        if folder < f:
            return v
    return v + 1


def get_info ( lines, version=2 ):
    """
    retrieve essential info
    """

    experiment      = "-----"
    dialogs_pre     = []
    dialogs_post    = []
    model           = "---"
    pre             = ""
    post            = ""
    means           = "---"
    profile         = "---"
    demo            = "no"
    for i, l in enumerate( lines ):
        if "News" in l:
            break
        if "experiment" in l:
            experiment      = l.split()[ -1 ]
        if "model " in l:
            model           = l.split()[ -1 ]
        if "dialogs_pre" in l:
            dialogs_pre     = re.sub( r'[\W]+', ' ', l ).split()[ 1 : ]
        if "dialogs_post" in l:
            dialogs_post    = re.sub( r'[\W]+', ' ', l ).split()[ 1 : ]
        if "demographics" in l and not "None" in l:
            n   = 1
            while True:
                l   = lines[ i+n ]
                if "sex" in l:
                    demo    = l.split()[ -1 ][ : 3 ]
                    break
                n   += 1
                if n > 5:
                    demo            = "yes"
                    break
    if len( dialogs_pre ):
        for d in dialogs_pre:
            if d in shortcuts.keys():
                pre         += shortcuts[ d ] + ' '
    if len( dialogs_post ):
        for d in dialogs_post:
            if d in shortcuts.keys():
                post        += shortcuts[ d ] + ' '
    if short_scan:
        return experiment, model, pre, post, demo, None

    n       = len( lines )
    split_means = False         # means split for true and false
    while True:
        l   = lines[ i ]
        if "f_mn" in l:
            split_means = True
            break
        if "mean" in l:
            break
        i   += 1
        if i == n:
            return None
    if split_means:
        f_mn            = l[ 6: ].split()
        i   += 1
        l   = lines[ i ]
        if "t_mn" in l:
            t_mn            = l[ 6: ].split()
        else:
            print( "found false means but not true means" )
            return None
        means           = [ t[ : -1 ] + '-' + f[ : -1 ] for t,f in zip( t_mn, f_mn ) ]
        means           = ' '.join( means )
    elif version == 0:
        split           = l.split()
        mean_im_tx      = split[ 2 ]
        mean_tx         = split[ 4 ]
        means           = mean_im_tx + 13 * ' ' + mean_tx
    else:
        means           = l[ 6:-1 ]
        means           = means.replace( '  ', '     ' )

    return experiment, model, pre, post, demo, means


# ===================================================================================================================
#
#   MAIN
#
# ===================================================================================================================

if __name__ == '__main__':
    list_res    = sorted( os.listdir( res ) )

    if len( sys.argv ) > 1:             # if there is an argument, use it at the latest result to show
        first_res   = sys.argv[ 1 ]
        if first_res in list_res:
            idx         = list_res.index( first_res )
            list_res    = list_res[ idx : ]

    n_dash  = 80
    if not short_scan:
        n_dash  *= 2
    print( n_dash * "_" )
    print( "shortcuts:" )
    for k in shortcuts.keys():
        s   = shortcuts[ k ]
        print( f"{k:<25}: {s}" )
    print( n_dash * "_" )
    print()
    header  = "     result     experiment   model"
    header  += 22 * ' '
    header  += "dialogs_pre            dialogs_post"
    header  += "  demo  "
    if not short_scan:
        header  += "YES+i(t/f) NO+(t/f) UN+(t/f)  YES-(t/f) NO-(t/f) UN-(t/f)"
    print( header )
    print( n_dash * "_" )
    for f in list_res:
        version = get_version( f )
        fname   = os.path.join( res, f, log )
        if not os.path.isfile( fname ):
            print( f"{f}  is not a file" )
            continue
        with open( fname, 'r' ) as fd:
            lines   = fd.readlines()
        if not len( lines ):
            print( f"{f}  has no lines" )
            continue
        info        = get_info ( lines, version=version )
        if info is None:
            print( f"{f}  no info found" )
        else:
            e, m, r, p, d, v    = info
            if len( m ) > 25:
                m   = m[ : 25 ] + "..."
        print( f"{f}  {e:<6} {m:<28} {r:<22} {p:<15} {d:<3} {v}" )
    print( n_dash * "_" )
