"""
#####################################################################################################################

    Module for plotting

#####################################################################################################################
"""

import  os
import  sys
import  copy
import  pickle
import  colorsys
import  numpy               as np
import  pandas              as pd
from    matplotlib          import pyplot
from    matplotlib.patches  import Patch, Polygon
from    matplotlib.lines    import Line2D
import  matplotlib          as mpl
# Set the global font to monospaced
mpl.rcParams["font.family"] = "monospace"


figsize         = ( 18.0, 8.0 )                             # figure size in inches
radar_fsize     = ( 8.0, 8.0 )                              # figure size for radar plot
labelspacing    = 1.2
extension       = ".pdf"

# colors organized in 6-shade per 12 groups
dim_colors = [
    [ "#cc2e24", "#e63c2d", "#ff3b30", "#ff6b61", "#ff9c92", "#ffccc3" ],  # Vibrant Red
    [ "#853bb0", "#9b45c2", "#af52de", "#c77bed", "#d9a4ee", "#ebc9f5" ],  # Vibrant Purple
    [ "#239a45", "#2fae54", "#34c759", "#6dd09d", "#9ee3c3", "#cfe6d9" ],  # Vibrant Green
    [ "#008ba3", "#009cbf", "#00bcd4", "#33cfe1", "#66e0ed", "#99f1f9" ],  # Vibrant Teal
    [ "#0051b3", "#0060d1", "#007aff", "#59a3ff", "#a6c8ff", "#d2e4ff" ],  # Vibrant Blue
    [ "#c10f71", "#d51080", "#ff1493", "#ff4c9f", "#ff80b0", "#ffb3c0" ],  # Vibrant Pink
    [ "#99bf00", "#a8bf00", "#BFFF00", "#d3ff33", "#e6ff66", "#f2ff99" ],  # Vibrant Lime (Chartreuse)
    [ "#cc7600", "#e18a00", "#ff9500", "#ffac33", "#ffc366", "#ffdb99" ],  # Vibrant Orange
    [ "#3b90c9", "#4fb1e2", "#5ac8fa", "#80d7ff", "#a6e5ff", "#cfeeff" ],  # Vibrant Cyan
    [ "#128f73", "#179d80", "#1abc9c", "#4ad1b3", "#76e5ca", "#a2f9e1" ],  # Vibrant Turquoise
    [ "#cc9900", "#e6aa00", "#ffcc00", "#ffdb33", "#ffe066", "#fff099" ],  # Vibrant Yellow
    [ "#cc223f", "#e03347", "#ff2d55", "#ff6971", "#ff9b8c", "#ffc5b0" ],  # Vibrant Magenta
]

# the elegant tableau-10 series of colors in matplotlib
# NOTE: there are - of course - just 10 colors
tab_colors  = [ 'tab:blue', 'tab:orange', 'tab:green', 'tab:red',
          'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
          'tab:olive', 'tab:cyan' ]
# overwrite to match barplot color scheme
tab_colors  = [ "#ff3b30", "#9b45c2", "#2fae54", "#0051b3" ]
tab_colors  = [ "#ff3b30", "#ff3b30", "#0051b3", "#0051b3" ]
line_style  = [ '-', '--', '-', '--' ]

# markers
dim_markers = ( 'o',  '*',  'D',  'P',  'X',  'h', 'H', '<', '>', 'x' )

char_len    = 0.009                                     # typical length of a character in legend, in plot units


def gen_colors( n_hues, n_levels, min_value=0.2, saturation=0.8 ) :
    """
    Generates a color palette matrix with customizable intensity and saturation.

    params:
        n_hues      [int] number of different hues (columns)
        n_levels    [int] number of intensity levels (rows)
        min_value   [float] minimum brightness (0 < min_value < 1)
        saturation  [float] saturation of colors (0 to 1)

    Returns:
        List of lists of hex color strings.
    """
    if not ( 0 < min_value <= 1 ):
        raise ValueError("min_value should be between 0 and 1.")
    if not ( 0 <= saturation <= 1 ):
        raise ValueError("saturation should be between 0 and 1.")

    palette     = []
    for level in range( n_levels ):
        # Linear interpolation of value between min_value and 1.0
        value   = min_value + ( level / (n_levels - 1) ) * ( 1 - min_value ) if n_levels > 1 else min_value
        row     = []
        for hue_index in range( n_hues ):
            hue = hue_index / n_hues
            r, g, b = colorsys.hsv_to_rgb( hue, saturation, value )
            hex_color = '#{:02x}{:02x}{:02x}'.format( int(r * 255), int(g * 255), int(b * 255) )
            row.append( hex_color )
        palette.append( row )
    return palette


def plot_values_no_glabel( df, groups=["value","age"], values=["yes_img", "yes_txt"], fname="plot", suptitle='' ):
    """
    Generate one plot of the specified values, for a specified group of independent variables
    this version produces labels only in the legend box, without labels on the X axis for the main group

    params:
        df          [pandas.core.frame.DataFrame] the data in pandas DataFrame
        groups      [list] columns of independent variables
        values      [list] the numerical values to be plot
        fname       [str] name of the output file
        suptitle    [str] plot title
    """

    columns = df.columns
    y       = []                        # list of all vectors to plot
    labels  = []                        # labels for all combinations of independent variables
    cat     = dict()                    # all categorical entries found for independent variables
    comb    = dict()                    # vectors for all combinations of independent variables
    for g in groups:                    # gather the categorical entries for independent variables
        assert g in columns, f"there is no column named {g}"
        cat[ g ]    = df[ g ].unique().tolist()

    for g in groups:                    # collect all vectors for combinations of independent variables
        if not len( comb ):
            first           = True
        else:
            first           = False
            new_comb        = dict()
        for c in cat[ g ]:
            if first:
                comb[ c ]   = df[ df[ g ] == c ]
            else:
                for prev in comb.keys():
                    dframe  = comb[ prev ]
                    new     = prev + '-' + c
                    new_comb[ new ] = dframe[ dframe[ g ] == c ]
        if not first:
            comb            = new_comb

    for c in comb.keys():               # assign vectors to y and their key combinations to labels
        dframe  = comb[ c ]
        for v in values:
            y.append( dframe [ v ] )
            labels.append( f"{v} for {c}" )

    n           = len( y )
    ng          = len( df[ groups[ 0 ] ].unique() ) * len( values )
    # compute a proper separation of bars by group
    x           = [  i + i // ng for i in range( n ) ]
    match ng:                           # see comments in the definition of dim_colors
        case 1:
            colors      = dim_colors[ 0:4*n:4 ]
        case 2:
            colors      = dim_colors[ 0:2*n:2 ]
        case _:
            colors      = dim_colors[ : n ]

    pyplot.rcParams.update( { "font.size": 14 } )
    handles     = [ Patch( facecolor=c, label=l ) for c,l in zip( colors, labels ) ]
    fig, ax     = pyplot.subplots( figsize=figsize )

    for i, ( xx, data ) in enumerate( zip( x, y ) ):
        m   = data.mean()
        s   = data.std() / 2.
        ax.bar( xx, m, yerr=s, color=colors[ i ], linewidth=2 )
        # ax.set_xticks( [] )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_ylabel( "YES fraction" )
    ax.set_ylim( bottom=0.0, top=1.0 )

    # compute an estimate of the max occupancy of labels in the plot
    lab_len     = char_len * max( [ len( l ) for l in labels ] )
    # compute the current geometry of the plotting box
    box = ax.get_position()
    # and now make room on the right for the labels
    ax.set_position([box.x0, box.y0, box.width * ( 1 - lab_len ), box.height])
    ax.legend( handles=handles, loc="center", bbox_to_anchor=(1+lab_len, 0.5), labelspacing=labelspacing )

    fname       = fname + extension
    pyplot.suptitle( suptitle, x=0.4 )
    pyplot.savefig( fname, bbox_inches="tight", pad_inches=0 )
    pyplot.close()
    pyplot.clf()


def plot_single( df, category, group="value", values=["yes_img", "yes_txt"], fname="plot", width=0.9 ):
    """
    Generate one plot of the specified values, for a specified category of one independent variable, and for all values
    of one single group
    the dataset should already be reduced for the value chosen for this independent variable
    in the list of groups, the last is used as main group, which components in the dataset are used as group label
    the first group shouls be the lowest loob, in combination with values
    WARNING: never tested for more than two groups...

    params:
        df          [pandas.core.frame.DataFrame] the data in pandas DataFrame
        category    [str] the name of the category plotted
        group       one column of independent variables
        values      [list] the numerical values to be plot
        fname       [str] name of the output file
        suptitle    [str] plot title
    """

    columns = df.columns
    y       = []                        # list of all vectors to plot
    labels  = []                        # labels for independent variables
    assert group in columns, f"there is no column named {group}"
    comb    = df[ group ].unique().tolist()
    for c in comb:               # assign vectors to y
        dframe  = df[ df[ group ] == c ]
        for v in values:
            y.append( dframe [ v ] )

    for c in comb:                      # assign key combinations of group 0 and values to labels
        for v in values:
            v1, v2  = v.split( '_' )
            # v1      = f"Ls{v1[2]}" if v1.startswith( "lk" ) else v1.upper()
            # v2      = "Im" if v2 == "img" else "Tx"
            # v3      = "T" if c == "true" else "F"
            v1      = f"L0{v1[2]}" if v1.startswith( "lk" ) else v1.upper()
            v2      = "IMG" if v2 == "img" else "TXT"
            v3      = "TRUE" if c == "true" else "FALSE"
            labels.append( f"{v3}  {v2}  {v1}" )

    n           = len( y )                                      # total numer of bars
    nv          = len( values )                                 # number of values to plot
    nc          = len( comb )                                   # number of categories in the group

    # compute a proper separation of bars
    # assume that in case of a large number of values, if they are even, their likely belong to two contiguous sets
    # so, add a separation between the two
    # NOTE: this strategy avoids an additional function parameter, but may not work for future cases
    if ( nv > 4 ) and ( not nv % 2 ):
        nv2     = nv // 2
        x           = [  i + i // nv + i // nv2 for i in range( n ) ]
    else:
        nv2     = nv
        x           = [  i + i // nv for i in range( n ) ]

    colors      = []
    hatches     = []
    for i in range( n // nv2 ):
        for j in range( nv2 ):
            colors.append( dim_colors[ i ][ j ] )
            hatches.append( "/" if j==0 else None )     # THIS WORKS ONLY IF UNK IS FIRST OF LIST

    pyplot.rcParams.update( { "font.size": 14 } )
    handles     = [ Patch( facecolor=c, label=l, hatch=h, edgecolor='white' ) for c,l,h in zip( colors, labels, hatches ) ]
    # handles     = [ Patch( facecolor=c, label=l ) for c,l in zip( colors, labels ) ]

    fig, ax     = pyplot.subplots( figsize=figsize )

    for i, ( xx, data ) in enumerate( zip( x, y ) ):
        m   = data.mean()
        s   = data.std() / 2.
        # bar with means as heigth, and a segment for standard deviation
        ax.bar( xx, m, yerr=s, width=width, color=colors[ i ], hatch=hatches[ i ], linewidth=2, edgecolor='white' )

    # ax.set_xticks( [] )
    ax.set_xticks( x )
    ax.set_xticklabels( labels, rotation=45, ha='right' )

    ax.set_ylabel( "Fraction" )
    ax.set_ylim( bottom=0.0, top=1.0 )

    # compute an estimate of the max occupancy of labels in the plot
    lab_len     = char_len * max( [ len( l ) for l in labels ] )
    # compute the current geometry of the plotting box
    box = ax.get_position()
    # and now make room on the right for the labels
    ax.set_position([box.x0, box.y0, box.width * ( 1 - lab_len ), box.height])
    ax.legend( handles=handles, loc="center", bbox_to_anchor=(1+lab_len, 0.5), labelspacing=labelspacing )

    fname       = f"{fname}_{category}{extension}"
    pyplot.suptitle( category, x=0.4 )
    pyplot.savefig( fname, bbox_inches="tight", pad_inches=0 )
    pyplot.close()
    pyplot.clf()


def plot_likert( df, main_var, group="value", variants=["img","txt"], fname="plot", suptitle="", width=0.9 ):
    """
    Generate a plot specific for Likert scale with stacket bars

    params:
        df          [pandas.core.frame.DataFrame] the data in pandas DataFrame
        main_var    [str] name of the main column of independent variables for which bars are aligned
        group       one column of independent variables
        variants    [list] possible variants of Likert values, elements must be str
        fname       [str] name of the output file
        suptitle    [str] plot title
        width       [float] bar width
    """
    # construct all the names for the used columns of dependent variables
    n_likert    = 5
    likert      = [ f"lk{i}" for i in range( 1, n_likert+1 ) ]
    likert_var  = [ [ l + '_' + v for l in likert ] for v in variants ]

    # ensure the necessary columns are in the dataframe
    columns     = df.columns
    assert main_var in columns, f"there is no column named {main_var}"
    assert group in columns, f"there is no column named {group}"
    for lv in likert_var:
        for l in lv:
            assert l in columns, f"there is no column named {l}"

    # get combinations of values for the independent variables, and their number
    grp_comb    = df[ group ].unique().tolist()
    main_comb   = df[ main_var ].unique().tolist()
    n_main      = len( main_comb )
    n_vars      = len( grp_comb ) * len( variants )

    y       = []                        # list of list with the vector of data
    labels  = []                        # labels for the group and variants independent variables
    for g in grp_comb:                  # assign vectors to y
        dfg     = df[ df[ group ] == g ]
        for lab, lv in zip( variants, likert_var ):
            labels.append( g + '-' + lab )
            ym  = []
            for m in main_comb:
                dframe  = dfg[ dfg[ main_var ] == m ]
                lkm     = np.array( dframe[ lv ].mean() )
                lks     = np.array( dframe[ lv ].std() ) / 2
                lkm     /= lkm.sum()     # one vector of Likert scale mean
                ym.append( ( lkm, lks ) )
            ym  = np.array( ym )        # Likert scale vectors for all values for main_var
            ym  = ym.T                  # arrange ym with shape n_likert X n_main
            y.append( ym )


    # arrange colors with n_likert rows and n_main columns
    colors  = gen_colors( n_main, n_likert )

    pyplot.rcParams.update( { "font.size": 14 } )
    # note that the legend is just the names of the main_var values
    # and as representative color the second brighter is selected
    handles     = [ Patch( facecolor=c, label=l ) for c,l in zip( colors[ -2 ], main_comb ) ]

    fig, ax     = pyplot.subplots( figsize=figsize )

    xticks      = []
    for i, ym in enumerate( y ):
        x0      = 1 + i * ( 1 + n_main )
        x1      = x0 + n_main
        xticks.append( ( x0 + x1 ) / 2 )
        x       = np.arange( x0, x1 )
        y0      = np.zeros( n_main )
        # to stack bars just assign the bottom value as the previous top
        # note that one ax.bar call produces n_main bar segmets
        for j, data in enumerate( ym ):
            y1, s   = data
            ax.bar( x, y1, yerr=s, capsize=3, width=width, color=colors[ j ], bottom=y0 )
            y0  += y1

    ax.set_xticks( xticks )
    ax.set_xticklabels( labels )

    ax.set_ylabel( "Fraction" )

    # compute an estimate of the max occupancy of labels in the plot
    lab_len     = char_len * max( [ len( l ) for l in main_comb ] )
    # compute the current geometry of the plotting box
    box = ax.get_position()
    # and now make room on the right for the labels
    ax.set_position([box.x0, box.y0, box.width * ( 1 - lab_len ), box.height])
    ax.legend( handles=handles, loc="center", bbox_to_anchor=(1+lab_len, 0.5), labelspacing=labelspacing )

    fname       = f"{fname}{extension}"
    if len( suptitle ):
        pyplot.suptitle( suptitle, x=0.4 )
    pyplot.savefig( fname, bbox_inches="tight", pad_inches=0 )
    pyplot.close()
    pyplot.clf()


def plot_radar( df, main_var, group="value", scores=["yes_img", "yes_txt"], fname="radar", suptitle="", t_angle=90 ):
    """
    Generate a radar plot
    data should be boolean

    params:
        df          [pandas.core.frame.DataFrame] the data in pandas DataFrame
        main_var    [str] name of the main column of independent variables for which bars are aligned
        group       [str] one column of independent variables or None
        scores      [list] of dependent variables
        fname       [str] name of the output file
        suptitle    [str] plot title
    """
    label_offset    = 1.20  # Adjust to increase/decrease distance
    columns         = df.columns
    assert main_var in columns, f"there is no column named {main_var}"

    ticks       = [ 0.2, 0.4, 0.6, 0.8, 1.0 ]

    # extract the categories to plot
    cat         = df[ main_var ].unique().tolist()
    n_cat       = len( cat )
    assert n_cat <= 10, f"cannot do radar plot for {n_cat} categories"

    poly_name   = []
    poly_data   = []

    # extract the labels of the group to plot
    if group is not None:
        assert group in columns, f"there is no column named {group}"
        values      = df[ group ].unique().tolist()
        for v in values:
            dv      = df[ df[ group ] == v ]
            for s in scores:
                poly_name.append( f"{v}-{s}" )
                poly_data.append( dv[ s ].tolist() )
    else:
        for s in scores:
            poly_name.append( s )
            poly_data.append( df[ s ].tolist() )

    # compute angle for each axis
    angles = np.linspace( 0, 2 * np.pi, n_cat, endpoint=False ).tolist()

    # close the polygons
    angles  += angles[ :1 ]
    for p in poly_data:
        p   += p[ :1 ]

    pyplot.rcParams.update( { "font.size": 14 } )

    fig, ax     = pyplot.subplots( figsize=radar_fsize, subplot_kw=dict( polar=True ) )

    for i, p in enumerate( poly_data ):
        l       = poly_name[ i ]
        ax.plot( angles, p, label=l, color=tab_colors[ i ], linewidth=2, linestyle=line_style[ i ] )
        ax.fill( angles, p, color=tab_colors[ i ], alpha=0.25 )

    ax.set_xticks( angles[:-1] )
    # ax.set_xticklabels( cat )
    ax.set_rgrids( ticks, angle=t_angle, fontsize=12 )
    ax.set_ylim(0, 1)

    # Hide default labels
    ax.set_xticklabels([])
    # Add custom labels with increased distance
    for angle, label in zip(angles[:-1], cat):
        ax.text(angle, label_offset, label,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=12)

    ax.yaxis.grid(True, color='gray', linestyle='dotted', linewidth=0.5)
    ax.xaxis.grid(True)

    ax.legend( loc='upper right', bbox_to_anchor=(1.2, 1.1) )
    fname       = f"{fname}{extension}"
    if len( suptitle ):
        pyplot.suptitle( suptitle, x=0.4 )
    pyplot.savefig( fname, bbox_inches="tight", pad_inches=0 )
    pyplot.close()
    pyplot.clf()


def plot_values( df, groups=["value","age"], values=["yes_img", "yes_txt"], fname="plot", suptitle='', xlabels=None, width=0.9 ):
    """
    Generate one plot of the specified values, for a specified group of independent variables
    in the list of groups, the last is used as main group, which components in the dataset are used as group label
    the first group shouls be the lowest loob, in combination with values
    WARNING: never tested for more than two groups...

    params:
        df          [pandas.core.frame.DataFrame] the data in pandas DataFrame
        groups      [list] columns of independent variables
        values      [list] the numerical values to be plot
        fname       [str] name of the output file
        suptitle    [str] plot title
        xlabels     [list] labels for the X axis, is None the components of the last group are assumed
    """

    columns = df.columns
    y       = []                        # list of all vectors to plot
    labels  = []                        # labels for all combinations of independent variables
    cat     = dict()                    # all categorical entries found for independent variables
    comb    = dict()                    # vectors for all combinations of independent variables
    for g in groups:                    # gather the categorical entries for independent variables
        assert g in columns, f"there is no column named {g}"
        cat[ g ]    = df[ g ].unique().tolist()

    for g in groups:                    # collect all vectors for combinations of independent variables
        if not len( comb ):
            first           = True
        else:
            first           = False
            new_comb        = dict()
        for c in cat[ g ]:
            if first:
                comb[ c ]   = df[ df[ g ] == c ]
            else:
                for prev in comb.keys():
                    dframe  = comb[ prev ]
                    new     = prev + '-' + c
                    new_comb[ new ] = dframe[ dframe[ g ] == c ]
        if not first:
            comb            = new_comb

    for c in comb.keys():               # assign vectors to y
        dframe  = comb[ c ]
        for v in values:
            y.append( dframe [ v ] )

    for c in cat[ groups[ 0 ] ]:        # assign key combinations of group 0 and values to labels
        for v in values:
            v1, v2  = v.split( '_' )
            v1      = f"Ls{v1[2]}" if v1.startswith( "lk" ) else v1.upper()
            v2      = "Im" if v2 == "img" else "Tx"
            v3      = "T" if c == "true" else "F"
            labels.append( f"{v1} {v2} {v3}" )

    if xlabels is None:                 # if no xlabels is specified, use the names found in the last group
        xlabels = cat[ groups[ -1 ] ]

    n           = len( y )                                      # total numer of bars
    nv          = len( values )                                 # number of values to plot
    ng          = len( cat[ groups[ 0 ] ] ) * nv                # number of bars for each component of the main group
    nc          = nv // len( cat[ groups[ 0 ] ] )               # number of bars for each subgroup
    # WARNING: the following computation is not convincing, seemingly it works by coincidence because
    # len( cat[ groups[ 0 ] ] ) is 2, but not in the general case
    ns          = ng // nc                                      # number of subgroup

    # compute a proper separation of bars by group
    # assume that in case of a large number of values, if they are even, their likely belong to two contiguous sets
    # so, add a separation between the two
    # NOTE: this strategy avoids an additional function parameter, but may not work for future cases
    if ( nv > 4 ) and ( not nv % 2 ):
        nv2     = nv // 2
        x           = [  i + i // ng + i // nv2 for i in range( n ) ]
    else:
        x           = [  i + i // ng for i in range( n ) ]
    xl          = x[ (ng//2)::ng ]      # X positions for the main group labels

    colors      = []
    hatches     = []
    for i in range( ns ):
        for j in range( nc ):
            colors.append( dim_colors[ i ][ j ] )
            hatches.append( "/" if j==0 else None )     # THIS WORKS ONLY IF UNK IS FIRST OF LIST

    pyplot.rcParams.update( { "font.size": 14 } )
    handles     = [ Patch( facecolor=c, label=l, hatch=h, edgecolor='white' ) for c,l,h in zip( colors, labels, hatches ) ]
    # handles     = [ Patch( facecolor=c, label=l ) for c,l in zip( colors, labels ) ]

    fig, ax     = pyplot.subplots( figsize=figsize )

    for i, ( xx, data ) in enumerate( zip( x, y ) ):
        m   = data.mean()
        s   = data.std() / 2.
        # bar with means as heigth, and a segment for standard deviation
        ax.bar( xx, m, yerr=s, color=colors[ i % ng ], hatch=hatches[ i % ng ], width=width, linewidth=2, edgecolor='white' )

    ax.set_xticks( xl, labels=xlabels, fontsize=15 )
    ax.set_ylabel( "YES fraction" )
    ax.set_ylim( bottom=0.0, top=1.0 )

    # compute an estimate of the max occupancy of labels in the plot
    lab_len     = char_len * max( [ len( l ) for l in labels ] )
    # compute the current geometry of the plotting box
    box = ax.get_position()
    # and now make room on the right for the labels
    ax.set_position([box.x0, box.y0, box.width * ( 1 - lab_len ), box.height])
    ax.legend( handles=handles, loc="center", bbox_to_anchor=(1+lab_len, 0.5), labelspacing=labelspacing )

    fname       = fname + extension
    pyplot.suptitle( suptitle, x=0.4 )
    pyplot.savefig( fname, bbox_inches="tight", pad_inches=0 )
    pyplot.close()
    pyplot.clf()


def plot_models( df, groups=["value","age"], values=["yes_img", "yes_txt"], fname="pl" ):
    """
    Wrapper for executing plot_values() on all models found in the dataset, together with a final
    plot with all models data

    params:
        df          [pandas.core.frame.DataFrame] the data in pandas DataFrame
        groups      [list] columns of independent variables
        values      [list] the numerical values to be plot
        fname       [str] name of the output file
    """
    models      = df[ "model" ].unique().tolist()    # find all models used in the dataframe

    g           = groups[ -1 ]
    for m in models:
        name    = f"{fname}_{m}_{g}"
        plot_values( df[ df[ "model" ] == m ], groups=groups, values=values, fname=name, suptitle=f"model {m}" )
    name    = f"{fname}_all_{g}"
    if len( models ) > 1:
        plot_values( df, groups=groups, values=values, fname=name, suptitle="all models" )


def plot_models_single( df, main_var, group="value", values=["yes_img", "yes_txt"], fname="pl", width=0.9 ):
    """
    Wrapper for executing plot_single() on all models found in the dataset, together with a final
    plot with all models data, for all categories in the specified main independent variable

    params:
        df          [pandas.core.frame.DataFrame] the data in pandas DataFrame
        main_var    [str] name of the main column of independent variables for which single plots are produced
        group       [str] column of independent variables
        values      [list] the numerical values to be plot
        fname       [str] name of the output file
        width       [float] bar width
    """
    models      = df[ "model" ].unique().tolist()    # find all models used in the dataframe
    categories  = df[ main_var ].unique().tolist()   # find all categories fot the specified main independent variable


    for m in models:
        dm      = df[ df[ "model" ] == m ]
        for c in categories:
            name    = f"{fname}_{m}"
            plot_single( dm[ dm[ main_var ] == c ], c, group=group, values=values, fname=name, width=width )

    if len( models ) > 1:
        name    = f"{fname}_all"
        for c in categories:
            name    = f"{fname}_all"
            plot_single( df[ df[ main_var ] == c ], c, group=group, values=values, fname=name, width=width )


def plot_models_likert( df, main_var, group="value", fname="pl", width=0.9 ):
    """
    Wrapper for executing plot_likert() on all models found in the dataset, together with a final
    plot with all models data, for all categories in the specified main independent variable

    params:
        df          [pandas.core.frame.DataFrame] the data in pandas DataFrame
        main_var    [str] name of the main column of independent variables for which single plots are produced
        group       [str] column of independent variables
        fname       [str] name of the output file
        width       [float] bar width
    """
    models      = df[ "model" ].unique().tolist()    # find all models used in the dataframe

    for m in models:
        dm      = df[ df[ "model" ] == m ]
        name    = f"{fname}_likert_{m}"
        plot_likert( dm, main_var, group=group, fname=name, width=width, suptitle=f"model {m}" )

    if len( models ) > 1:
        name    = f"{fname}_likert_all"
        plot_likert( df, main_var, group=group, fname=name, width=width, suptitle="all models" )


def plot_models_radar( df, mdf, main_var, group="value", scores=["yes_img", "yes_txt"], fname="pl", t_angle=100 ):
    """
    Wrapper for executing plot_radar() on all models found in the dataset, together with a final
    plot with all models data, for all categories in the specified main independent variable

    params:
        df          [pandas.core.frame.DataFrame] the data in pandas DataFrame for all models
        mdf         [pandas.core.frame.DataFrame] the data in pandas separated by models
        main_var    [str] name of the main column of independent variables for which single plots are produced
        group       [str] column of independent variables
        fname       [str] name of the output file
        t_angle     [float] angle at which ticks are drawn
    """
    models      = mdf[ "model" ].unique().tolist()    # find all models used in the dataframe

    for m in models:
        dm      = mdf[ mdf[ "model" ] == m ]
        name    = f"{fname}_radar_{m}"
        plot_radar( dm, main_var, group=group, scores=scores, fname=name, t_angle=t_angle, suptitle=f"model {m}" )

    if len( models ) > 1:
        name    = f"{fname}_radar_all"
        plot_radar( df, main_var, group=group, scores=scores, fname=name, t_angle=t_angle, suptitle="all models" )
