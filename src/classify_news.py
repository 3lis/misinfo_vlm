"""
#####################################################################################################################

    Module to extract news tags

#####################################################################################################################
"""

import      requests
import      json
import      re
import      copy
from        bs4             import  BeautifulSoup
from        collections     import  Counter
import      classify_img    as      cim


FJSON_O     = "../data/news_200.json"

IMG_PATH    = "../imgs"
# IMG_PATH    = "../imgs_all/imgs_alice"


def load_json( f ):
    """
    Load json file
    """
    with open( f, "r", encoding="utf-8" ) as file:
        return json.load( file )


def get_tags( url ):
    """
    Scrape the tags from a given Politifact news URL
    """
    try:
        response    = requests.get( url, timeout=10 )
        response.raise_for_status()  # Raise an error for bad responses
        soup        = BeautifulSoup( response.text, "html.parser" )

        tags_text   = []
        tags_pers   = []

        # Find all elements with class "m-list__item" and extract tag names
        for tag in soup.select( "li.m-list__item a.c-tag" ):
            tag_text    = tag.text.strip()
            tag_href    = tag.get( "href", "" )

            if tag_href.startswith( "/personalities/" ):
                tags_pers.append( tag_text )
            else:
                tags_text.append( tag_text )
        return tags_text, tags_pers

    except requests.RequestException as e:
        return f"ERROR: {e}"


def get_all_tags( dset ):
    """
    Scrape the tags from all news in dataset
    """
    d_tags      = {}    # tags related to topics
    d_tags_p    = {}    # tags related to people (not uset)
    d_tags_all  = {}    # tags of both types
    cnt     = 0
    tot     = len( dset )

    for n in dset:
        print( f"Processing tag of news {cnt+1} out of {tot}" )
        cnt += 1

        url                     = n[ "url" ]
        tags, tags_p            = get_tags( url )
        d_tags[ n[ "id" ] ]     = tags
        d_tags_p[ n[ "id" ] ]   = tags_p
        d_tags_all[ n[ "id" ] ] = tags + tags_p

    return d_tags, d_tags_p, d_tags_all


def count_tags( d_tags ):
    """
    Count occurrences of tags in news
    """
    l           = []
    all_tags    = [ tag for tags in d_tags.values() for tag in tags ]
    tag_counts  = Counter( all_tags )
    for tag, count in tag_counts.most_common():
        l.append( ( tag, count ) )
    return l


def extract_tags():
    """
    Extract all tags and print them
    """
    d_tags, d_tags_p, d_tags_all    = get_all_tags( load_json( FJSON_o ) )
    ct  = count_tags( d_tags )
    cp  = count_tags( d_tags_p )

    # all tags associated per news
    with open( "tag_news.txt", "w", encoding="utf-8" ) as f:
        for k, v in d_tags_all.items():
            f.write( f"{k.ljust( 10 ) } {v}\n")

    # all topic tags with count
    with open( "tags_t.txt", "w", encoding="utf-8" ) as f:
        for i in ct:
            f.write( f"{i[ 0 ].ljust( 40 ) } {i[ 1 ]}\n")

    # all people tags with count
    with open( "tags_p.txt", "w", encoding="utf-8" ) as f:
        for i in cp:
            f.write( f"{i[ 0 ].ljust( 40 ) } {i[ 1 ]}\n")


def aggregate_tags( d_tags ):
    """
    Aggregate tags into categories.
    Return dict of tags per news with new categories.
    """
    d_aggr      = { k: set() for k in d_tags }
    categ       = {
            "to_delete": [ # Tags to remove
                "pundits",
                "Fake News",
                "Facebook Fact-checks",
                "PunditFact",
                "Transparency",
                "Ad Watch",
                "Negative Campaigning",
                "Ask PolitiFact",
                "Message Machine 2014",
                "Good Enough to Be True",
                "Wisconsin",
                "Texas",
                "California",
                "New York",
                "Georgia",
                "Florida",
                "State Governments",
                "City Governments",
                "County Governments",
                "North Carolina",
                "West Virginia",
                "Rhode Island",
                "Illinois",
                "Pennsylvania",
                "Arizona",
                "Michigan",
                "Virginia",
                "New Hampshire",
                "Ohio",
                "Vermont",
                "New Jersey",
                "the 2018 california governor's race",
                "Colorado",
                "Missouri",
                "this week - abc news",
                "Tennessee",
                "Nevada",
                "National",
                "States",
                "corrections and updates",
                "obama birth certificate",
                "global news service",
                "occupy wall street",
            ],
            "politics": [ # Politics & Elections
                "Elections",
                "Congress",
                "Voting Record",
                "Campaign Finance",
                "Political Parties",
                "Redistricting",
                "Supreme Court",
                "Impeachment",
                "Constitutional Amendments",
                "Candidate Biography",
                "Polls and Public Opinion",
                "2024 Senate Elections",
                "doge",
                "Debates",
                "jan. 6",
                "regulation",
                "city government",
                "government regulation",
                "county government",
                "bush administration",
                "Bipartisanship",
                "Party Support"
            ],
            "health": [ # Health & Public Safety
                "Coronavirus",
                "Health Care",
                "Public Health",
                "Drugs",
                "Marijuana",
                "Medicaid",
                "Medicare",
                "Food Safety",
                "Abortion",
                "alcohol",
                "Disability",
                "Ebola",
                "Public Safety",
                "Health Check"
            ],
            "economy": [ # Economy & Jobs
                "Economy",
                "Jobs",
                "Federal Budget",
                "State Budget",
                "Taxes",
                "Trade",
                "Small Business",
                "social security",
                "county budget",
                "Debt",
                "Deficit",
                "transportation",
                "Infrastructure",
                "Income",
                "Workers",
                "Labor",
                "city budget",
                "tourism",
                "housing",
                "corporations",
                "welfare",
                "gas prices",
                "Unions",
                "Pensions",
                "Financial Regulation"
            ],
            "law": [ # Law & Justice
                "Crime",
                "Criminal Justice",
                "Legal Issues",
                "Immigration",
                "Border Security",
                "Supreme Court",
                "Death Penalty",
                "gambling",
                "human rights",
                "Civil Rights",
                "Ethics",
                "guns",
                "Homeland Security",
                "census",
                "voter id laws",
                "Voting ID Laws"
            ],
            "environment": [ # Science & Environment
                "Climate Change",
                "Energy",
                "Environment",
                "Natural Disasters",
                "Water",
                "Oil Spill",
                "Space",
                "fires",
                "weather",
                "Agriculture",
                "Science"
            ],
            "foreign": [ # Foreign Affairs & Security
                "Foreign Policy",
                "Military",
                "Terrorism",
                "Homeland Security",
                "Russia",
                "China",
                "Ukraine",
                "Iran",
                "Afghanistan",
                "Iraq",
                "Israel",
                "Nuclear",
                "International Relations"
            ],
            "society": [ # Society & Culture
                "Race and Ethnicity",
                "Education",
                "LGBTQ",
                "Women",
                "Religion",
                "Marriage",
                "Sexuality",
                "public service",
                "islam",
                "baseball",
                "Patriotism",
                "Pop Culture",
                "Sports",
                "Families",
                "homeless",
                "veterans",
                "animals",
                "Population",
                "urban",
                "retirement",
                "Wealth",
                "Poverty",
                "Recreation",
                "children",
                "history",
                "Food"
            ],
            "technology": [ # Technology & Innovation
                "Technology",
                "Artificial Intelligence",
                "Social Media",
                "Consumer Safety",
                "Privacy Issues",
                "Transparency",
                "Space"
            ]
    }

    # reverse mapping: tag -> category (excluding "to_delete")
    tag_to_categ = {
        t.lower(): c
        for c, tags in categ.items()
        for t in tags
    }

    for k, tags in d_tags.items():
        for t in tags:
            t   = t.lower()
            if t in tag_to_categ:
                if tag_to_categ[ t ] != "to_delete":
                    d_aggr[ k ].add( tag_to_categ[ t ] )
            else:
                print( f"WARNING: tag {t} not found in manual categories" )

    c = 0
    for k in d_aggr:
        if d_aggr[ k ] == set(): c += 1
    if c:
        print( f"WARNING: {c} news out of {len(d_aggr)} don't have tags")

    return d_aggr


def write_json( fo, fn, d_tags, d_aggr, d_imgs, check_year=False ):
    """
    Update json file with tags of news and images

    There's an option to generate a small stat about the years of publication of news

    params:
        fo      [str] filename of original json with news
        fn      [str] filename of modified json with news
        d_tags  [dict] key: news id, value: list of original tags
        d_aggr  [dict] key: news id, value: list of aggregated tags
        d_imgs  [dict] key: news id
                       value: [bool] true if people occupy the majority of the image
                              [float] coverage of people in image
        check_year [bool] optional check
    """
    news_o      = load_json( fo )
    news_n      = []
    years       = {}

    for news in news_o:
        n                   = copy.deepcopy( news )
        n[ "tags_orig" ]    = d_tags[ n[ "id" ] ]
        n[ "tags" ]         = list( d_aggr[ n[ "id" ] ] )

        if d_imgs[ n[ "id" ] ][ 0 ]:
            n[ "tags_img" ] = [ "people" ]
        else:
            n[ "tags_img" ] = [ "no_people" ]

        news_n.append( n )

        if check_year:
            y   = re.search( r'\b\d{4}\b', n[ "more" ] ).group()
            if y in years:  years[ y ] += 1
            else:           years[ y ] = 1

    with open( fn, "w") as f:
        json.dump( news_n, f, indent=4 )

    return news_n, years


def stat_news( fjson ):
    """
    Print a stat of the news across true/false, tags, and tags_imgs
    """
    data            = load_json( fjson )
    tag_counts      = {}
    tag_img_counts  = {}
    tf_counts       = { 'true': 0, 'false': 0, 'all': 0 }

    # Iterate over the data to count the tags and tags_img
    for entry in data:
        tf_counts[ 'all' ]          += 1
        if entry[ 'true' ] == 1:
            tf_counts[ 'true' ]     += 1
        else:
            tf_counts[ 'false' ]    += 1

        # Count the regular tags
        for tag in entry[ 'tags' ]:
            if tag not in tag_counts:
                tag_counts[ tag ]   = { 'true': 0, 'false': 0, 'all': 0 }

            # Increment total count for the tag
            tag_counts[ tag ][ 'all' ]      += 1

            # Increment 'true' or 'false' based on the value of 'true' in the entry
            if entry[ 'true' ] == 1:
                tag_counts[ tag ][ 'true' ]     += 1
            else:
                tag_counts[ tag ][ 'false' ]    += 1

        # Count the tags_img
        tag = entry[ 'tags_img' ][ 0 ]
        print( entry[ 'tags_img' ] )
        if tag not in tag_img_counts:
            tag_img_counts[ tag ]   = { 'true': 0, 'false': 0, 'all': 0 }

        # Increment total count for the tag
        tag_img_counts[ tag ][ 'all' ]  += 1

        # Increment 'true' or 'false' based on the value of 'true' in the entry
        if entry['true'] == 1:
            tag_img_counts[ tag ][ 'true' ]     += 1
        else:
            tag_img_counts[ tag ][ 'false' ]    += 1

    # Print the table headers for tags
    print("tag".ljust(15), "true".ljust(6), "false".ljust(6), "all")
    print("-" * 40)

    # Loop through each tag and print the counts
    for tag, counts in tag_counts.items():
        print(f"{tag.ljust(15)} {str(counts['true']).ljust(6)} {str(counts['false']).ljust(6)} {str(counts['all']).ljust(6)}")
    print("-" * 40)

    # Loop through each tag_img and print the counts
    for tag, counts in tag_img_counts.items():
        print(f"{tag.ljust(15)} {str(counts['true']).ljust(6)} {str(counts['false']).ljust(6)} {str(counts['all']).ljust(6)}")

    print("-" * 40)
    print(f"{'total'.ljust(15)} {str(tf_counts['true']).ljust(6)} {str(tf_counts['false']).ljust(6)} {str(tf_counts['all']).ljust(6)}")


#####################################################################################################################

if __name__ == '__main__':
    # d_tags, _, _    = get_all_tags( load_json( FJSON_O ) )
    # d_aggr          = aggregate_tags( d_tags )
    # d_imgs          = cim.check_person_imgs( IMG_PATH )
    # news_n, years   = write_json( FJSON_O, FJSON_N, d_tags, d_aggr, d_imgs, check_year=True )
    #
    # for y in sorted( years, reverse=True ):
    #     print( f"{y}\t{years[ y ]}" )

    stat_news( FJSON_O )
