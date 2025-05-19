kwargs      = {
    'model_id':             7,                      # gpt-4o-mini
    'experiment':           "both",                 # use image+text and text-only news
    'f_news':               "news_200.json",        # text content of dataset
    'f_demo':               "demo_small.json",      # demographic attributes
    'f_dialog':             "dialogs_user.json",    # prompt templates in the 3rd person

    'dialogs_pre':          [                       # specify the content of the prompt preceding the news item
            "intro_profile",                        # introduce the persona profile
            "p_agre",                               # agreeableness personality trait profile
            "context"                               # context of the task
    ],

    'dialogs_post':         [                       # specify the content of the prompt following the news item
            "reason_3steps",                        # prompt the model to use chain-of-though
            "reason_share_likert5",                 # prompt the model to rate using 5-point likert scale
            "ask_user_dsc"                          # specify the task in the 3rd person
    ],

    'info_source':          True,                   # include info about the source of the news
    'info_more':            True,                   # include info about the news date

    # 'news_amount':          2,                    # optional, limit the number of news to process

    'n_returns':            10,                     # number of completions to generate
    'max_tokens':           1200,                   # max number of tokens in completions
    'agreement':            True,                   # compute the agreement of likert ratings
    'temperature':          0.9,                    # sampling randomness in completions
    'top_p':                1.0,                    # nucleus sampling in completions
}
