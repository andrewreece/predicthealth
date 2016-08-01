
ratings_means <- c('likable','interesting','sad','happy')
hsv_means <- c('hue','saturation','brightness')
hsv_full <- c(hsv_means,'before_diag','before_susp','target')
metadata_means <- c('comment_count','like_count','has_filter')
face_means <- c('has_face','face_ct')

# a number of LIWC labels are linearly dependent on their hierarchical sub-categories. just use these ones.
liwc_vars <- paste0('LIWC_',c('i','we','you','shehe','they','ipron','article','verb','negate',
                              'swear','social','posemo','anx','anger','sad','cogmech','percept',
                              'body','health','sexual','ingest','relativ','work','achieve',
                              'leisure','home','money','relig','death','assent'))
vars <- list(
  ig=list(
    post=list( 
      ratings_means=ratings_means,
      hsv_means=hsv_means,
      hsv_full=c(hsv_means,'before_diag','before_susp','target'),
      metadata_means=metadata_means,
      face_means=face_means,
      all_ig_means=c(hsv_means,metadata_means),
      ig_face_means=c(hsv_means, metadata_means, face_means),
      all_means=c(ratings_means,hsv_means,metadata_means,face_means),
      full=c(ratings_means, hsv_means,metadata_means,
             'likable.var','interesting.var','sad.var','happy.var',
             'before_diag','before_susp','target')
    ),
    username=list(
      ratings_means=ratings_means,
      hsv_means=hsv_means,
      hsv_full=c(hsv_means,'before_diag','before_susp','target'),
      metadata_means=c(metadata_means,'url'),
      face_means=face_means,
      all_ig_means=c(hsv_means,metadata_means,'url'),
      ig_face_means=c(hsv_means, metadata_means, 'url', face_means),
      all_means=c(ratings_means,hsv_means,metadata_means,'url',face_means),
      full=c(ratings_means, hsv_means,metadata_means,'url', face_means,
             'likable.var','interesting.var','sad.var','happy.var',
             'one_word','description','target')
    ),
    created_date=list(
      ratings_means=ratings_means,
      hsv_means=hsv_means,
      hsv_full=c(hsv_means,'before_diag','before_susp','target'),
      metadata_means=c(metadata_means,'url'),
      face_means=face_means,
      all_ig_means=c(hsv_means,metadata_means,'url'),
      ig_face_means=c(hsv_means, metadata_means, 'url', face_means),
      all_means=c(ratings_means,hsv_means,metadata_means,'url',face_means),
      full=c(ratings_means, hsv_means,metadata_means,'url', face_means,
             'likable.var','interesting.var','sad.var','happy.var',
             'one_word','description','target')
    )
  ),
  tw=list(
    weekly=list( 
      all_means=c('LabMT_happs', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
              'tweet_count', 'word_count', 'has_url', 'is_rt', 'is_reply', 'LIWC_happs', liwc_vars),
      basic_tweet_means=c('LabMT_happs','tweet_count','word_count','is_rt','is_reply','has_url'),
      test_means=c('LabMT_happs', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
                   'tweet_count', 'word_count', 'has_url', 'is_rt', 'is_reply', 'LIWC_happs', liwc_vars)
    ),
    user_id=list(
      all_means=c('LabMT_happs', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
                  'tweet_count', 'word_count', 'has_url', 'is_rt', 'is_reply', 'LIWC_happs', liwc_vars),
      basic_tweet_means=c('LabMT_happs','tweet_count','word_count')
      ),
      created_date=list(
        all_means=c('LabMT_happs', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
                    'tweet_count', 'word_count', 'has_url', 'is_rt', 'is_reply', 'LIWC_happs', liwc_vars),
        basic_tweet_means=c('LabMT_happs','tweet_count','word_count','is_rt','is_reply','has_url'),
        test_means=c('LabMT_happs', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
                     'tweet_count', 'word_count', 'has_url', 'is_rt', 'is_reply', 'LIWC_happs', liwc_vars)
      )
    )
  )

saveRDS(vars,'varset.rds')
