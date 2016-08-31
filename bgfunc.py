import sqlite3, itertools, datetime, pytz, cv2, os, sys, re, pickle, urllib, filecmp
import pandas as pd
import numpy as np
from re import findall, UNICODE
from email.utils import parsedate_tz, mktime_tz
from os import listdir
from os.path import isfile, join
from string import Template

from IPython.display import Image, display
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import seaborn as sns
sns.set_style('white')

from skimage import io, data, color

from sklearn import cross_validation
from sklearn import mixture
from sklearn.preprocessing import scale
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression as logreg
from sklearn.linear_model import LogisticRegressionCV as logregcv
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn import mixture
from pykalman import KalmanFilter
from hmmlearn.hmm import GaussianHMM

import scipy.stats as stats
from scipy import linalg
from scipy.stats import ttest_ind as ttest
from scipy.stats import ttest_rel
from scipy.stats import pearsonr
from scipy import interp

import statsmodels.api as sm
from statsmodels.sandbox.stats.multicomp import multipletests
import statsmodels.api as sm
import statsmodels.tools.tools as smtools

import warnings
warnings.simplefilter("ignore")

from labMTsimple.speedy import *


def analysis_specifications(pl, condition):
	''' Defines basic specs for analysis, based on platform and target condition.
		These specs are used to set larger params dict in define_params(). '''

	specs = {}

	p_vars = ['gb_types','plong','fields']
	p_opts = {
		'gb_types':{
			'ig':['post','created_date','username'],
			'tw':['created_date','weekly','user_id']
		},
		'plong':{
			'ig':'instagram',
			'tw':'twitter'
		},
		'fields':{
			'ig':'url, comment_count, like_count, filter, has_face, face_ct',
			'tw':'id, text, has_url'
		}
	}

	for v in p_vars:
		specs[v] = p_opts[v]

	c_vars = ['has_test','test_name','test_cutoff','photos_rated']
	c_opts = {
		'has_test':{
			'depression':True,
			'ptsd':True,
			'pregnancy':False,
			'cancer':False
		},
		'test_name':{
			'depression':'cesd',
			'ptsd':'tsq',
			'pregnancy':None,
			'cancer':None
		},
		'test_cutoff':{
			'depression':21,
			'ptsd':5,
			'pregnancy':None,
			'cancer':None
		},
		'photos_rated':{
			'depression':True,
			'ptsd':True,
			'pregnancy':True,
			'cancer':False
		}
	}
	
	for v in c_vars:
		specs[v] = c_opts[v]

	return specs


def define_params(condition, test_name, test_cutoff, impose_cutoff,
				  platform, platform_long, fields,
				  photos_rated, has_test, additional_data, ratings_min=3):
	''' Creates params dict for queries and other foundational parameters used throughout analysis '''

	if photos_rated and (platform=='ig') and additional_data:
		ratings_clause = ' and ratings_ct >= {}'.format(ratings_min)
	else:
		ratings_clause = ""

	if has_test and impose_cutoff:
		cutoff_clause = " and {name} > {cutoff}".format(name=test_name, cutoff=test_cutoff)
	else:
		cutoff_clause = ""

	# a number of LIWC labels are linearly dependent on their hierarchical sub-categories. just use these ones.
	LIWC_vars = ['LIWC_{}'.format(v) for v in ['i','we','you','shehe','they','ipron','article','verb','negate',
											   'swear','social','posemo','anx','anger','sad','cogmech','percept',
											   'body','health','sexual','ingest','relativ','work','achieve',
											   'leisure','home','money','relig','death','assent']]
	if condition in ['ptsd','pregnancy']:
		extra_fields = ', d_from_event_{cond} as from_event, event_date_{cond} as event_date'.format(cond=condition)
	else:
		extra_fields = ''

	params = {
			'q': {
				't': {
					'meta':	'select {fields}{extra_fields}, username, {plat_long}_user_id as user_id, created_date, diag_date_{cond} as diag_date, d_from_diag_{cond} as from_diag, d_from_susp_{cond} as from_susp from meta_{plat} where created_date is not null and created_date is not "" and d_from_diag_{cond} is not null{ratings_clause}'.format(cond=condition,
																																																											  ratings_clause=ratings_clause,
																																																											  plat=platform,
																																																											  plat_long=platform_long,
																																																											  fields=fields,
																																																											  extra_fields=extra_fields),
					'photo_ratings':'select url, rater_id, happy, sad, interesting, likable, one_word, description from photo_ratings_{cond}_target'.format(cond=condition),
					'before_diag':	'select {fields}, username, {plat_long}_user_id as user_id, created_date, diag_date_{cond} as diag_date from meta_{plat} where d_from_diag_{cond} < 0{ratings_clause}'.format(cond=condition,
																																										plat=platform,
																																										fields=fields,
																																										plat_long=platform_long,
																																										ratings_clause=ratings_clause),
					'after_diag':	'select {fields}, username, {plat_long}_user_id as user_id, created_date, diag_date_{cond} as diag_date from meta_{plat} where d_from_diag_{cond} >= 0{ratings_clause}'.format(cond=condition,
																																										 fields=fields,
																																										 plat=platform,
																																										 plat_long=platform_long,
																																										 ratings_clause=ratings_clause),
					'unames':		'select username from {cond} where platform="{plat_long}" and username is not null and disqualified=0{cutoff_clause}'.format(cond=condition, 
																																								 plat_long=platform_long,
																																								 cutoff_clause=cutoff_clause)
				},
				'c':{
					'photo_ratings':'select url, rater_id, happy, sad, interesting, likable, one_word, description from photo_ratings_{cond}_control'.format(cond=condition),
					'unames':		'select username from control where platform="{plat_long}" and username is not null and {cond}="No" and disqualified=0'.format(plat_long=platform_long,
																																								   cond=condition)
				},
				'all_meta':	'select {fields}{extra_fields}, username, {plat_long}_user_id as user_id, created_date, diag_date_{cond} as diag_date, d_from_diag_{cond} as from_diag, d_from_susp_{cond} as from_susp from meta_{plat}'.format(cond=condition,
																																															 plat=platform,
																																															 plat_long=platform_long,
																																															 fields=fields,
																																															 extra_fields=extra_fields),
				'all_hsv':	'select url, hue, saturation, brightness, username from hsv'
			},
			'agg_func':{
				'ig': {
					'post':{
						'url': 			'first',
						'username':		'first',
						'created_date':	'first',
						'hue':			'first',
						'saturation':	'first',
						'brightness':	'first',
						'before_diag':	'first',
						'before_susp':	'first',
						'comment_count':'first',
						'like_count':	'first',
						'has_filter':	'first',
						'has_face':		'first',
						'face_ct':		'first'
					},
					'username':{'url':				'mean',
								'hue':				'mean',
								'saturation':		'mean',
								'brightness':		'mean',
								'comment_count': 	'mean',
								'like_count':		'mean',
								'has_filter':		'mean',
								'has_face':			'mean',
								'face_ct':			'mean'
					},
					'created_date':{'url':			'count',
									'username':		'first',
									'hue':			'mean',
									'saturation':	'mean',
									'brightness':	'mean',
									'before_diag':	'first',
									'before_susp':	'first',
									'comment_count':'mean',
									'like_count':	'mean',
									'has_filter':	'sum',
									'has_face':		'sum',
									'face_ct':		'mean'
					}
				},
				'tw':{
					'created_date':{'id':			'count',
									'target':		'first',
									'has_url':		'mean',
									'is_rt':		'mean',
									'is_reply':		'mean',
									'from_diag':	'first',
									'from_susp':	'first',
									'text':			' '.join,
									'diag_date':	'first',
									'user_id':		'first',
									'word_count':	'mean',
									'before_diag':	'first',
									'before_susp':	'first'
					},
					'user_id':{	'id':		'count',
								'target':	'first',
								'has_url':	'mean',
								'is_rt':	'mean',
								'is_reply':	'mean',
								'text':		' '.join,
								'user_id':	'first',
								'word_count':'mean'
					},
					'weekly':{'id':			'count',
							  'target':		'first',
							  'has_url':	'mean',
							  'is_rt':		'mean',
							  'is_reply':	'mean',
							  'text':		' '.join,
							  'from_diag':	'mean',
							  'from_susp':	'mean',
							  'diag_date':	'first',
							  'user_id':	'first',
							  'word_count':'mean',
							  'before_diag':'first',
							  'before_susp':'first'
					}

				}
			},
			'vars':{
				'ig': {
					'post':{ 
						'means':['likable','interesting','sad','happy', 'has_filter', 'has_face', 'face_ct',
								 'hue','saturation','brightness','comment_count','like_count'],
						'full':	['likable','interesting','sad','happy', 'has_filter', 'has_face', 'face_ct',
								 'likable|var','interesting|var','sad|var','happy|var',
								 'hue','saturation','brightness','comment_count','like_count','before_diag','before_susp','target'],
						'only_ratings': ['happy','sad','likable','interesting'],
						'no_addtl_means':['hue','saturation','brightness','comment_count','like_count', 'has_filter', 'has_face', 'face_ct',],
						'no_addtl_full':['hue','saturation','brightness','comment_count','like_count', 'has_filter', 'has_face', 'face_ct', 'before_diag','before_susp','target']
						},
					'username':{
						'means':['likable','interesting','sad','happy', 'has_filter', 'has_face', 'face_ct',
								 'hue','saturation','brightness','comment_count','like_count','url'],
						'full':	['likable','interesting','sad','happy', 'has_filter', 'has_face', 'face_ct',
								 'likable|var','interesting|var','sad|var','happy|var',
								 'hue','saturation','brightness','comment_count','like_count','url','one_word','description','target'],
						'only_ratings': ['happy','sad','likable','interesting'],
						'no_addtl_means':['hue','saturation','brightness','comment_count','like_count','url', 'has_filter', 'has_face', 'face_ct',],
						'no_addtl_full':['hue','saturation','brightness','comment_count','like_count','url', 'has_filter', 'has_face', 'face_ct','target']
						},
					'created_date':{
						'means':['likable','interesting','sad','happy', 'has_filter', 'has_face', 'face_ct',
								 'hue','saturation','brightness','url','comment_count','like_count'],
						'full':['likable','interesting','sad','happy', 'has_filter', 'has_face', 'face_ct',
								'likable|var','interesting|var','sad|var','happy|var',
								'hue','saturation','brightness','url','comment_count','like_count','one_word','description',
								'before_diag','before_susp','target'],
						'only_ratings': ['happy','sad','likable','interesting'],
						'no_addtl_means':['hue','saturation','brightness','comment_count','like_count','url', 'has_filter', 'has_face', 'face_ct',],
						'no_addtl_full':['hue','saturation','brightness','comment_count','like_count','url', 'has_filter', 'has_face', 'face_ct','before_diag','before_susp','target']
						}
				},
				'tw':{
					'weekly':{ 
						'means':['LabMT_happs', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
								 'tweet_count', 'word_count', 'has_url', 'is_rt', 'is_reply','LIWC_happs'] + LIWC_vars,
						'no_addtl_means':['LabMT_happs', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
										  'is_rt', 'is_reply','tweet_count', 'word_count', 
										  'has_url', 'LIWC_happs'] + LIWC_vars,
						'model':['LabMT_happs','ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
								   'tweet_count', 'word_count', 'has_url', 'is_rt', 'is_reply', 'LIWC_happs'] + LIWC_vars,
						'full':	['tweet_id', 'user_id', 'LabMT_happs','ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
								   'time_unit', 'tweet_count', 'word_count', 'has_url', 'is_rt', 'is_reply', 'target',
								   'before_diag','before_susp', 'created_date','diag_date','LIWC_happs'] + LIWC_vars,
						'no_addtl_full': ['tweet_id', 'user_id', 'LabMT_happs', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
										  'time_unit', 'tweet_count', 'word_count', 'has_url',
										  'is_rt', 'is_reply', 'target','before_diag','before_susp',
										  'created_date','diag_date', 'LIWC_happs'] + LIWC_vars
						},
					'user_id':{
						'means':['LabMT_happs', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance', 'tweet_count', 'word_count', 
								 'has_url', 'is_rt', 'is_reply', 'LIWC_happs'] + LIWC_vars,
						'no_addtl_means':['LabMT_happs', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance', 'tweet_count', 'word_count', 
								 'has_url', 'is_rt', 'is_reply', 'LIWC_happs'] + LIWC_vars,
						'model':['LabMT_happs', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance', 
								   'word_count', 'has_url', 'is_rt', 'is_reply', 'LIWC_happs'] + LIWC_vars,
						'full':	['tweet_id', 'user_id', 'LabMT_happs', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
								   'time_unit', 'tweet_count', 'word_count', 'has_url', 
								   'is_rt', 'is_reply', 'target', 'LIWC_happs'] + LIWC_vars,
						'no_addtl_full': ['tweet_id', 'user_id', 'LabMT_happs', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
										  'time_unit', 'tweet_count', 'word_count', 'has_url', 
										  'is_rt', 'is_reply', 'target', 'LIWC_happs'] + LIWC_vars
						},
					'created_date':{
						'means':['LabMT_happs', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
								   'tweet_count', 'word_count', 'has_url', 'is_rt', 'is_reply', 'LIWC_happs'] + LIWC_vars,
						'no_addtl_means':['LabMT_happs', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
								   'tweet_count', 'word_count', 'has_url', 'is_rt', 'is_reply', 'LIWC_happs'] + LIWC_vars,
						'model':['LabMT_happs', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
								 'tweet_count', 'word_count', 'has_url', 'is_rt', 'is_reply','LIWC_happs'] + LIWC_vars,
						'full':['tweet_id', 'user_id', 'LabMT_happs', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
								 'time_unit', 'tweet_count', 'word_count', 'has_url', 'is_rt', 'is_reply', 'target',
								 'from_diag','from_susp','before_diag','before_susp', 
								 'created_date','diag_date','LIWC_happs'] + LIWC_vars,
						'no_addtl_full':['tweet_id', 'user_id', 'LabMT_happs', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
								 'time_unit', 'tweet_count', 'word_count', 'has_url', 'is_rt', 'is_reply', 'target',
								 'from_diag','from_susp','before_diag','before_susp', 
								 'created_date','diag_date','LIWC_happs'] + LIWC_vars,
						}
				}
					
			},
			'fields_to_merge':{
				'tw':{
					'weekly':		['user_id','has_url','is_rt','is_reply','target','text','tweet_count','word_count','from_diag','from_susp','before_diag','before_susp','created_date','diag_date'],
					'created_date':	['user_id','has_url','is_rt','is_reply','target','text','tweet_count','word_count','from_diag','from_susp','before_diag','before_susp','created_date','diag_date'],
					'user_id':		['user_id','has_url','is_rt','is_reply','target','text','tweet_count','word_count']
					},
				'ig':{}
			},
			'merge_on':{
				'tw':{
					'weekly':		['user_id','created_date'],
					'created_date':	['user_id','created_date'],
					'user_id':		['user_id']
					},
				'ig':{}
			},
			'model_vars_excluded':{
				'ig': {
					'post':['username','url','rater_id|count','likable|var',
							'description|join', 'interesting|var',
							'sad|var', 'one_word|join', 'happy|var','before_diag','before_susp','target'],
					'username':['username','interesting|var','sad|var',
								'happy|var','likable|var', 
								'one_word','description', 'target'],
					'created_date':['username','interesting|var','sad|var',
									'happy|var','likable|var', 
									'one_word','description','before_diag','before_susp','target']
				},
				'tw':{
					'weekly':['user_id','before_diag','before_susp','target'],
					'user_id':['user_id', 'target','total_words', 'tweet_count',
							   'LIWC_num_words', 'LabMT_num_words', 
							   'ANEW_num_words','LIWC_total_count'],
					'created_date':['before_diag','before_susp','target']
				}
			},
			'rated':photos_rated,
			'has_test':has_test 
		}

	# additional_data flag means ratings data should be included, so we add those vars to the aggregation dict
	if (platform == 'ig') and additional_data:
		params['agg_func'][platform]['post'].update(			
			{'rater_id':	'count',
			'happy':		['mean','var'],
			'sad':			['mean','var'],
			'interesting':	['mean','var'],
			'likable':		['mean','var'],
			'one_word':		' '.join,
			'description':	'    '.join}
		)
		params['agg_func'][platform]['username'].update(
			{'likable':		['mean','var'],
			'interesting':	['mean','var'],
			'happy':		['mean','var'],
			'sad':			['mean','var'],
			'one_word':		'_|_'.join,
			'description':	'__|__'.join}
		)
		params['agg_func'][platform]['created_date'].update(
			{'likable':		['mean','var'],
			'interesting':	['mean','var'],
			'happy':		['mean','var'],
			'sad':			['mean','var'],
			'one_word':		'_|_'.join,
			'description':	'__|__'.join}
		)

	# add event_date to aggregation vars for trauma (PTSD), conception (preg)
	if condition in ['ptsd','pregnancy']: 
		if platform == 'tw':
			params['agg_func'][platform]['weekly'].update(			
				{'event_date':	'first',
				'from_event':	'mean'}
			)
			params['agg_func'][platform]['created_date'].update(			
				{'event_date':	'first',
				'from_event':	'first'}
			)
			for gbt in ['created_date','weekly']:
				params['vars'][platform][gbt]['full'].extend(['event_date','from_event'])
				params['vars'][platform][gbt]['no_addtl_full'].extend(['event_date','from_event'])
				params['fields_to_merge'][platform][gbt].extend(['event_date','from_event'])

		elif platform == 'ig':
			params['agg_func'][platform]['created_date'].update(			
				{'event_date':	'first',
				'from_event':	'mean'}
			)
			params['agg_func'][platform]['post'].update(			
				{'event_date':	'first',
				'from_event':	'first'}
			)

	return params 


def data_loader(load_from, condition, platform, path_head):
	''' Loads data dict from saved file '''

	if load_from == 'pickle':
		data = pickle.load(open("{path}{cond}_{pl}_data.p".format(path=path_head,
																  cond=condition,
																  pl=platform), 
								"rb" ) )
	elif load_from == 'file':
		pass


def report_sample_sizes(params, conn, cond, plat_long, test_cutoff, 
						test=None, show_all=False,
						table=['depression','pregnancy','ptsd','cancer'], extra_field=''):
	''' Printout of sample sizes across conditions and thresholds '''

	print 'SAMPLE SIZES: target populations'
	print

	if not show_all:
		table = [cond] # if not show_all, then just show current condition being analyzed

	for t in table:
		if test:
			extra_field = ', {}'.format(test)

		q = 'select platform{} from {} where username is not null and diag_year is not null and disqualified=0'.format(extra_field, t)
		samp = pd.read_sql_query(q,conn)
		print 'TARGET :: {cond} / {plat} total:'.format(cond=t.upper(), plat=plat_long.upper()), samp.ix[samp.platform=='{}'.format(plat_long),:].shape[0]
		if t in ['ptsd','depression']:
			print 'TARGET :: {cond} / {plat} {test} > {tc}:'.format(cond=t.upper(),plat=plat_long.upper(),test=test,tc=test_cutoff), samp.ix[(samp.platform=='{}'.format(plat_long)) & (samp[test] > test_cutoff), :].shape[0]
		print
		print

	print 'SAMPLE SIZES: control populations'
	print

	for t in table:
		q = 'select platform from control where {}="No" and disqualified=0'.format(t)
		samp = pd.read_sql_query(q,conn)
		print 'CONTROL :: {cond} / {plat} total:'.format(cond=t.upper(),plat=plat_long.upper()),samp.ix[samp.platform=='{}'.format(plat_long),:].shape[0]
		print


def report_share_sm_disq(fname):
	''' Reports on the # of subjects disqualified because they refused to share social media data '''

	d = pd.read_csv(fname).drop(0,0) # first row of qualtrics data is extra header info
	disq = d.share_sm.str.startswith('No')
	attempted = d.shape[0]
	disqualified = np.sum(disq)
	print 'Total attempted:', attempted
	print 'Disqualified for share_sm:', disqualified
	print '% disq for share_sm:', disqualified/float(attempted)
	return attempted, disqualified


def urls_per_user(data, local_params):
	''' avg num urls rated per user '''

	locus = local_params['locus']
	user_unit = local_params['user']
	post_unit = local_params['post']

	targ_url_ct = data['target'][locus].groupby([user_unit,post_unit]).count().reset_index().groupby(user_unit).count()[post_unit]
	control_url_ct = data['control'][locus].groupby([user_unit,post_unit]).count().reset_index().groupby(user_unit).count()[post_unit]
	all_url_ct = pd.concat([targ_url_ct,control_url_ct])
	return targ_url_ct, control_url_ct, all_url_ct


def urls_rated_by_pop(data, local_params):
	''' Count of number of rated photos, per target/control samples 
		Called by get_descriptives() '''

	post_unit = local_params['post']

	c = data['control']['ratings'][post_unit].unique().shape[0]
	t = data['target']['ratings'][post_unit].unique().shape[0]
	return (c,t)


def subj_data_by_pop(data, target, platform, local_params, conn):
	''' Gets descriptive data for study participants, grouped by target/control 
		Called by get_descriptives() '''

	locus = local_params['locus']
	user_unit = local_params['user']

	output = {'control':{},'target':{}}
	popdata = {}

	popdata['control'] = pd.Series(data['control'][locus][user_unit].unique())
	popdata['target'] = pd.Series(data['target'][locus][user_unit].unique())

	output['control']['ct'] = popdata['control'].shape[0]
	output['target']['ct'] = popdata['target'].shape[0]

	for pop in ['control','target']:
		tup = tuple(popdata[pop].astype(str).values)

		if pop == 'control':
			q = 'select {uunit}, year_born, gender from control where {cond}="No" and {uunit} in {uset}'.format(uunit=user_unit,
																												cond=target,
																												uset=tup)
			d = pd.read_sql_query(q,conn)
			d.drop_duplicates(subset=[user_unit], inplace=True)

			output['control']['femprop'] = round(np.sum(d.gender=="Female")/float(d.shape[0]),2)
			output['control']['perc_diag_within_13_15'] = None

		elif pop == 'target':
			q = 'select {uunit}, year_born, diag_date from {cond} where platform="{pl}" and {uunit} in {uset} and diag_date is not null'.format(cond=target,
																													  pl=platform,
																													  uunit=user_unit,
																													  uset=tup)
			d = pd.read_sql_query(q,conn)
			d.drop_duplicates(subset=[user_unit], inplace=True)
			
			d.dropna(subset=['diag_date'], inplace=True)
			
			output['target']['femprop'] = None

			ts = pd.to_datetime(d.diag_date)
			ct_within_2013_2015 = ts[(ts < pd.to_datetime('2015-12-31')) & (ts > pd.to_datetime('2013-01-01'))].shape[0]
			perc_within_2013_2015 = ct_within_2013_2015/float(ts.shape[0])
			output['target']['perc_diag_within_13_15'] = round(perc_within_2013_2015,2)
			
		current_year = datetime.datetime.now().year
		age = pd.Series(current_year - d.year_born) 
		output[pop]['age'] = {'min':age.min(),
							  'max':age.max(),
							  'mean':round(age.mean(),2),
							  'std':round(age.std(),2)}

	return output

def get_descriptives(data, target, platform, additional_data, conn, return_output=False, doPrint=True):
	''' Reports on descriptive statistics of overall dataset used for analysis.
		Also includes stats broken down by target/control pop.
		Meant for printing, but also returns data from function call '''

	if platform == 'twitter':
		local_params = {'locus':'tweets', 'post':'id', 'user':'username'}
	elif platform == 'instagram':
		local_params = {'locus':'all', 'post':'url', 'user':'username'}

	pop_data = subj_data_by_pop(data,target,platform,local_params,conn)

	ct = {}

	ct['target'], ct['control'], post_ct = urls_per_user(data,local_params)

	if additional_data: # instagram only, additional_data means ratings
		urls_rated_c, urls_rated_t = urls_rated_by_pop(data,local_params)
	
	if doPrint:
		print 'Total posts across all groups:', data['target'][local_params['locus']].shape[0] + data['control'][local_params['locus']].shape[0]
		print 'Mean posts per participant:', round(post_ct.mean(),2), '(SD={})'.format(round(post_ct.std(),2))
		print 'Median posts per participant:', round(post_ct.median(),2)
		
		if additional_data:
			print 'Photos rated: TARGET population ::', urls_rated_t
			print 'Photos rated: CONTROL population ::', urls_rated_c
		print
		
		for pop in ['target','control']:
			print 'POPULATION: {}'.format(pop.upper())
			print 'Total participants analyzed:', pop_data[pop]['ct']
			print 'Total posts:', data[pop][local_params['locus']].shape[0]
			print 'Mean posts per participant:', round(ct[pop].mean(),2), '(SD={})'.format(round(ct[pop].std(),2))
			print 'Median posts per participant:', round(ct[pop].median(),2)
			if pop == 'control': 
				print 'Proportion female (control only):', pop_data[pop]['femprop']
			if pop == 'target': 
				print 'Earliest date of first diagnosis:', pd.to_datetime(data[pop][local_params['locus']].diag_date).describe()['first']
				print 'Latest date of first diagnosis:', pd.to_datetime(data[pop][local_params['locus']].diag_date).describe()['last']
				print 'Proportion diagnosed between 2013-2015 (target only):', pop_data[pop]['perc_diag_within_13_15']
			print 'Average age:', pop_data[pop]['age']['mean'], '(SD={})'.format(pop_data[pop]['age']['std'])
			print 'Min age:', pop_data[pop]['age']['min']
			print 'Max age:', pop_data[pop]['age']['max']
			print
			print

	output = {'pop_data':pop_data, 'post_ct':post_ct}

	if additional_data: # instagram only
		output['urls_rated'] = {'c':urls_rated_c,'t':urls_rated_t}
		
	output['post_ct'].plot(kind='hist', 
						  bins=50, 
						  title='{} posts per participant, {} (target + control)'.format(platform.upper(),target))
	plt.show()

	if return_output:
		return output


def out_of_100(prev, prec, spec, rec, N=100):
	''' Reports model accuracy based on a hypothetical sample of 100 data points.
		Eg. "out of 100 samples, there were X positive IDs, Y false alarms, and Z false negatives..." '''

	import math 

	print 'Out of {} random observations...'.format(N)
	print
	n_1 = math.ceil(N*prev)
	print 'n1', n_1
	n_0 = N - n_1
	print 'n0', n_0
	pos_id = math.ceil(n_1 * rec)
	neg_id = math.ceil(n_0 * spec)
	f_alarm = n_0 - neg_id
	f_neg = n_1 - pos_id
	print pos_id, "positive IDs"
	print f_alarm, "false alarms"
	print f_neg, "false negatives"
	print neg_id, "negative IDs"
	print
	print 'reconstituted total:', np.sum([pos_id, f_alarm, f_neg, neg_id])


def sample_2_ratings(x,col_pad):
	tmp = x.dropna()
	if tmp.shape[0] > 1:
		out = np.append(tmp.sample(2).values, np.zeros(col_pad))
		return out
	else:
		return None

def get_ratings_corr(data, K=5):
	''' Computes 5-fold CV Pearson's r for each of the four ratings categories '''
	
	ratings_rs = []

	for cat in ['likable','interesting','sad','happy']:
		ratedata = data['target']['ratings'][['url','rater_id',cat]].pivot(index='url', 
																		columns='rater_id', 
																		values=cat)
		raters_samp_size = 2
		col_shape = ratedata.shape[1]
		col_padding = col_shape - raters_samp_size

		pearson_rs = []
		for i in range(K):
			x = ratedata.apply(sample_2_ratings, args=(col_padding,), axis=1)
			''' I do not understand why apply insists I return an object with the same dimensions, but in this case, it does.
				I only want two columns to come back though, representing the two raters' values I sampled from each rate set.
				So, you padded the two column return with zeros to match up with the original df shape.
				Then you drop them once the apply() is finished.  Seems really dumb. '''
			x.drop(x.columns[range(raters_samp_size,col_shape)],1,inplace=True)
			x = x.dropna()
			x.columns = ['r1','r2']
			pearson_rs.append( pearsonr(x.r1,x.r2) )

		pearson_cv = np.mean([x[0] for x in pearson_rs])
		pearson_p = np.mean([x[1] for x in pearson_rs])
		print 'Avg Pearson correlation for {} between random two raters (5-fold CV):'.format(cat.upper()), round(pearson_cv,2)
		print 'Avg p-value:', pearson_p
		print
		ratings_rs.append((cat,pearson_cv,pearson_p))

	return ratings_rs


def make_data_dict(params, condition, test_name, conn, doPrint=False):
	''' defines basic nested data structure for entire analysis 
		note: m == medium (ie. platform) '''

	data = {'target':{}, 'control':{}, 'master':{}, 'before':{}, 'after':{}}

	if params['has_test']:
		data['cutoff'] = {}
		data['cutoff']['unames'] = pd.read_sql_query(params['q']['t']['unames'],conn)

		if doPrint:
			print  ('Number {cond} subjects above {test} cutoff:'.format(cond=condition,
																		 test=test_name),
					data['cutoff']['unames'].shape[0])
	return data 


def get_additional_data(data, params, platform, condition, pop, pop_long, 
						additional_data, include_filter, conn, doPrint=False):
	''' For Instagram: Checks if this condition has photo ratings, if so, adds to basic (hsv) data. '''

	# this if- block deals with conditions where photos have been rated
	if params['rated'] and additional_data: 
		
		kind = 'ratings'
		
		d = get_photo_ratings(params, conn, pop)

		# print 'Indices with broken photos:', find_and_drop_broken_photos(d, drop_from_df=False)
		find_and_drop_broken_photos(d) # this call actually drops them

		d2 = get_meta(params, conn, pop)
		
		# Now merge the meta_ig data with the photo_ratings_depression data
		consolidate_data(d2, d, platform, pop_long, kind, data)

		if doPrint:
			# Hunting for ratings errors
			print_coltype(data, condition, platform, pop_long)
			ixs = find_chartype(data,'interesting')

		''' Warning: This isn't very robust - you have this conditional set here because you discovered it's only the 
			depression target group photo ratings that have the problematic "strings for photo ratings" issue.  

			A more generalized way to go about this would be to create a flag from find_chartype() that identifies whether 
			map_str_ratings_to_numeric() is necessary...but even then, the custom mapping you created is really case specific 
			and does not easily generalize. 
			
			It's probably not worth trying to generalize these functions for the purposes of your dissertation research. '''

		# quirky corrections
		if (pop_long == 'target') & (condition == 'depression'):
			map_str_ratings_to_numeric(data) 
		elif (pop_long == 'control') & (condition == 'ptsd'):
			data[pop_long]['ratings'].ix[data[pop_long]['ratings'].sad=='g','sad'] = 1.0
			data[pop_long]['ratings'].sad = data[pop_long]['ratings'].sad.astype(float)

		if doPrint:
			# Check fixed ratings 
			print_coltype(data, condition, platform, pop_long)

		# And now merge ratings data with hsv data
		consolidate_data(data[pop_long][kind], data[pop_long]['hsv'], platform, pop_long, 'all', data)

	else:
		d = data[pop_long]['hsv']
		d2 = get_meta(params, conn, pop)
		tmp = d2.copy()
		
		# Now merge the meta_ig data with the photo_ratings_depression data
		consolidate_data(d2, d, platform, pop_long, 'all', data)

	if include_filter:
		data[pop_long]['all']['has_filter'] = 1
		data[pop_long]['all'].ix[ data[pop_long]['all']['filter'] == 'Normal','has_filter'] = 0

		if doPrint:
			print 'Prop {} with filter:'.format( pop_long ), data[pop_long]['all']['has_filter'].mean()


def cut_low_posters(data, pop_long, std_frac=0.5, doPrint=True):
	''' Cuts all data from participants with a low number of posts.
		'Low' is defined as fewer than (mean_posts_within_pop - 0.5*std_posts_within_pop) 
		WARNING: Does not work for Twitter data as-is!!   '''

	df = data[pop_long]['all']
	d = (df.groupby(['user_id','url'])
		   .count()
		   .reset_index()
		   .groupby('user_id')
		   .count()
		)
	offset = std_frac * d.url.std()
	cutoff = d.url.mean() - offset
	cutdf = d.ix[d.url >= cutoff,:]
	if doPrint:
		print 'target pop before cut:', d.shape
		print 'target pop after cut:', cutdf.shape
	new_ids = cutdf.reset_index().user_id
	data[pop_long]['all'] = df.ix[df.user_id.isin(new_ids),:].copy()


def convert_field_to_float(x):
	''' Some fields are empty string in db but need to be None objects '''
	if (x == '') | (x == 'None'):
		return None
	elif x == None:
		return x
	else:
		try:
			return float(x)
		except:
			print 'FAILED ON:', x


def prepare_raw_data(data, platform, params, conn, gb_types, condition, periods, turn_points, 
					 post_event=False, posting_cutoff=False, additional_data=False, 
					 include_filter=True, limit_date_range=False):
	''' Pulls data from db, cleans, and aggregates. Also creates subsets based on diag/susp date '''

	# collect target and control data
	for pop, pop_long in [('t','target'),('c','control')]:
		
		print '{} DATA:'.format(pop_long.upper())
		
		# get basic data (hsv or tweets...no photo ratings yet)
		get_basic_data(data, platform, params, conn, pop, pop_long, limit_date_range)

		''' For tweets, additional data are word features, but because they are already in db as gb'd data,
			we add them in during the make_groupby() process, instead of here.  Photo ratings, on the other hand,
			are not stored as aggregated data. '''
		if (platform == 'ig'):
			get_additional_data(data, params, platform, condition, pop, pop_long, 
								additional_data, include_filter, conn)
			
		if posting_cutoff and (platform == 'ig'):
			cut_low_posters(data, pop_long)
					
		# aug 11 2016: database diag_date for all conditions but depression are of form "YYYYMMDD"
		# but we need "YYYY-MM-DD". this line converts them.  should just update db register eventually.
		if condition != 'depression':
			if platform == 'ig':
				dset = 'all'
			elif platform == 'tw':
				dset = 'tweets'

			#if pop == 't':
			#        data[pop_long][dset].diag_date = data[pop_long][dset].diag_date.apply(lambda x: '-'.join([x[0:4],x[4:6],x[6:]]))
			for date_type in ['diag','event']:
				field = 'from_{}'.format(date_type)
				data[pop_long][dset][field] = data[pop_long][dset][field].apply(convert_field_to_float)

				if pop_long == 'control':
					field2 = '{}_date'.format(date_type)
					data[pop_long][dset][field] = data[pop_long][dset][field].astype(float)
					try:
						data[pop_long][dset][field2] = data[pop_long][dset][field2].astype(float)
					except Exception, e:
						print 'ERROR in prepare_raw_data with making control dtypes floats:'
						print str(e)
						a = data[pop_long][dset].apply(lambda x: (x['user_id'], x[field2]) if x[field2] is not None else (None,None))
						a = pd.Series(a)
						print a.dropna()


		# aggregate data by groupby types (url, username, created_date)
		make_groupby(data[pop_long], platform, pop_long, params, gb_types, 
					 conn, condition, additional_data,
					 doPrint=True)

		if pop_long == 'target':
			# creates before/after subsets for target pop
			# subsets stored in: data[m]['target'][period][turn_point]['all']
			make_timeline_subsets(data, platform, periods, post_event)
		
			for period in periods:
				for turn_point in turn_points:
					turn_point = turn_point.replace("-","_") # because of the weird -/_ switch you did in Twitter
					# aggregate data by groupby types (url, username, created_date)
					make_groupby(data[pop_long][period][turn_point], platform, pop_long, params, gb_types, 
								 conn, condition, additional_data, period = period, turn_point=turn_point)


def get_pop_unames(params, m, conn, pop):
	''' Get all usernames for a given population '''

	return pd.read_sql_query(params['q'][pop]['unames'], conn)


def get_photo_ratings(params, conn, pop, doPrint=False):
	''' Gets set of all Instagram photo ratings for either target or control condition ''' 

	d = pd.read_sql_query(params['q'][pop]['photo_ratings'], conn)
	if doPrint:
		print 'Num ratings BEFORE dropping empty rater_id fields:', d.shape[0]
	d = d.dropna(subset=['rater_id']) # there are a few empties, we get rid of them here
	if doPrint:
		print 'Num ratings AFTER dropping empty rater_id fields:', d.shape[0]
	return d 


def find_and_drop_broken_photos(x, drop_from_df=True):
	''' find and delete rows where photo failed to load '''

	broken = x.ix[(x.one_word=='broken') & x.description.str.contains('image|photo', flags=re.IGNORECASE),['one_word','description']]
	x.drop(broken.index,0,inplace=True)
	if drop_from_df:
		x.reset_index(drop=True, inplace=True)
	else:
		return broken.index


def get_meta(params, conn, pop, doPrint=False):
	''' get data from meta_{platform} table '''

	if pop == 't':
		d = pd.read_sql_query(params['q'][pop]['meta'], conn)

	elif pop == 'c':
		unames = pd.read_sql_query(params['q'][pop]['unames'],conn)
		all_meta = pd.read_sql_query(params['q']['all_meta'],conn)
		d = all_meta.ix[all_meta.username.isin(unames.username),:]
	
	if doPrint:
		print 'Num META {} entries:'.format(pop.upper()), d.shape[0]
	return d


def get_basic_data(data, m, params, conn, pop, pop_long, limit_date_range=False, doPrint=False):
	''' gets the set of variables that all observations have filled (eg. hsv for photos, tweet data for twitter)
		this is basically just a wrapper for get_hsv() and get_tweet_metadata() '''

	if m == 'ig':
		get_hsv(data, m, params, conn, pop, pop_long, limit_date_range)
	elif m == 'tw':
		get_tweet_metadata(data, m, params, conn, pop, pop_long, limit_date_range, doPrint=doPrint)


def get_word_feats(params, conn, pop_long, condition, tunit, key='word_features'):
	''' Gets data from word_features table. 
		Query is already determined for a specific condition (pregnancy, etc) from define_params() '''

	# we can't drop this in params because we want different sets for each groupby time-unit
	# this isn't set at the point where we create the params object
	if pop_long == 'target':
		field = condition 
	elif pop_long == 'control':
		field = 'no_{}'.format(condition)

	q = "select * from word_features where {f} = 1 and time_unit = '{tu}'".format(f=field,tu=tunit)
	
	wf = pd.read_sql_query(q,conn)
	
	to_drop = ['table_id','depression','no_depression','pregnancy','no_pregnancy','ptsd','no_ptsd','cancer','no_cancer']
	wf.drop(to_drop, 1, inplace=True)

	Imputer(missing_values='0', strategy='mean', axis=0, verbose=0, copy=True)
	return wf 


def fix_has_url(df, doPrint=True):
	''' The Twitter has_url field apparently didn't get populated a lot.  We can fix it with some regex: '''

	before = np.sum(df.has_url==1)
	df.ix[df.text.str.contains('http:'), 'has_url'] = 1
	df.ix[~df.text.str.contains('http:'), 'has_url'] = 0
	df.has_url = df.has_url.astype(int)
	if doPrint:
		print 'Number of tweets with url in text:', before 
		print 'Adding url tags...'
		print 'Number of tweets with url in text:', np.sum(df.has_url==1)


def add_is_reply(df, doPrint=True):
	''' 'is_reply' field looks to see whether there's an @ tag in the tweet '''

	print 'Adding reply tags...'
	df['is_reply'] = 0
	df.ix[df.text.str.contains('@'), 'is_reply'] = 1
	if doPrint:
		print 'Number of tweets with @ in text:', np.sum(df.is_reply==1)


def add_is_rt(df, doPrint=True):
	''' 'has_rt' field, checking whether post is a retweet '''

	print 'Adding RT tags...'
	df['is_rt'] = 0
	df.ix[df.text.str.contains('^RT '), 'is_rt'] = 1
	if doPrint:
		print 'Number of tweets with RT:', np.sum(df.is_rt==1)


def count_words_in_tweet(d):
	''' Counts the number of space-separated strings in a single tweet '''
	d['word_count'] = d.text.apply(lambda x: len(x.split()))


def fix_from_counts(d):
	''' Fixes some rows with missing from_diag or from_susp values '''

	def inner_fix_from_counts(dd):

		try: 
			float(dd['from_diag'])
		except:
			created = pd.to_datetime(dd['created_date'])
			diagnosed = pd.to_datetime(dd['diag_date'])
			from_d = created - diagnosed
			if from_d < 0:
				before_diag = 1
			else:
				before_diag = 0
		return 

	d = d.apply(inner_fix_from_counts, axis=1)


def hourly_plot(conn, condition):
	''' Makes plot of aggregated hourly post data (used mainly for Twitter comparison against De Choudhury) 
		WARNING: Use this function with care, it overwrites meta_tw and causes strange behavior that is not fixed! '''
	
	tz, tups = to_localtime_wrapper(conn, condition)

	# Rather than try and add a column into meta_tw...
	# We put meta_tw in a df, add local_time, and then overwrite the old meta_tw with the new one.  Pandas is great.

	metatw = pd.read_sql_query('select * from meta_tw', conn)
	metatw.fillna('', inplace=True)


	reformat_diag_date(metatw)
	get_local_time_data(metatw, tz)
	metatw['at_night'] = metatw.local_time.apply(tag_insomnia)

	# Overwrite meta_tw in project.db
	a = pd.read_sql_query('pragma table_info(meta_tw)', conn)
	dtypes = {x[1]:x[2] for x in a.values}

	metatw.to_sql('meta_tw', conn, if_exists='replace')

	q = "select username, local_time, d_from_diag_{} as ddate from meta_tw where local_time != ''".format(condition)
	hastime = pd.read_sql_query(q, conn)

	def make_time_obj(x):
		if len(x) > 0:
			return datetime.datetime.strptime(x,"%H:%M:%S").time()
		else:
			return x

	q = "select username from control where platform = 'twitter' and {}='No'".format(condition)
	c_unames = pd.read_sql_query(q, conn)

	fields = ['username','local_time']

	hastime_target = hastime.ix[hastime.ddate.notnull(), fields]
	hastime_target.index = pd.to_datetime(hastime_target.local_time)

	hastime_control = hastime.ix[hastime.username.isin(c_unames.username),fields]
	hastime_control.index = pd.to_datetime(hastime_control.local_time)

	hastime_target.local_time.resample('H').count().plot(label='target')
	hastime_control.local_time.resample('H').count().plot(label='control')
	plt.legend()


def get_tweet_metadata(data, m, params, conn, pop, pop_long, limit_date_range=False, 
					   t_maxback=-(365 + 60), t_maxfor=365, c_maxback=365*2, doPrint=False,
					   key='tweets'):
	''' Collects all Twitter metadata for a given population, and does some cleaning.
		If limit_date_range == True, then posts are limited to within maxback/maxfor days.
		For target populations, these limits are based on diag/susp date. 
		For control populations, limits extend back from 01 Jan 2016 '''

	unames = get_pop_unames(params, m, conn, pop)
	meta = get_meta(params, conn, pop, doPrint)

	# because you converted None to '' in make_hourly_plot for some reason
	if (meta.from_diag.dtype != float) & (meta.from_diag.dtype != int):
		meta[['from_diag','from_susp']] = meta[['from_diag','from_susp']].replace({'':None}).astype(float) 
	
	all_tweets = meta.ix[meta.username.isin(unames.username), :].copy()
	all_tweets.drop_duplicates(subset='id', inplace=True)

	# indicates membership of target population
	if pop == 't':
		all_tweets['target'] = 1
	elif pop == 'c':
		all_tweets['target'] = 0

	fix_has_url(all_tweets)
	add_is_reply(all_tweets)
	add_is_rt(all_tweets)
	count_words_in_tweet(all_tweets)
	mark_before_after(all_tweets, pop_long)

	all_tweets.created_date = pd.to_datetime(all_tweets.created_date) # convert date field to explicit date type

	''' To restrict the date range of tweets analyzed, we can set the t_maxback, t_maxfor, and c_maxback 
		variables, and set limit_date_range=True. Otherwise, we use all tweets. '''
	if limit_date_range:
		if pop_long == 'target':
			max_backward = t_maxback # one year back plus 60 days to make sure we're including a full year back from suspected 
			max_forward = t_maxfor
			mask = (all_tweets.from_diag >= max_backward) & (all_tweets.from_diag <= max_forward)
		
		elif pop_long == 'control':
			day_span = c_maxback # to approximate going one year back and forwards from diag_date in target pop
			dfp = [(datetime.datetime.now() - d).days for d in all_tweets.created_date]
			all_tweets['days_from_present'] = dfp
			mask = (all_tweets.groupby('username')['days_from_present']
							  .apply(lambda x: x.nsmallest(day_span))
							  .reset_index()['level_1'] # level_1 gets row indices from all_tweets to use as mask
					)
		tweets = all_tweets.ix[ mask, : ].copy()
	else:
		fix_from_counts(all_tweets)
		tweets = all_tweets.copy()
	
	if doPrint:
		print 'Num tweets in {}, before dropping duplicates:'.format(pop_long.upper()), tweets.shape[0]
		print 'Num tweets in {}, after dropping duplicates:'.format(pop_long.upper()), data[pop_long][key].shape[0]

	data[pop_long]['tweets'] = tweets 


def get_hsv(data, m, params, conn, pop, pop_long, limit_date_range=False, 
			t_maxback=-(365 + 60), t_maxfor=365, c_maxback=365*2, 
			doPrint=False, cols=['url','hue','saturation','brightness']):
	''' Gets HSV values for Instagram photos '''

	hsv = pd.read_sql_query(params['q']['all_hsv'],conn)
	hsv.dropna(inplace=True)

	if doPrint:
		print 'Number photos with HSV ratings (all conditions):', hsv.shape[0]

	unames = get_pop_unames(params, m, conn, pop)
	metaig = get_meta(params, conn, pop)


	''' To restrict the date range of tweets analyzed, we can set the t_maxback, t_maxfor, and c_maxback 
		variables, and set limit_date_range=True. Otherwise, we use all tweets. '''
	if limit_date_range:
		if pop_long == 'target':
			max_backward = t_maxback # one year back plus 60 days to make sure we're including a full year back from suspected 
			max_forward = t_maxfor
			mask = (metaig.from_diag >= max_backward) & (metaig.from_diag <= max_forward)
		
		elif pop_long == 'control':
			day_span = c_maxback # to approximate going one year back and forwards from diag_date in target pop
			dfp = [(datetime.datetime.now() - d).days for d in pd.to_datetime(metaig.created_date)]
			metaig['days_from_present'] = dfp
			mask = (metaig.groupby('username')['days_from_present']
							  .apply(lambda x: x.nsmallest(day_span))
							  .reset_index()['level_1'] # level_1 gets row indices from all_tweets to use as mask
					)
		metaig = metaig.ix[ mask, : ].copy()
	else:
		metaig = metaig.copy()

	urls = metaig.ix[metaig.username.isin(unames.username),'url'].values
	if doPrint:
		print 'Num HSV-rated photos with URL in {}, before dropping duplicates:'.format(pop_long.upper()), hsv.ix[hsv.url.isin(urls),:].shape[0]

	data[pop_long]['hsv'] = hsv.ix[hsv.url.isin(urls),cols].drop_duplicates(subset='url').copy()

	if doPrint:
		print 'Num HSV-rated photos with URL in {}, after dropping duplicates:'.format(pop_long.upper()), data[pop_long]['hsv'].shape[0]


def mark_before_after(d, pop_long):
	''' Creates indicator for whether post occurred before or after diag/susp date '''
	if pop_long == 'target':
		for date in ['diag','susp']:
			d['before_{}'.format(date)] = 0
			d.ix[d['from_{}'.format(date)] < 0, 'before_{}'.format(date)] = 1
	else:
		for date in ['diag','susp']:
			d['before_{}'.format(date)] = np.nan 


def consolidate_data(a, b, m, pop_long, kind, data):
	''' merges dfs, adds 0/1 class indicator variable '''

	data[pop_long][kind] = a.merge(b, how='inner',on='url')
	
	if pop_long == 'target':
		cl = 1
	elif pop_long == 'control':
		cl = 0

	data[pop_long][kind]['target'] = cl
	mark_before_after(data[pop_long][kind], pop_long)

	print 'Shape of consolidated {} {} data:'.format(pop_long.upper(), kind.upper()), data[pop_long][kind].shape 
	print 'Note: Number of actual data points will be lower for ratings...see get_descriptives()'


''' Note for following two functions regarding column datatypes:
	
	It appears that a few raters somehow figured out how to give their own descriptive ratings, 
	instead of using the star system.  
	As such, there are a few cases where people actually wrote in 
	'Very' or 'Slightly' instead of numerical ratings.  

	This throws off the data type of the entire column, 
	so the next few cells are devoted to fixing the problem.  

	Note: I made a key that seemed roughly in line with the raters' intentions, 
	although I admit there's some subjectivity involved.  
	So for instance, 'None' = 0, 'Slightly' = 1, etc.  
	There are only a few cases where this conversion takes place - fewer than 20 out of thousands, 
	so I decided it was acceptable.  
	I could have put it up on MTurk to have people decide how to score each term, 
	but with such a small sample, I decided it wasn't worth it. '''



def print_coltype(data, condition, m, pop_long):
	''' Get data type of all columns in data frame '''
	x = data[pop_long]['ratings']
	print 'Data column types for medium: {}, population: {}, condition: {}'.format(m, pop_long, condition)
	for c in x.columns:
		print x[c].dtype


def find_chartype(data,col):
	''' find the rows where raters used strings as photo ratings '''

	x = data['target']['ratings']
	ixs = np.where(x[col].apply(type)==unicode)[0]
	print 'Indices where raters used strings for ratings instead of numeric values:', ixs 
	print 'Total cases:', ixs.shape[0]
	print 'Actual cases:'
	print data['target']['ratings'].ix[ixs,['happy','sad','interesting','likable']]


def map_str_ratings_to_numeric(data):
	''' construct a replacement key (str -> int) and perform replacement
		NB: 'happy' is indented to show the structure - the other rating types are exactly the same keys '''

	key = 	{'happy':
				{'Not':0,'None':0,'Not sad':0,'Not at all':0,
				'Very little':1,'Not really':1,
				'Slightly':2,
				'Kind of ':3,'Kind of':3,
				'Sure':4,
				'Very':5},
			'sad':{'Not':0,'None':0,'Not sad':0,'Not at all':0,'g':1,'Very little':1,'Not really':1,'Slightly':2,'Kind of ':3,'Kind of':3,'Sure':4,'Very':5},
			'likable':{'Not':0,'None':0,'Not sad':0,'Not at all':0,'Very little':1,'Not really':1,'Slightly':2,'Kind of ':3,'Kind of':3,'Sure':4,'Very':5},
			'interesting':{'Not':0,'None':0,'Not sad':0,'Not at all':0,'Very little':1,'Not really':1,'Slightly':2,'Kind of ':3,'Kind of':3,'Sure':4,'Very':5}}

	data['target']['ratings'].replace(to_replace=key,inplace=True)


def make_timeline_subsets(data, m, periods=['before','after'], post_event=False, doPrint=True):
	''' Creates subset of target observations based on diagnosis date.

		Currently (Apr 28 2016) we make the following subsets.  Note that "before/after" means two separate sets.
		Instagram:
		  - all data from before/after DIAG_date
		  #- only observations with photo ratings, before/after DIAG_date (in conditions where we have ratings)
		  - all data from before/after SUSP_date
		  #- only observations with photo ratings, before/after SUSP_date
		Twitter:
		  - all data from before/after DIAG_date
		  - all data from before/after SUSP_date 

		(Aug 12 2016) 
		Added post_event arg, use to backward-bound pre-diag data by event_date (eg. ptsd trauma, preg concep)'''

	if m == 'ig':
		base = 'all'
	elif m == 'tw':
		base = 'tweets'

	for period in periods:
		data['target'][period] = {}

		for turn_point in ['from_diag','from_susp']:
			if period == 'before':
				if post_event:
					subset = (data['target'][base][turn_point] < 0) & (data['target'][base]['from_event'] >= 0)
				else:
					subset = (data['target'][base][turn_point] < 0)
			elif period == 'after':
				subset = (data['target'][base][turn_point] >= 0)
				
			data['target'][period][turn_point] = {base:data['target'][base].ix[subset,:]}
		
			if doPrint:
				print ('Subset shape for {} {} {}:'.format(m.upper(),
														   period.upper(),
														   turn_point.upper()), 
						data['target'][period][turn_point][base].shape
				)
	print 


def add_class_indicator(gbdf, pop, doPrint=False):
	''' Add a target class column indicator variable, for later merging '''

	if pop == 'target':
		cl = 1
	elif pop == 'control':
		cl = 0

	gbdf['target'] = cl


def make_groupby(df, m, pop, params, gb_types, 
				 conn=None, condition=None, additional_data=False,
				 period=None, turn_point=None, username_gb='created_date',doPrint=False):
	''' Create aggregated datasets 

		We collect mean and variance for each numerical measure, 
		and we combine all of the text descriptions

		Note: This kind of groupby results in a MultiIndex colum format (at least where we have [mean,var] aggs).  
		The code after the actual groupby flattens the column hierarchy.

		Note: username_gb parameter determines whether username gb df is grouped by posts or created_date.
		Analytical results may vary highly based on this choice. '''

	df['gb'] = {}
	
	if m == 'ig':
		post_unit = 'url'
		base = 'all'
		grouped = 'post'
	elif m == 'tw':
		post_unit = 'id'
		base = 'tweets'
		grouped = 'created_date'

	for gb_type in gb_types:
		''' Some groupbys act on the base (ie. un-grouped) data.  Others act on already-groupby'd data.
			Depending on whether we're in Twitter or Instagram, the base-acting groupbys differ. 
			See comments below for more. 

			IMPORTANT! Put username last in the ordering of the gb_type list. username gb relies on the fact that 
			something has already been grouped! 

			A note on Twitter Groupby:
			Initially, you made a groupby of the data based on user and time-unit, and then extracted word feats
			using Andy Reagan extractor code. The extraction code is in the Twitter eda file.  But since doing that,
			this function does the normal groupby, then grabs the word_features (which are already representative of
			grouped data, and have a time_unit field noting what kind of groupby each row is), and merges them with
			a few of the features (has_url, is_rt, is_reply) from the basic groupby. You could have just put those fields
			in the word_features table, but you decided not to because they are not really word features.'''

		if gb_type == 'post': # (instagram-only)
			gb_list = ['username', post_unit]
			to_group_df = df[base] # for instagram, this is the 'outermost' groupby we do, on original df

		elif (gb_type == 'created_date') and (m=='ig'):
			gb_list = ['username', 'created_date']
			to_group_df = df['gb'][grouped] # this groupby is acting on the gb-url aggregate df

		elif (gb_type == 'username') and (m=='ig'):
			gb_list = ['username']
			to_group_df = df['gb'][username_gb] # this groupby is acting on the gb-created_date aggregate df

		elif (gb_type == 'created_date') and (m=='tw'):
			gb_list = ['user_id', 'created_date']
			to_group_df = df[base] # this groupby is acting on the gb-url aggregate df

		elif gb_type == 'weekly': # (twitter only)
			gb_list = ['user_id', pd.Grouper(key='created_date',freq='W')]
			to_group_df = df[base] # this groupby is acting on the gb-url aggregate df

		elif (gb_type == 'user_id') and (m=='tw'):
			gb_list = ['user_id']
			to_group_df = df[base] # this groupby is acting on the gb-url aggregate df
			
		# testing
		#print 
		#print 'ROUND:', m, pop, gb_type, 'period:', period, 'turn:', turn_point
		#if gb_type == 'user_id':
		#	print 'before gb, unique user_id:', to_group_df.user_id.unique().shape[0]
		#print 'to_group_df shape:', to_group_df.shape
		#for col in to_group_df.columns:
		#	print col, to_group_df[col].dtype 
		#print to_group_df.columns
		#print 'agg cols:'
		#print params['agg_func'][m][gb_type]
		#print to_group_df.head()
		#print 

		df['gb'][gb_type] = (to_group_df.groupby(gb_list)
										.agg(params['agg_func'][m][gb_type])
							)
		# testing
		#if gb_type == 'user_id':
		#	print 'gb df shape:', df['gb'][gb_type].shape
			#print 'columns:', df['gb'][gb_type].columns
		#	print 'after gb, unique user_id:', df['gb'][gb_type].user_id.unique().shape[0]

		if m == 'ig':
			if additional_data:
				# collapses multiindex columns into |-separated names
				new_colname = ['%s%s' % (a, '|%s' % b if b else '') for a, b in df['gb'][gb_type].columns]
				
				df['gb'][gb_type].columns = new_colname
				# bring url into column set (instead of as row index)
				df['gb'][gb_type].reset_index(inplace=True)

				if 'created_date|first' in df['gb'][gb_type].columns:
					df['gb'][gb_type].rename(columns={'created_date|first':'created_date',
													  'one_word|join':'one_word',
													  'description|join':'description'}, inplace=True)
				for field in df['gb'][gb_type].columns:
					if re.search('mean|first|join|count|sum',field):
						df['gb'][gb_type].rename(columns={field:field.split("|")[0]}, inplace=True)

				# removes duplicate columns caused by reset_index()
				# http://stackoverflow.com/a/36513262/2799941
				Cols = list(df['gb'][gb_type].columns)
				for i,item in enumerate(df['gb'][gb_type].columns):
					if item in df['gb'][gb_type].columns[:i]: Cols[i] = "toDROP"
				df['gb'][gb_type].columns = Cols
				if "toDROP" in Cols:
					df['gb'][gb_type] = df['gb'][gb_type].drop("toDROP",1)

			add_class_indicator(df['gb'][gb_type], pop)

		elif m == 'tw':

			for gbel in gb_list:
				if gbel in df['gb'][gb_type].columns:
					df['gb'][gb_type].drop(gbel, 1, inplace=True)

			wf = get_word_feats(params, conn, pop, condition, tunit = gb_type)
			wf.created_date = pd.to_datetime(wf.created_date)
			if 'tweet_count' in wf.columns:
				wf.drop('tweet_count',1,inplace=True)

			df['gb'][gb_type].rename(columns={'id':'tweet_count'}, inplace=True)
			# bring url into column set (instead of as row index)
			df['gb'][gb_type].reset_index(inplace=True)

			merge_on = params['merge_on'][m][gb_type]
			fields_to_merge = params['fields_to_merge'][m][gb_type]

			#testing
			#print 'gb type:', gb_type
			#print 'merge on:', merge_on
			#print 'fields to merge:', fields_to_merge
			#print df['gb'][gb_type][fields_to_merge].shape 
			#print df['gb'][gb_type][fields_to_merge].columns 

			df['gb'][gb_type] = df['gb'][gb_type][fields_to_merge].merge(wf, on=merge_on, how='left')

		# testing:
			#print 'wf shape:', wf.shape 
			#print ('final gb shape [{} {} {} {} {}]:'.format(m, pop, gb_type, period, turn_point), 
					#df['gb'][gb_type].shape )
		
		# testing
		#print 'final gb columns:'
		#print df['gb'][gb_type].columns

	if doPrint:
		print 


def summary_stats(data, gb_type, level, additional_data):
	''' Prints summary statistics for each variable in model '''

	# level values: main, before_diag, before_susp

	if additional_data:
		varset = ['happy','sad','interesting','likable','hue','saturation','brightness','comment_count','like_count','url','has_filter','has_face','face_ct']
	else:
		varset = ['hue','saturation','brightness','comment_count','like_count','url','has_filter','has_face','face_ct']
	for v in varset:
		print 'Variable:', v.upper()

		for p, pval in [('target',1),('control',0)]:

			if level == 'main':
				df = data['master'][gb_type]
				mask = (df.target==pval).values
				df = df.ix[mask,:]
			else:
				when = level.split("_")[0]
				turn = 'from_{}'.format(level.split("_")[1])
				df = data['master'][when][turn][gb_type]
				df = df.ix[df.target==pval,:]

			mean = round(df[v].mean(),3)
			std = round(df[v].std(),3)
			print p.upper(), 'mean:', mean, '(SD={})'.format(std)
		print


def merge_to_master(master, target, control, m, varset, gb_type, additional_data, doPrint=False):
	''' merge target and control dfs into master df '''

	c = control[gb_type]
	t = target[gb_type]

	#subset = params['master_subset'][m][gb_type]
	master[gb_type] = pd.concat([c,t], axis=0)#.dropna(subset=subset)


	if gb_type == 'user_id':
		print 'unique CONTROL user_id:', c.user_id.unique().shape[0]
		print 'unique TARGET user_id:', t.user_id.unique().shape[0]
		print 'unique MASTER user_id:', master[gb_type].user_id.unique().shape[0]

	if additional_data:
		vlist = 'full'
	else:
		vlist = 'no_addtl_full'
	master[gb_type] = master[gb_type][ varset[m][gb_type][vlist] ]

	if doPrint:
		print 'Master {} {} nrows:'.format(m.upper(), gb_type.upper()), master[gb_type].shape[0]


def compare_density(df, m, gbtype, varset, additional_data, ncols=4):
	''' Overlays density plots of target vs control groups for selected aggregation type '''

	print 'target vs control for {} {}-groupby:'.format(m.upper(), gbtype.upper())
	plt.figure()
	gb = df[gbtype].groupby('target')
	if additional_data:
		vlist = 'means'
	else:
		vlist = 'no_addtl_means'
	numvars = float(len(varset[m][gbtype][vlist]))
	nrows = np.ceil(numvars/ncols).astype(int)
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 3*nrows), tight_layout=True)

	for ax, p in zip(axes.ravel(), varset[m][gbtype][vlist]):
		for k, v in gb[p]:
			try:
				sns.kdeplot(v, ax=ax, label=str(k)+":"+v.name)
			except Exception, e:
				print 'Error or all zeros for variable:', v.name
	plt.show()


def corr_plot(df, m, gb_type, varset, additional_data, print_corrmat=False):

	print 'Correlation matrix:'
	if additional_data:
		vlist = 'means'
	else:
		vlist = 'no_addtl_means'
	metrics = varset[m][gb_type][vlist]
	corr = df[gb_type][metrics].corr().round(2)
	corr.columns = [x.split("|")[0] for x in corr.columns]
	corr.index = [x.split("|")[0] for x in corr.index]

	# Generate a mask for the upper triangle
	mask = np.zeros_like(corr, dtype=np.bool)
	mask[np.triu_indices_from(mask)] = True

	plt.figure()
	# Set up the matplotlib figure
	f, ax = plt.subplots(figsize=(9, 7))

	# Generate a custom diverging colormap
	cmap = sns.diverging_palette(220, 10, as_cmap=True)

	# Draw the heatmap with the mask and correct aspect ratio
	sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0,
				square=True, linewidths=.5, 
				cbar_kws={"shrink": .7}, ax=ax)
	_=plt.title('Correlations among {} Variables'.format(m))
	plt.show()

	if print_corrmat:
		print corr 
	

def make_roc(name, clf, ytest, xtest, ax=None, labe=5, proba=True, skip=0):
	initial=False
	if not ax:
		ax=plt.gca()
		initial=True
	if proba:#for stuff like logistic regression
		fpr, tpr, thresholds=roc_curve(ytest, clf.predict_proba(xtest)[:,1])
	else:#for stuff like SVM
		fpr, tpr, thresholds=roc_curve(ytest, clf.decision_function(xtest))
	roc_auc = auc(fpr, tpr)
	if skip:
		l=fpr.shape[0]
		ax.plot(fpr[0:l:skip], tpr[0:l:skip], '.-', alpha=0.3, label='ROC curve for %s (area = %0.2f)' % (name, roc_auc))
	else:
		ax.plot(fpr, tpr, '.-', alpha=0.3, label='ROC curve for %s (area = %0.2f)' % (name, roc_auc))
	label_kwargs = {}
	label_kwargs['bbox'] = dict(
		boxstyle='round,pad=0.3', alpha=0.2,
	)
	if labe!=None:
		for k in xrange(0, fpr.shape[0],labe):
			#from https://gist.github.com/podshumok/c1d1c9394335d86255b8
			threshold = str(np.round(thresholds[k], 2))
			ax.annotate(threshold, (fpr[k], tpr[k]), **label_kwargs)
	if initial:
		ax.plot([0, 1], [0, 1], 'k--')
		ax.set_xlim([0.0, 1.0])
		ax.set_ylim([0.0, 1.05])
		ax.set_xlabel('False Positive Rate')
		ax.set_ylabel('True Positive Rate')
		ax.set_title('ROC')
	ax.legend(loc="lower right")
	return ax


def print_confusion_matrix(X_pred, y_test, ctype, labels=None):

	print
	print 'CONFUSION MATRIX ({}):'.format(ctype)
	
	if labels:
		# for confusion matrix
		known_0 = labels['known_0']
		known_1 = labels['known_1']
		pred_0 = labels['pred_0']
		pred_1 = labels['pred_1']
	else:
		known_0 = 'known_0'
		known_1 = 'known_1'
		pred_0 = 'pred_0'
		pred_1 = 'pred_1'

	cm_df = pd.DataFrame(confusion_matrix(y_test, X_pred), 
						 columns=[pred_0,pred_1], 
						 index=[known_0,known_1])
	print cm_df
	print 
	print 'Proportion of {} in {}:'.format(pred_1,known_0), round(cm_df.ix[known_0,pred_1] / float(cm_df.ix[known_0,:].sum()),3)
	print 'Proportion of {} in {}:'.format(pred_1,known_1), round(cm_df.ix[known_1,pred_1] / float(cm_df.ix[known_1,:].sum()),3)
	print
	print


def print_model_summary(fit, ctype, target, title, X_test, y_test, labels, average='binary', pos_label=1):
	''' formats model output in a notebook-friendly print layout '''

	print 'MODEL: {} ({}):'.format(fit['name'],title)
	print 'NAIVE ACCURACY:', round(fit['clf'].score(X_test,y_test),3)

	actual_neg = y_test==0
	preds = fit['clf'].predict(X_test)
	pred_neg = preds==0

	tn = np.sum(actual_neg & pred_neg)
	pneg = np.sum(pred_neg)
	neg = np.sum(actual_neg)
	print 'tn:', tn 
	print 'pneg:', pneg
	print 'neg:', neg 
	print 'NPV:', round( tn / float(pneg), 3)
	print 'SPECIFICITY:', round( tn / float(neg), 3 )
	print 'PRECISION (PPV):', round(precision_score(y_test, fit['clf'].predict(X_test), average=average), 3)
	print 'RECALL (SENSITIVITY):', round(recall_score(y_test, fit['clf'].predict(X_test), average=average), 3)
	print 'F1:', round(f1_score(y_test, fit['clf'].predict(X_test), average=average), 3)
	print 
	print 'Note: Weighted scores compute for each label, then weight-average together.'
	print 'See: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html'
	print_confusion_matrix(fit['clf'].predict(X_test), y_test, ctype, labels)


def roc_wrapper(fits, ctype, y_test, X_test, plat):
	
	if plat == 'ig':
		labe = 100
		skip = 1
	elif plat == 'tw':
		labe = 500
		skip = 5

	proba = False if ctype == 'svc' else True
		
	plt.figure()
	make_roc(fits[ctype]['name'], fits[ctype]['clf'], y_test, X_test, proba=proba, labe=labe, skip=skip)
	plt.show()


# this removes the leading zero (0.99 -> .99), used for pyplot formatting
def drop_leading_zero_formatter(x, pos):
	''' Format 1 as 1, 0 as 0, and all values whose absolute values is between
	0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
	formatted as -.4).
		Source: http://stackoverflow.com/questions/8555652/removing-leading-0-from-matplotlib-tick-label-formatting '''

	val_str = '{:g}'.format(x)
	if np.abs(x) > 0 and np.abs(x) < 1:
		return val_str.replace("0", "", 1)
	else:
		return val_str

	
def importance_wrapper(fits, ctype, model_feats, title, condition, tall_plot=False, imp_cutoff=.01, imp_subset=10):

	# replace "happs" with "happy" (eg. LabMT_happs -> LabMT_happy)
	model_feats = [re.sub('happs','happy',x) for x in model_feats]

	# Plot the feature importances of the forest
	fimpdf = pd.DataFrame(fits[ctype]['clf'].feature_importances_, index=model_feats, columns=['importance'])
	
	if tall_plot:
		fsize = (5,11)
	else:
		fsize = (3,4)
	plt.figure() 
	fimpdf = fimpdf.sort_values('importance', ascending=False).ix[fimpdf.importance > imp_cutoff,:]
	
	feat_names = fimpdf.index
	fimpdf.reset_index(drop=True)
	ax = fimpdf.ix[0:imp_subset,'importance'].plot(kind='barh', figsize=fsize, fontsize=14)
	plt.gca().invert_yaxis()
	major_formatter = FuncFormatter(drop_leading_zero_formatter)
	ax.xaxis.set_major_formatter(major_formatter)
	plt.xticks(fontsize=10)
	plt.title("Top {} predictors ({})".format(condition.upper(),title), fontsize=16)
	plt.show()


def cleanX(X, doPrint=False):
	''' Drops columns which are all NaN 
		Imputer should take care of most issues, this function is for when an entire column is NaN '''

	to_drop = X.columns[X.isnull().all()]
	if doPrint:
		print 'Columns to drop:', to_drop

	X.drop(to_drop,1,inplace=True)


def pca_explore(pca, X, unit, best_pca_num_comp, show_plot=False):
	''' Shows variance accounted for by principal components '''

	already_optimized = best_pca_num_comp
	X_reduced = pca.fit_transform(scale(X))
	n_comp = best_pca_num_comp

	if (not already_optimized):
		show_plot = True

	if show_plot:
		print 'Total vars:', X.shape[1]
		print 'Num components selected by Minka MLE:', pca.components_.shape[0]
		
		print 'Cumulative % variance explained per component:'
		cumvar = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
		print cumvar
		
		plt.figure()
		plt.plot(cumvar)
		_=plt.ylim([0,100])
		plt.title('PCA: Variance explained per additional component ({})'.format(unit))
		plt.show()

		if (not already_optimized):
			n_comp = pca.n_components_
		
	return X_reduced, n_comp


def pca_model(pca, X_reduced, y, num_pca_comp):
	''' Shows cross-validated F1 scores using PCA components in logistic regression 
		http://stats.stackexchange.com/questions/82050/principal-component-analysis-and-regression-in-python '''

	n = len(X_reduced)
	kf_10 = cross_validation.KFold(n, n_folds=10, shuffle=True)
	lr = logreg()
	f1 = [] # https://en.wikipedia.org/wiki/F1_score

	if not num_pca_comp:
		num_pca_comp = X_reduced.shape[1]
	
	score = cross_validation.cross_val_score(lr, np.ones((n,1)), y.ravel(), cv=kf_10, scoring='f1').mean()    
	f1.append(score) 
	
	for i in np.arange(1,num_pca_comp+1):
		print 'pca comp #:', i
		score = cross_validation.cross_val_score(lr, X_reduced[:,:i], y.ravel(), cv=kf_10, scoring='f1').mean()
		f1.append(score) 

	new_num_pca_comp = np.argmax(np.array(f1)) + 1 # redefine num components based on max F1 score
	print 'Num pca comp displayed:', num_pca_comp
	print 'Optimal number of components:', new_num_pca_comp # optimal num components based on max F1 score
	if new_num_pca_comp > num_pca_comp:
		print 'Optimal number based on F1 max exceeds Minka MLE...scaling back to Minka'
		new_num_pca_comp = num_pca_comp

	plt.figure()
	fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))
	ax1.plot(f1, '-v')
	ax2.plot(np.arange(1,num_pca_comp+1), f1[1:num_pca_comp+1], '-v')
	ax2.set_title('Intercept excluded from plot')

	for ax in fig.axes:
		ax.set_xlabel('Number of principal components in regression')
		ax.set_ylabel('F1')
		ax.set_xlim((-0.2,num_pca_comp+1))
	plt.show()
	
	return new_num_pca_comp


def pca_report(fit, feats, N_comp=3, N_elem=10):
	''' Prints N_elem components loadings for the top N_comp components.
		Three separate printouts are made, each is sorted with the highest N_elem loading variables per component on top
	'''
	loaddf = (pd.DataFrame(fit.components_.T[:,0:N_comp], 
						   columns=['PCA_{}'.format(x) for x in np.arange(N_comp)], 
						   index=feats)
			  .abs()
			  )
	for n in np.arange(N_comp):
		comp_ix = 'PCA_{}'.format(n)
		print loaddf.sort_values(comp_ix,ascending=False).ix[0:N_elem,comp_ix]
		print 

		
def update_acc_metrics(fit, X_test, y_test, cv_iters, acc_avg):
	''' Updates accuracy metrics for each round of model cross-validation '''
	
	probas_ = fit.predict_proba(X_test)
	actual_neg = y_test==0
	preds = fit.predict(X_test)
	pred_neg = preds==0
	tn = np.sum(actual_neg & pred_neg)
	pneg = np.sum(pred_neg)
	neg = np.sum(actual_neg)
	cv_iters['tn'].append(tn)
	cv_iters['pneg'].append(pneg)
	cv_iters['neg'].append(neg)
	cv_iters['npv'].append(round( tn / float(pneg), 3))
	cv_iters['specificity'].append(round( tn / float(neg), 3 ))
	cv_iters['precision'].append(round(precision_score(y_test, 
											   fit.predict(X_test), 
											   average=acc_avg), 3))
	cv_iters['recall'].append(round(recall_score(y_test, 
										 fit.predict(X_test), 
										 average=acc_avg), 3))
	cv_iters['f1'].append(round(f1_score(y_test, 
								 fit.predict(X_test), 
								 average=acc_avg), 3))
	return probas_
	

def plot_roc(y_test, probas_, mean_tpr, mean_fpr, condition, i=None, tinyfig=True, clf_name='Random Forests'):
	''' Plots ROC curve '''
	
	if i is not None:
		if (i == 0) and tinyfig:
			plt.figure(figsize=(4,4))
		# Compute ROC curve and area under the curve
		fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
		mean_tpr += interp(mean_fpr, fpr, tpr)
		mean_tpr[0] = 0.0
		roc_auc = auc(fpr, tpr)
		plt.plot(fpr, tpr, lw=1, label='ROC (area = {})'.format(round(roc_auc,2)))
	else:
		plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random chance')
		plt.xlim([-0.05, 1.05])
		plt.ylim([-0.05, 1.05])
		plt.xlabel('False Positive Rate', fontsize=14)
		plt.ylabel('True Positive Rate', fontsize=14)
		plt.xticks(fontsize=11)
		plt.title('{} ROC ({})'.format(clf_name, condition), fontsize=16)
		plt.legend(loc="lower right", fontsize=12)
		plt.show()
	
	
def initialize_model_fits(fits, kernel, rfp):
	''' Initializes machine learning model objects with specified hyperparameters '''

	fits['lr'] = {'name':'Logistic Regression','clf':logregcv(class_weight='auto')}
	fits['svc'] = {'name':'Support Vector Machine','clf':SVC(class_weight='auto', kernel=kernel, probability=True)}
	fits['rf'] = {'name':'Random Forests','clf':RFC(n_estimators=rfp['n_est'], 
													class_weight=rfp['class_wt'],
													max_features=rfp['max_feat'],
													min_samples_split=rfp['min_ss'],
													min_samples_leaf=rfp['min_sl'],
													max_depth=rfp['max_depth']
												   )}
	

def make_models(d, condition, clf_types=['lr','rf','svc'], 
				excluded_set=None, use_pca=False, stratify_split=False,
				labels={'known_0':'known_control',
						'known_1':'known_target',
						'pred_0':'pred_control',
						'pred_1':'pred_target'}):

	''' Makes, fits, and reports on machine learning models.  The ML workhorse in this script.
		NOTE: clf_types can include: 'lr' (logistic reg.),'rf' (random forests),'svc' (support vec.) 
		NOTE: set stratify_split = True to create stratified train/test splits '''

	mdata = d['data']
	title = d['name']
	unit = d['unit']
	target = d['target'] 
	feats = d['features']
	
	X = mdata[feats].copy()
	y = mdata[target]

	print 'Stratify split:', stratify_split
	if stratify_split:
		stratify = y
	else:
		stratify = None 

	cleanX(X)

	model_feats = X.columns

	imp = Imputer(strategy='median')
	X = imp.fit_transform(X,y)
	
	if use_pca:
		# stats.stackexchange.com/questions/82050/principal-component-analysis-and-regression-in-python
		# nxn.se/post/36838219245/loadings-with-scikit-learn-pca
		# stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ration-in-pca-with-sklearn
		
		pca = PCA(n_components='mle')
		X_reduced, num_pca_comp = pca_explore(pca, X, unit, d['best_pca_num_comp'], d['show_pca_comp_plot'])
		
		# IMPORTANT! Only need to run pca_model2 once to determine best num_pca_comp. Takes awhile to run (~20 min for tw).
		if d['best_pca_num_comp'] is None:
			num_pca_comp = pca_model(pca, X_reduced, y, num_pca_comp)
		else:
			num_pca_comp = d['best_pca_num_comp'] # passed in from notebook
			
		pca_report(pca, model_feats)

		# we do all subsequent modeling with the best PCA component vectors
		if num_pca_comp + 1 > X_reduced.shape[1]:
			max_ix = X_reduced.shape[1] 
		else:
			max_ix = num_pca_comp+1

		X = X_reduced[:,0:max_ix].copy() 
		model_feats = pd.Series(['pca_{}'.format(x) for x in np.arange(max_ix)])

		pca_df = pd.DataFrame(X, columns=model_feats)
		pca_df['target'] = y.values

	else:
		pca_df = None
	
	fits = {}
	initialize_model_fits(fits, d['kernel'], d['rf_params'])
	
	master_results = []
	
	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)
	ct = 0
	best_f1 = 0
	
	cv_len = 1
	
	master_results = {}
	cv_iters = {'tn':[], 'fn':[], 'pneg':[], 'neg':[], 'precision':[], 'specificity':[], 'recall':[], 'npv':[], 'f1':[] }
   
	for ctype in clf_types:
		for i in range(cv_len):
			
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=d['test_size'], stratify=stratify)

			fit = fits[ctype]['clf'].fit(X_train, y_train)

			probas_ = update_acc_metrics(fit, X_test, y_test, cv_iters, d['acc_avg'])
			plot_roc(y_test, probas_, mean_tpr, mean_fpr, condition, i=i)

		# compute mean accuracy metrics over cv iterations, print results
		for metric in ['tn','pneg','neg','npv','specificity','precision','recall','f1']:
			master_results[metric] = (round(np.mean(cv_iters[metric]),4), round(np.std(cv_iters[metric]),4))  
			print metric.upper(), '::', master_results[metric]
		
		plot_roc(y_test, probas_, mean_tpr, mean_fpr, condition)

		if use_pca:
			print 
			print 'NOTE: ALL MODELS ON THIS RUN USE PCA COMPONENTS!'
			print
		
		print 'UNIT OF OBSERVATION:', unit.upper()

		total_obs = X_test.shape[0]
		total_neg = np.sum(y_test==0)
		prop_neg = round((total_neg)/float(total_obs),3)

		output = {}
		output['X_test'] = X_test
		output['y_test'] = y_test

		output[ctype] = fits[ctype]['clf']

		if ctype == 'rf':
			importance_wrapper(fits, ctype, model_feats, title, condition,
							   d['tall_plot'], d['rf_params']['imp_cutoff'], d['rf_params']['imp_subset'])
	
		output[ctype] = fits[ctype]['clf']

		actualpos = np.sum(output['y_test'])
		prevalence = actualpos / float(output['y_test'].shape[0])

		print 'actualpos:', actualpos
		print 'total y_test:', output['y_test'].shape[0]
		print 'prevalence:', prevalence 
		print 
		#print_model_summary(fits[ctype], ctype, target, title, X_test, y_test, labels, average=acc_avg)

		out_of_100(prevalence, master_results['precision'][0], master_results['specificity'][0], master_results['recall'][0])
	
	return output, pca_df, best_f1, master_results
	

def ttest_output(a, b, varset, ttype, correction=True, alpha=0.05, method='bonferroni'):
	''' performs independent-samples t-tests with bonferroni correction '''

	pvals = []

	for metric in varset:
		#if metric[0:5] != 'LIWC': # this controls for 0s on happs scores and other vars that shouldn't have 0
		#	a = a.ix[a[metric] != 0, :]
		#	b = b.ix[b[metric] != 0, :]
		print
		print 'VARIABLE: {}'.format(metric)
		if ttype == 'ind':
			test = ttest(a[metric], b[metric])
		elif ttype == 'dep':
			test = ttest_rel(a[metric], b[metric])
		print 'A mean: {} (sd={})'.format(a[metric].mean(), a[metric].std())
		print 'B mean: {} (sd={})'.format(b[metric].mean(), b[metric].std())
		print
		print 't = {}, p = {}'.format(test.statistic,test.pvalue)
		print
		pvals.append( test.pvalue )
		print
		
	if correction:
		corrected = multipletests(pvals,alpha=alpha,method=method)
	
		print '{}-corrected alpha of {}:'.format(method,alpha),corrected[3]
		for i, metric in enumerate(varset):
			print '{} significant post-correction? {} ({})'.format(metric, corrected[0][i], corrected[1][i])
			
	return test, pvals


def ttest_wrapper(master, gb_type, varset, additional_data, split_var='target', ttype='ind'):
	''' formatting wrapper for ttest_output() '''

	print 'UNIT OF MEASUREMENT:', gb_type
	print

	a = master[gb_type].ix[master[gb_type][split_var]==0,:]
	b = master[gb_type].ix[master[gb_type][split_var]==1,:]
	if additional_data:
		vlist = 'means'
	else:
		vlist = 'no_addtl_means'
	return ttest_output(a, b, varset[gb_type][vlist], ttype)


def compare_filters(data, conn, level, gb_type, label, show_figs=True, show_neg_x=False):
	''' Chi2, plotting comparisons of Instagram filter use between target and control pops '''

	metaig = pd.read_sql_query('select username, filter, d_from_diag_depression as ddiag from meta_ig', conn)

	target_label = label.capitalize()

	if level == 'main':

		cids = data['control']['all'].username
		tids = data['target']['all'].username 

	else:

		when = level.split("_")[0]
		turn = 'from_{}'.format(level.split("_")[1])
		subdf = data['master'][when][turn][gb_type].reset_index()
		cids = subdf.username[subdf.target==0]
		tids = subdf.username[subdf.target==1]

	tfilt = metaig.ix[metaig.username.isin(tids),['username','filter']]
	cfilt = metaig.ix[metaig.username.isin(cids),['username','filter']]
	tfilt['target'] = 1
	cfilt['target'] = 0
	filts = pd.concat([tfilt,cfilt], axis=0)

	def get_prop(x, tsh = tfilt.shape[0], csh = cfilt.shape[0]):
		''' Gets proportion that a filter was used among all members of a given pop (target/control) 
			We thought to limit analysis based on minimum proportion, but ended up going with raw counts,
			so this function and filtprop df are not really necessary anymore. '''
		return pd.Series([round(x[0] / float(csh),4), round(x[(1)] / float(tsh),4)])

	filtct = filts.groupby(['target','filter']).count().unstack('target')
	filtct.columns = ['control','target']
	filtprop = filtct.apply(get_prop,axis=1)

	
	filtprop.columns = ['control','target']
	filtct_chi2 = filtct.ix[(filtct.control>=5) & (filtct.target>=5)] # we need >= 5 obs per cell for chi2

	print 'filters that missed the cut:',filtct.index[~filtct.index.isin(filtct_chi2.index)].astype(str).values

	no_filt = filts['filter']=="Normal"
	tm = filts.target==1
	filts['has_filter'] = True

	filts.ix[no_filt, 'has_filter'] = False

	print "Prevalence of filter usage in pops:"
	print

	print 'Target:'
	print "mean:", filts.ix[tm, 'has_filter'].mean()
	print "std:", filts.ix[tm, 'has_filter'].std()

	print

	print 'Control:'
	print "mean:", filts.ix[~tm, 'has_filter'].mean()
	print "std:", filts.ix[~tm, 'has_filter'].std()

	print

	#above1pct = filtprop.ix[(filtprop.target > 0.01)&(filtprop.control > 0.01),:].index
	#filtct_chi2 = filtct.ix[filtct.index.isin(above1pct),:]
	chi2 = stats.chi2_contingency(observed=filtct_chi2)
	filtct_chi2expect = pd.DataFrame(chi2[3], columns=['Healthy',target_label], index=filtct_chi2.index)
	filtct_chi2.rename(columns={'control':'Healthy','target':target_label}, inplace=True)
	filtct_chi2offset = filtct_chi2 - filtct_chi2expect 

	print 'chi2 stats comparing Instagram filters:'
	print 'chi2 value:',chi2[0]
	print 'p-value:',chi2[1]
	print 'deg.f.:',chi2[2]

	if show_figs:
		plt.figure()
		(filtct_chi2.sort_values(target_label,ascending=False)
					.plot(kind='bar', figsize=(16,8), fontsize=14,
						  color=[sns.xkcd_rgb['tomato'],sns.xkcd_rgb['french blue']])
		)
		plt.title('Instagram filter use', fontsize=18)
		plt.ylabel('Frequency',fontsize=14)
		plt.xlabel('Filter names', fontsize=14)
		if not show_neg_x:
			plt.ylim(ymin=0)

		plt.figure()
		(filtct_chi2offset.sort_values(target_label,ascending=True)
						  .plot(kind='bar', figsize=(16,8), fontsize=14, width=1,
								color=[sns.xkcd_rgb['tomato'],sns.xkcd_rgb['french blue']])
		)
		plt.title('Instagram filter use', fontsize=18)
		plt.ylabel('Usage difference (Chi^2 observed-expected)', fontsize=14)
		plt.xlabel('Filter names', fontsize=14)
		if not show_neg_x:
			plt.ylim(ymin=0)

		plt.figure()
		color_palette_mask = (filtct_chi2offset[target_label].values > 0)
		color_palette = ['gray' if case else 'blue' for case in color_palette_mask]
		df_notnorm = filtct_chi2offset.ix[filtct_chi2offset.index!='Normal',:].sort_values(target_label,ascending=True)
		
		ax = df_notnorm.plot(kind='bar', figsize=(16,8), fontsize=14,width=.8, align='center', 
							 color=[sns.xkcd_rgb['tomato'],sns.xkcd_rgb['french blue']])
		#df_notnorm.reset_index(inplace=True)
		#ax=sns.barplot(data=df_notnorm)
		plt.title('Instagram filter usage difference between label and healthy users', fontsize=20)
		plt.ylabel('Usage difference (Chi^2 observed-expected)', fontsize=18)
		plt.xlabel('Filter names', fontsize=18)
		plt.xticks(fontsize=14)
		ax.set_xticks(np.arange(filtct_chi2.index.shape[0]))
		plt.legend(fontsize=16)
		if not show_neg_x:
			plt.ylim(ymin=0)

	return filtct_chi2offset


def logreg_output(dm, resp, preds, doPrint=True, maxiter=100):
	''' Performs frequentist logistic regression, prints model stats, returns log odds '''
	
	logit = sm.Logit(resp, dm)
	# fit the model
	result = logit.fit(maxiter=maxiter)
	if doPrint:
		print result.summary()
		print 


	log_odds = zip(['intercept']+preds, result.params)
	
	# sorts by absolute magnitude of log odds
	sorted_log_odds = sorted([x for x in log_odds], key=lambda x: abs(x[1]), reverse=True)

	# converts to regular odds ratios
	sorted_odds_ratio = [(x[0], np.exp(x[1])) for x in sorted_log_odds]

	
	oddsdf = pd.DataFrame(sorted_odds_ratio, columns=['field','odds_ratio'])
	oddsdf['log_odds'] = [x[1] for x in sorted_log_odds]

	if doPrint:
		print 'Odds ratios (sorted by magnitude):'
		print oddsdf
	
	return log_odds, sorted_log_odds, sorted_odds_ratio


def logreg_wrapper(master, gb_type, vlist, varset, additional_data, scale_data=False, target='target', doPrint=True):
	''' formatting wrapper for logreg_output() 
		returns list of tuples: (predictor_name, log_odds) '''

	if additional_data == 'pca':
		predictors = [col for col in master.columns if col != 'target']
		response = master[target]
		df = master

	else:
		#print 'vars:', varset[gb_type][vlist]
		predictors = varset[gb_type][vlist] 
		response = master[gb_type][target]
		df = master[gb_type]

	if scale_data:
		# standardize X vars 
		# LR may not converge without this (especially for Twitter data)
		X = pd.DataFrame(scale(df[predictors]), columns=predictors)
		response.reset_index(drop=True, inplace=True)
	else:
		X = df[predictors]

	design_matrix = smtools.add_constant(X) # adds bias term, ie. vector of ones

	print 'UNIT OF MEASUREMENT:', gb_type
	print

	log_odds, sorted_log_odds, sorted_odds_ratio = logreg_output(design_matrix, response, predictors,
																 maxiter=100, doPrint=doPrint)
	return design_matrix, sorted_log_odds


def save_master_to_file(additional_data, posting_cutoff, use_pca, pca_num_comp, gb_type, report, condition, m, save_df):
	''' save csv of master data for target-type // groupby-type '''

	addtl = 'addtl-data' if additional_data else 'no-addtl-data'
	postcut = 'post-cut' if posting_cutoff else 'post-uncut'
	pca_status = 'pca-{}'.format(pca_num_comp) if use_pca else 'no-pca'
	fpath = '/'.join(['data-files',condition,m,gb_type])
	if m == 'tw':
		fname = '_'.join([condition,m,gb_type,report,pca_status])
	elif m == 'ig':
		fname = '_'.join([condition,m,gb_type,report,postcut,pca_status])
	csv_name = '{}/{}.csv'.format(fpath,fname)
	save_df.to_csv(csv_name, encoding='utf-8', index=False)


def master_actions(master, target, control, condition, m, params, gb_type, 
				   report, aparams, clfs, additional_data, posting_cutoff,
				   use_pca=False, scale_data=False):
	''' Performs range of actions on master data frame, including plotting, modeling, and saving to disk. 
		
		Note: "Master" may refer to any set of target+control data, including timeline subsets. 
		In other words, both the full dataset and subsets of full dataset may be passed as "master" argument.'''

	if aparams['create_master']:

		# merge target, control, into master
		print 
		print 'Merge to master: {} {}'.format(report, gb_type)
		merge_to_master(master, target, control, m, params['vars'], gb_type, additional_data)

		print 'master {} shape:'.format(gb_type), master[gb_type].shape
		print

		master['model'][gb_type] = {}
	
	if aparams['density']:
		compare_density(master, m, gb_type, params['vars'], additional_data)

	if aparams['corr']:
		corr_plot(master, m, gb_type, params['vars'], additional_data, aparams['print_corrmat'])

	if aparams['ml']:
		print 'Building ML models...'
		print 
		if additional_data:
			vlist = 'means'
		else:
			vlist = 'no_addtl_means'

		if aparams['rf_imp_cutoff']:
			imp_cutoff = aparams['rf_imp_cutoff']
		else:
			imp_cutoff = .015 # use .01 for instagram, .015 for twitter (depression)

		gb_type_report = 'daily bins' if gb_type == 'created_date' else '{} bins'.format(gb_type)
		
		model_df = {'name':'{}'.format(gb_type_report),
					'unit':gb_type,
					'data':master[gb_type],
					'features':params['vars'][m][gb_type][vlist],
					'target':'target',
					'platform':m,
					'test_size':.3, # test split for train/test
					'acc_avg':'binary',
					'best_pca_num_comp':aparams['best_pca'], 
					'show_pca_comp_plot':aparams['show_pca_comp_plot'],
					'kernel':'rbf', # for SVC, not used anymore
					'tall_plot':aparams['tall_plot'],
					'acc_avg':aparams['acc_avg'],
					'rf_params': {# params optimized with 5-fold CV, see optimize_rf_hyperparams()
								  'class_wt':'balanced',
								  'max_feat':'sqrt',
								  'n_est':aparams['rf_n_est'],
								  'min_ss':2,
								  'min_sl':1,
								  'max_depth':None,
								  'imp_cutoff':aparams['rf_imp_cutoff'],
								  'imp_subset':aparams['rf_imp_subset']} 
				   }

		output, pca_df, best_f1, master_results = make_models(model_df, condition=condition, clf_types=clfs, 
														   excluded_set=params['model_vars_excluded'][m][gb_type],
														   use_pca=use_pca, stratify_split=aparams['stratify'])

		for k in output.keys():
			master['model'][gb_type][k] = output[k]

	if aparams['save_to_file']:
		if use_pca:
			save_df = pca_df.copy()
			pca_ct = action_params['best_pca']+1
		else:
			save_df = master[gb_type].copy()
			pca_ct = None
			
		save_master_to_file(additional_data, posting_cutoff, use_pca, pca_ct, 
							gb_type, report, condition, m, save_df)

	if aparams['nhst']:

		if additional_data:
			vlist = 'means'
		else:
			vlist = 'no_addtl_means'

		print
		print 'LOGISTIC REGRESSION'
		
		_, master['model']['logodds'] = logreg_wrapper(master, gb_type, vlist, params['vars'][m],
													   additional_data, scale_data=scale_data)

		if aparams['use_ttest']:
			print
			print 'TTEST'
			ttest_out = ttest_wrapper(master, gb_type, params['vars'][m], additional_data)
			master['model'][gb_type]['ttest'] = ttest_out[0]
			master['model'][gb_type]['ttest_pvals'] = ttest_out[1]


def before_vs_after(df, gb_type, m, condition, varset, aparams, additional_data, ncols=4):
	for date in ['diag','susp']:
		print
		print 'before vs after (target: {}) for {}-groupby, based on {}_date:'.format(condition, gb_type, date)

		splitter = 'before_{}'.format(date)
		gb = df[gb_type].groupby(splitter)

		if additional_data:
			vlist = 'means'
		else:
			vlist = 'no_addtl_means'
		if aparams['density']:
			plt.figure()
			numvars = float(len(varset[gb_type][vlist]))
			nrows = np.ceil(numvars/ncols).astype(int)
			fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 3*nrows), tight_layout=True)

			for ax, p in zip(axes.ravel(), varset[gb_type][vlist]): # don't use gb_type here, it's only for url
				for k, v in gb[p]:
					sns.kdeplot(v, ax=ax, label=str(k)+":"+v.name)
			plt.show()

		if aparams['nhst']:
			ttest_wrapper(df, gb_type, varset, additional_data, split_var=splitter, ttype='ind')


def dictify(wordVec):
	'''Turn a word list into a word,count hash.'''
	thedict = dict()
	for word in wordVec:
		thedict[word] = 1
	return thedict

def listify(raw_text,lang="en"):
	"""Make a list of words from a string."""

	punctuation_to_replace = ["---","--","''"]
	for punctuation in punctuation_to_replace:
		raw_text = raw_text.replace(punctuation," ")
	words = [x.lower() for x in findall(r"[\w\@\#\'\&\]\*\-\/\[\=\;]+",raw_text,flags=UNICODE)]

	return words

my_LIWC_stopped = LIWC(stopVal=0.5)
my_LIWC = LIWC()
my_LabMT = LabMT(stopVal=1.0)
my_ANEW = ANEW(stopVal=1.0)


def all_features(data, tunit, condition):
	'''Return the feature vector for a given set of tweets'''

	rawtext = data['text']
	
	result = {}

	result['tweet_id'] = data['id'] if tunit == 'id' else None 
	result['user_id'] = data['user_id']
	result['created_date'] = None if tunit == 'user_id' else data['created_date'].strftime('%Y-%m-%d')
	result['time_unit'] = tunit
	
	for cond in ['depression','pregnancy','ptsd','cancer']:
		if cond == condition:
			result[cond] = data['target']
			result['no_{}'.format(cond)] = np.logical_not(data['target']).astype(int)
		else:
			result[cond] = ''
			result['no_{}'.format(cond)] = ''
			
	words = listify(rawtext)
	word_dict = dictify(words)
	result['total_words'] = len(words)

	# load the classes that we need
	my_word_vec = my_LIWC_stopped.wordVecify(word_dict)
	happs = my_LIWC_stopped.score(word_dict)
	result['LIWC_num_words'] = sum(my_word_vec)
	result['LIWC_happs'] = happs

	my_word_vec = my_LabMT.wordVecify(word_dict)
	happs = my_LabMT.score(word_dict)
	result['LabMT_num_words'] = sum(my_word_vec)
	result['LabMT_happs'] = happs
	my_word_vec = my_ANEW.wordVecify(word_dict)
	happs = my_ANEW.score(word_dict)

	result['ANEW_num_words'] = sum(my_word_vec)
	result['ANEW_happs'] = happs
	result['ANEW_arousal'] = my_ANEW.score(word_dict,idx=3)
	result['ANEW_dominance'] = my_ANEW.score(word_dict,idx=5)

	# make a word vector
	my_word_vec = my_LIWC.wordVecify(word_dict)
	all_features = zeros(len(my_LIWC.data["happy"])-2)
	liwc_names = ['LIWC_total_count','LIWC_funct','LIWC_pronoun','LIWC_ppron','LIWC_i','LIWC_we','LIWC_you','LIWC_shehe','LIWC_they','LIWC_ipron','LIWC_article','LIWC_verb','LIWC_auxverb','LIWC_past','LIWC_present','LIWC_future','LIWC_adverb','LIWC_preps','LIWC_conj','LIWC_negate','LIWC_quant','LIWC_number','LIWC_swear','LIWC_social','LIWC_family','LIWC_friend','LIWC_humans','LIWC_affect','LIWC_posemo','LIWC_negemo','LIWC_anx','LIWC_anger','LIWC_sad','LIWC_cogmech','LIWC_insight','LIWC_cause','LIWC_discrep','LIWC_tentat','LIWC_certain','LIWC_inhib','LIWC_incl','LIWC_excl','LIWC_percept','LIWC_see','LIWC_hear','LIWC_feel','LIWC_bio','LIWC_body','LIWC_health','LIWC_sexual','LIWC_ingest','LIWC_relativ','LIWC_motion','LIWC_space','LIWC_time','LIWC_work','LIWC_achieve','LIWC_leisure','LIWC_home','LIWC_money','LIWC_relig','LIWC_death','LIWC_assent','LIWC_nonfl','LIWC_filler'] 
	for word in my_LIWC.data:
		all_features += array(my_LIWC.data[word][2:])*my_word_vec[my_LIWC.data[word][0]]
	for i,score in enumerate(all_features):
		result[liwc_names[i]] = all_features[i]

	return pd.Series(result)


def create_word_feats_wrapper(pop_longs, gb_types, data, condition, conn, write_to_db, testing):
	''' Wraps iterators around create_word_feats for each type of pop/gb combo '''

	for pop_long in pop_longs:
		for tunit in gb_types:
			print 'In {} :: {}'.format(pop_long,tunit)
			create_word_feats(data[pop_long], tunit, condition, conn, write_to_db=True, testing=testing)


def create_word_feats(df, tunit, condition, conn, write_to_db=False, testing=False):
	
	if 'word_feats' not in df.keys():
		df['word_feats'] = {}
	
	if testing:
		return df['gb'][tunit].ix[0:3,:].apply(all_features, tunit=tunit, condition=condition, axis=1)
	else:
		df['word_feats'][tunit] = df['gb'][tunit].apply(all_features, tunit=tunit, condition=condition, axis=1)
		
	if write_to_db:
		print
		print "Writing to db..."
		table_fields = pd.read_sql_query('pragma table_info(word_features)',conn).name
		feats_to_write = [feat for feat in df['word_feats'][tunit].columns if feat in table_fields.values ]
		word_features = "(" + ",".join(feats_to_write) + ")"
		q = 'insert or ignore into word_features{} values ('.format(word_features) + ",".join(['?']*len(feats_to_write)) +")"
		vals = [tuple(x) for x in df['word_feats'][tunit].values]

		with conn:
			cur = conn.cursor()
			cur.executemany(q,vals)
			conn.commit()

def select_gmm(X):
	''' BIC selection of best number of mixture components.
		Code taken entirely from sklearn docs:
		http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#example-mixture-plot-gmm-selection-py
	'''
	lowest_bic = np.infty
	bic = []
	n_components_range = range(1, 7)
	cv_types = ['spherical', 'tied', 'diag', 'full']
	for cv_type in cv_types:
		for n_components in n_components_range:
			# Fit a mixture of Gaussians with EM
			gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type)
			gmm.fit(X)
			bic.append(gmm.bic(X))
			if bic[-1] < lowest_bic:
				lowest_bic = bic[-1]
				best_gmm = gmm

	bic = np.array(bic)
	color_iter = itertools.cycle(['gray', 'red', 'purple', 'blue'])
	clf = best_gmm
	bars = []

	plt.figure(figsize=(8,6))
	# Plot the BIC scores
	for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
		xpos = np.array(n_components_range) + .2 * (i - 2)
		bars.append(plt.bar(xpos, bic[i * len(n_components_range):
									  (i + 1) * len(n_components_range)],
							width=.2, color=color, alpha=0.7))
	plt.xticks(n_components_range)
	plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
	plt.title('BIC score per model')
	xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
		.2 * np.floor(bic.argmin() / len(n_components_range))
	plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
	plt.xlabel('Number of components')
	plt.legend([b[0] for b in bars], cv_types)
	plt.show()

	return best_gmm


def to_localtime_wrapper(conn):
	''' Converts GMT timestamps to user's localtime, returns df with username, timestamp, localtime info '''

	q = 'select id, created_at, tz from meta_tw'
	tzinfo = pd.read_sql_query(q, conn)
	tzinfo['local_timestamp'] = tzinfo.apply(to_localtime, axis=1)
	tups = [tuple([x[3],x[0]]) for x in tzinfo.values]
	return tzinfo, tups


def to_localtime(row):
	timestamp = row[1]
	tz = row[2]
	if tz is not None:
		timestamp = mktime_tz(parsedate_tz(timestamp))
		dt = datetime.datetime.fromtimestamp(timestamp, pytz.timezone(tz))
		s = dt.strftime('%Y-%m-%d %H:%M:%S')
	else:
		s = None
	return s


def reformat_diag_date(metatw):
	''' Somehow the diag_dates except for depression ended up without hyphens separating yyyymmdd '''
	conditions = ['pregnancy','cancer','ptsd'] # depression seems fine

	for cond in conditions:
		field = 'diag_date_{}'.format(cond)
		metatw.ix[metatw[field]!='',field] = metatw.ix[metatw[field]!='',field].apply(lambda x: x[0:4]+'-'+x[4:6]+'-'+x[6:])
	
def get_time_only(x):
	if x:
		return x.split()[1]
	else:
		return x

def get_local_time_data(metatw, tz):
	metatw['local_timestamp'] = tz.local_timestamp
	metatw['local_time'] = metatw.local_timestamp.apply(get_time_only)

def tag_insomnia(x):
	''' See De Choudhury et al (2013)...details are based on the metric they created '''
	
	insomnia_start = datetime.datetime.strptime("21:00:00","%H:%M:%S").time()
	insomnia_end = datetime.datetime.strptime("06:00:00","%H:%M:%S").time()
	if x:
		t = datetime.datetime.strptime(x,"%H:%M:%S").time()
		if (t > insomnia_start) or (t < insomnia_end):
			return 1
		else:
			return 0
	else:
		return None


def find_faces(row, pop, condition):
	url = row['url']
	path_url = url.split('//scontent.cdninstagram.com/')[1].replace('/','___') + ".jpg"
	savepath = '/'.join(["photos",condition,pop,"{}".format(path_url)])
	
	if not isfile(savepath):
		urllib.urlretrieve(url, savepath)
	
	row['has_face'], row['face_ct'] = face_detect(savepath)

	return row


def face_detect(img):
	''' Detects faces in a photograph using Haar cascades '''
	''' Modified from source: https://gist.github.com/dannguyen/cfa2fb49b28c82a1068f '''
	
	# first argument is the haarcascades path
	face_cascade_path = "../../haarcascade_frontalface_default.xml"
	face_cascade = cv2.CascadeClassifier(face_cascade_path)

	# profiles didn't work better than default (and also didn't catch things default missed)
	#profile_cascade_path = "../../haarcascade_profileface.xml"
	#profile_cascade = cv2.CascadeClassifier(profile_cascade_path)

	scale_factors = [1.05, 1.4] # you played around and found these two runs to be the best
	min_neighbors = 4 # you experimented between 1 and 5 here, 4 seemed best.
	min_size = (20,20) # don't go higher than this, many tiny faces
	flags = cv2.CASCADE_SCALE_IMAGE

	image = cv2.imread(img)

	found_face = False
	face_ct = 0
	
	for scale in scale_factors:
		if not found_face:
			scale_factor = scale
			for cascade, view in [(face_cascade,'straight')]:#[(face_cascade,'straight'),(profile_cascade,'profile')]:
				faces = cascade.detectMultiScale(image, 
												scaleFactor = scale_factor, 
												minNeighbors = min_neighbors,
												minSize = min_size, 
												flags = flags)
				if len(faces) != 0:
					found_face = True
					face_ct += len(faces)

					for ( x, y, w, h ) in faces:
						cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
						savedir = '/'.join(['photos',condition,pop,'detected',view])
						savefname = img.split('/')[-1] + ".jpg"
						savepath = join(savedir,savefname)
						cv2.imwrite(savepath, image)
					
	return(found_face, face_ct)            


def show_photo(url, pop, pred_face, doPrint=True):
	''' Displays photo in Jupyter notebook for face assessment '''
	
	if pred_face:
		path_head = "photos/depression/{}/detected/straight".format(pop)
	else:
		path_head = "photos/depression/{}".format(pop)
		
	new_fname = url.split('//scontent.cdninstagram.com/')[1].replace('/','___') + ".jpg"
	
	if doPrint:
		print new_fname
	
	path = join(path_head, new_fname)
	
	try:
		display(Image(filename=path))
	except:
		display(Image(filename=path+'.jpg'))


def get_face_stats(subset, when='before', gb_type='created_date'):
	''' Reports descriptive stats for face detection in given population subset '''
	
	if subset == 'main':
		aset = data['master'][gb_type]
		subdf = 'main'
	else:
		subdf = "from_{}".format(subset)
		col = "{}_{}".format(when, subset)
		turn = '{}date'.format(subset)

		aset = data['master'][when][subdf][gb_type].reset_index()

		
	# masks for all depression (target + control)
	t_mask = aset.target==1
	c_mask = aset.target==0
	
	print 'face ct avg (target):', aset.ix[t_mask,'face_ct'].mean()
	print 'face ct std (target):', aset.ix[t_mask,'face_ct'].std()
	print
	print 'face ct avg (control):', aset.ix[c_mask,'face_ct'].mean()
	print 'face ct std (control):', aset.ix[c_mask,'face_ct'].std()
	
	
	#aset = a.ix[a.username.isin(bdate.username) & (a[turn].isnull() | (a[turn]<0)),:]

	print
	print 'For all data (not just validation samples...):'
	print '{} set size: {}'.format(subdf,aset.shape[0])
	print 'target set size:', aset.ix[t_mask,:].shape[0]
	print 'control set size:', aset.ix[c_mask,:].shape[0]
	print
	print
	
	print 'Prop HAS FACE for TARGET:', round(aset.ix[t_mask, 'has_face'].mean(), 3)
	print 'Std prop HAS FACE for TARGET:', round(aset.ix[t_mask, 'has_face'].std(), 3)
	print
	print 'Prop HAS FACE for CONTROL:', round(aset.ix[c_mask, 'has_face'].mean(), 3)
	print 'Std prop HAS FACE for CONTROL:', round(aset.ix[c_mask, 'has_face'].std(), 3)

	print
	print

	print 'Considering all photos with at least one face...'
	print
	print 'Mean FACE CT for TARGET:', round(aset.ix[t_mask & (aset.face_ct>0), 'face_ct'].mean(), 3)
	print 'STD FACE CT for TARGET:', round(aset.ix[t_mask & (aset.face_ct>0), 'face_ct'].std(), 3)
	print
	print 'Mean FACE CT for CONTROL:', round(aset.ix[c_mask & (aset.face_ct>0), 'face_ct'].mean(), 3)
	print 'STD FACE CT for CONTROL:', round(aset.ix[c_mask & (aset.face_ct>0), 'face_ct'].std(), 3)

	print
	print

	print 'ttest for has_face:'
	tout = ttest(aset.ix[t_mask, 'has_face'], aset.ix[c_mask, 'has_face'])
	print 't = {}, p = {}'.format(tout.statistic, tout.pvalue)
	print

	print 'ttest for face_ct:'
	tout = ttest(aset.ix[t_mask & (aset.face_ct>0), 'face_ct'], aset.ix[c_mask & (aset.face_ct>0), 'face_ct'])
	print 't = {}, p = {}'.format(tout.statistic, tout.pvalue)
	
	print 'ttest for face_ct all photos:'
	tout = ttest(aset.ix[t_mask, 'face_ct'], aset.ix[c_mask, 'face_ct'])
	print 't = {}, p = {}'.format(tout.statistic, tout.pvalue)


def optimize_rf_hyperparams(X, y):
	''' Optimizes Random Forest hyperparameters. 

		WARNING: Takes 50+ hours to run!!!  (2160 different configurations)

		Uses suggestions from: http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur '''
	
	labels={'known_0':'known_control',
						'known_1':'known_target',
						'pred_0':'pred_control',
						'pred_1':'pred_target'}
	target = 'target'
	title = 'rf hyperparam optimization'
	acc_avg = 'binary'
	
	fits = {}
	ctype = 'rf'
	unit = 'created_date'
	
	cv = StratifiedKFold(y, n_folds=5)

	n_ests = [120, 300, 500, 800, 1200]
	max_depths = [5, 8, 15, 25, 30, None]
	min_samples_splits = [1, 2, 5, 10, 15, 100]
	min_samples_leaf = [1, 2, 5, 10]
	max_features = ['log2','sqrt',None]
	
	master_results = []
	
	ct = 0
	best_f1 = 0
	
	for max_feat in max_features:
		for n_est in n_ests:
			for max_depth in max_depths:
				for min_ss in min_samples_splits:
					for min_sl in min_samples_leaf:
						
						if ct % 10 == 0:
							print 'ROUND:', ct
						
						master_results.append({'max_feat':max_feat,
												'n_est':n_est,
												'max_depth':max_depth,
												'min_ss':min_ss,
												'min_sl':min_sl
											  }
											 )
  
						
						fits['rf'] = {'name':'Random Forests',
									  'clf':RFC(n_estimators=n_est, 
												class_weight='balanced',
												max_features=max_feat,
												min_samples_split=min_ss,
												min_samples_leaf=min_sl,
												max_depth=max_depth)}
						
						
						mean_tpr = 0.0
						mean_fpr = np.linspace(0, 1, 100)
						cv_iters = {'tn':[],
									'fn':[],
									'pneg':[],
									'neg':[],
									'precision':[],
									'specificity':[],
									'recall':[],
									'npv':[],
									'f1':[]
								   }
						
						for i, (train, test) in enumerate(cv):
							fit = fits[ctype]['clf'].fit(X[train], y[train])
							probas_ = fit.predict_proba(X[test])

							actual_neg = y[test]==0
							preds = fit.predict(X[test])
							pred_neg = preds==0

							tn = np.sum(actual_neg & pred_neg)
							pneg = np.sum(pred_neg)
							neg = np.sum(actual_neg)
							cv_iters['tn'].append(tn)
							cv_iters['pneg'].append(pneg)
							cv_iters['neg'].append(neg)
							cv_iters['npv'].append(round( tn / float(pneg), 3))
							cv_iters['specificity'].append(round( tn / float(neg), 3 ))
							cv_iters['precision'].append(round(precision_score(y[test], 
																	   fit.predict(X[test]), 
																	   average=acc_avg), 3))
							cv_iters['recall'].append(round(recall_score(y[test], 
																 fit.predict(X[test]), 
																 average=acc_avg), 3))
							cv_iters['f1'].append(round(f1_score(y[test], 
														 fit.predict(X[test]), 
														 average=acc_avg), 3))
							
						for metric in ['tn','pneg','neg','npv','specificity','precision','recall','f1']:
							
							master_results[ct][metric] = np.mean(cv_iters[metric])
						
						if master_results[ct]['f1'] > best_f1:
							best_f1 = master_results[ct]['f1']
							print 'ALERT new best f1 = {} at params :: '.format(best_f1)
							print 'max_feat:',max_feat, 'n_est:',n_est, 'max_depth:',max_depth, 'min_ss:',min_ss, 'min_sl',min_sl
							print    
						
						ct += 1

	return master_results, best_f1


def fit_hmm(df, preds, K=2, show_hist=True,
			sort_cols=['user_id','created_date'], gb_col='user_id', ct_col='target'):
	''' Fits K-state Hidden Markov Model on sorted data, returns class probabilities'''
	
	# sort data by user and then by posting chronology
	hmmdf = df.sort_values(sort_cols).copy()

	# HMM needs to know how many posts are in the chronology for each user
	# NB: ct_col here is not special, we can use any column that count() turns into a count vector.
	lengths = hmmdf.groupby(gb_col).count()[ct_col].values

	# see Working with Multiple Sequences for more on the lengths argument
	# http://hmmlearn.readthedocs.io/en/latest/tutorial.html#training-hmm-parameters-and-inferring-the-hidden-states
	hmm = GaussianHMM(n_components=K).fit( hmmdf[preds], lengths )

	probas = hmm.predict_proba( hmmdf[preds] )
	
	for i in range(K):
		hmmdf['proba{}'.format(i)] = probas[:,i]

	if show_hist:
		which_prob = 1
		plt.figure(figsize=(6,3))
		_=plt.hist(probas[:,which_prob], bins=100)
		plt.yscale('log')
		plt.title('Probability histogram for State {}'.format(which_prob))
		
	return hmm, hmmdf.reset_index(drop=True)


def show_class_diffs(hmm, hmmdf, predictors, to_show=10, show_neg=True, diff_colname='State1-State0'):
	''' Shows difference of parameter means between HMM State 1 and State 2.  For 2-state HMM only. 
		Note: State 0 for a given fit may be State 1 for a subsequent fit! '''

	# the zeros column is a placeholder for State1-State0, which you create using apply() 
	hmm_diff = pd.DataFrame(np.array([xrange(hmmdf[predictors].columns.shape[0]), 
									  np.zeros(hmmdf[predictors].shape[1])]    ).T, 
							index = hmmdf[predictors].columns, 
							columns = ['col_idx',diff_colname])

	hmm_diff.col_idx = hmm_diff.col_idx.astype(int)
	
	hmm_diff = hmm_diff.apply(lambda x: np.array([int(x[0]), hmm.means_[1,x[0]] - hmm.means_[0,x[0]]]), axis=1)
	hmm_diff = pd.DataFrame(hmm_diff)
	
	print 'Top {} differences between State 1 and State 0'.format(to_show)
	print
	return hmm_diff.sort_values(diff_colname, ascending=show_neg).iloc[0:to_show]


def build_comparison_data(compare_source, data, platform, gb_type, means, varset, additional_data):
	''' Builds comparison data points to evaluate whether HMM latent states map onto response variable states 
		Uses either logistic regression coefficients or mean differences between target/control classes. '''
	
	if compare_source == 'logistic regression':
		
		# we use logreg coefficients (log odds) to compare against HMM predictor means, as a means of labeling states 0/1
		dm, log_odds = logreg_wrapper(data['master'], gb_type, means, 
									  varset, 
									  additional_data, doPrint=False)
		compare_data = [x for x in log_odds if x[0] != 'intercept']
	
	elif compare_source == 'raw means':
		
		preds = varset[gb_type][means]
		df = data['master'][gb_type][preds]
		df['target'] = data['master'][gb_type].target
		compare_data = []
		for field in preds:
			compare_data.append( (field, df.ix[df.target==1, field].mean() - df.ix[df.target==0, field].mean()) )

	return compare_data
	

def compare_hmm_means(hmm, hmmdf, cols, compare_source, state=0, decision=0.5, K=2, reporting=True):
	''' T-tests to determine if State1 and State0 means are different for obs. assigned to each state by HMM 
		Kwargs: state -> which HMM state number to split on, decision -> cut point for state membership '''
	
	colnames = np.array([x[0] for x in cols]) # gets predictor names for np.where comparison later
	antistate = int(np.logical_not(state)) # sets int value for the other state in 2-state HMM (compared to 'state' arg)
	ct = {0:0,1:0} # for keeping track of which state is more in line with logistic regression coefficients
	
	for col, sway in cols:
		
		var_idx = np.where(colnames==col)[0][0]
		impact = 'larger' if sway > 0 else 'smaller'
		state_col = 'proba{}'.format(state)
		masked_var = {}
		state_likely = hmmdf[state_col] >= decision
		masked_var[state] = hmmdf[col][ state_likely ] # observations probably in state (default state = 0)
		masked_var[antistate] = hmmdf[col][ ~state_likely ] # observations probably in antistate (default = 1)

		try:
			
			hmm_means = {0:hmm.means_[0,var_idx],
						 1:hmm.means_[1,var_idx]}

			test = ttest(masked_var[state], masked_var[antistate])

			if reporting:
				print 'Comparison for variable: {}'.format(col.upper())
				for i in range(K):
					print 'State {} HMM {} mean: {} (sd={}) [HMM param. mean: {}]'.format(i, col, 
																						 round(masked_var[i].mean(), 3), 
																						 round(masked_var[i].std(), 3),
																						 round(hmm_means[i],3))
				print
				print 't = {}, p = {}'.format(test.statistic,test.pvalue)
				print
				print 'According to logistic regression, this variable should be {} for affected class.'.format(impact.upper())
				print
			diff = hmm_means[state].mean() - hmm_means[antistate].mean()
			if (diff < 1) and (sway < 0):
				ct[state] += 1
			else:
				ct[antistate] += 1
		except Exception, e:
			print 'Comparison of {} failed: {}'.format(col, str(e))
			
	# we assign "target_state" as the state which is most in agreement with logistic regression output
	if ct[state] > ct[antistate]:
		target_state = state
	else:
		target_state = antistate
	
	print 'state=0 agree ct: {}, state=1 agree ct: {}'.format(ct[state],ct[antistate])
	print
	print 'target_state, as determined by HMM agreement with {}:'.format(compare_source), target_state
	
	return target_state


def prepare_hmm_plot_data(hmmdf, hmm_master, key_var, klass, 
						  date_type='diag', offset=365, roll=90, doPrint=False):
	
	ct = 0
	color_ct = 0
	uct = 0
	last_uid = ''
	
	date_field = '{}_date'.format(date_type)
	from_field = 'from_{}'.format(date_type)

	hmm_master[klass] = pd.DataFrame()

	if klass == 1:
		# notnull() condition because some ptsd/preg will not have event_date
		mask = (hmmdf.target==klass) & (hmmdf[date_field].notnull())
	else:
		mask = (hmmdf.target==klass)
		
	for idx in hmmdf.loc[mask].index:

		uid = hmmdf.ix[idx,'user_id']
		
		if uid != last_uid:
			ct += 1
			last_uid = uid
			# for healthy class, we need to set a date range since there's no diagnosis/event date
			# we end at current_date_str and go back 2*offset
			current_date_str = '2016-01-01'
			current = pd.to_datetime(current_date_str)
			
			diag = pd.to_datetime(hmmdf.ix[idx,date_field])
			hmm_oneuser = hmmdf.ix[hmmdf.user_id==uid, :].copy()
			
			if klass == 1: #target
				
				ts = hmm_oneuser.ix[:,[key_var,from_field]].copy()
				ts.index = pd.to_datetime(hmm_oneuser.created_date)
				ts['from_point'] = ts[from_field]
				mask = (ts.index > diag-pd.DateOffset(offset)) & (ts.index < diag+pd.DateOffset(offset))
				
			else: #healthy
				
				ts = hmm_oneuser.ix[:,[key_var,'created_date']].copy()
				ts.index = pd.to_datetime(hmm_oneuser.created_date)
				ts['from_point'] = (ts.index-current).days
				# target class goes offset-many-days forw and back from diag_date, so we double offset back here
				mask = (ts.from_point > offset*-2) 
 
			ts['rmean'] = ts[key_var].rolling(roll).mean()
			ts2 = ts.loc[mask]
		
			if klass == 0:
				ts2.from_point = ts2.from_point + offset # cosmetic offset to match up with target class x-axis
				
			hmm_master[klass] = pd.concat([hmm_master[klass],ts2])

	if doPrint:
		hmm_master[klass][key_var].describe()



''' OLD REUSABLE CODE

# for if we need to reset word_features table 
q = 'drop table word_features'
cur = conn.cursor()
cur.execute(q)
conn.commit()

q = 'CREATE TABLE word_features(table_id INTEGER PRIMARY KEY, tweet_id INT, user_id INT, created_date TEXT, total_words INT, LIWC_num_words INT, LIWC_happs REAL, LabMT_num_words INT, LabMT_happs REAL, ANEW_num_words INT, ANEW_happs REAL, ANEW_arousal REAL, ANEW_dominance REAL, LIWC_total_count INT, LIWC_funct INT, LIWC_pronoun INT, LIWC_ppron INT, LIWC_i INT, LIWC_we INT, LIWC_you INT, LIWC_shehe INT, LIWC_they INT, LIWC_ipron INT, LIWC_article INT, LIWC_verb INT, LIWC_auxverb INT, LIWC_past INT, LIWC_present INT, LIWC_future INT, LIWC_adverb INT, LIWC_preps INT, LIWC_conj INT, LIWC_negate INT, LIWC_quant INT, LIWC_number INT, LIWC_swear INT, LIWC_social INT, LIWC_family INT, LIWC_friend INT, LIWC_humans INT, LIWC_affect INT, LIWC_posemo INT, LIWC_negemo INT, LIWC_anx INT, LIWC_anger INT, LIWC_sad INT, LIWC_cogmech INT, LIWC_insight INT, LIWC_cause INT, LIWC_discrep INT, LIWC_tentat INT, LIWC_certain INT, LIWC_inhib INT, LIWC_incl INT, LIWC_excl INT, LIWC_percept INT, LIWC_see INT, LIWC_hear INT, LIWC_feel INT, LIWC_bio INT, LIWC_body INT, LIWC_health INT, LIWC_sexual INT, LIWC_ingest INT, LIWC_relativ INT, LIWC_motion INT, LIWC_space INT, LIWC_time INT, LIWC_work INT, LIWC_achieve INT, LIWC_leisure INT, LIWC_home INT, LIWC_money INT, LIWC_relig INT, LIWC_death INT, LIWC_assent INT, LIWC_nonfl INT, LIWC_filler INT,depression INT,no_depression INT,pregnancy INT,no_pregnancy INT,ptsd INT,no_ptsd INT,cancer INT,no_cancer INT,time_unit TEXT, tweet_count INT)'
cur = conn.cursor()
cur.execute(q)
conn.commit()

# for resetting control entries in word_features
q = 'delete from word_features where depression=0'
with conn:
	cur = conn.cursor()
	cur.execute(q)
	conn.commit()

# write/read text features file
data['target']['word_feats']['weekly'].to_csv('tw_target_wordfeats_weekly.csv',index=False)
data['control']['word_feats']['weekly'].to_csv('tw_control_wordfeats_weekly.csv',index=False)

'''
	