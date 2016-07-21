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
import seaborn as sns
sns.set_style('white')

from skimage import io, data, color

from sklearn import cross_validation
from sklearn import mixture
from sklearn.preprocessing import scale
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression as logreg
from sklearn.linear_model import LogisticRegressionCV as logregcv
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, mean_squared_error
from sklearn.decomposition import PCA

import scipy.stats as stats
from scipy import linalg
from scipy.stats import ttest_ind as ttest
from scipy.stats import ttest_rel
from scipy.stats import pearsonr

import statsmodels.api as sm
from statsmodels.sandbox.stats.multicomp import multipletests
import statsmodels.api as sm
import statsmodels.tools.tools as smtools

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

	params = {
			'q': {
				't': {
					'meta':	'select {fields}, username, {plat_long}_user_id as user_id, created_date, diag_date_{cond} as diag_date, d_from_diag_{cond} as from_diag, d_from_susp_{cond} as from_susp from meta_{plat} where d_from_diag_{cond} is not null{ratings_clause}'.format(cond=condition,
																																																											  ratings_clause=ratings_clause,
																																																											  plat=platform,
																																																											  plat_long=platform_long,
																																																											  fields=fields),
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
				'all_meta':	'select {fields}, username, {plat_long}_user_id as user_id, created_date, diag_date_{cond} as diag_date, d_from_diag_{cond} as from_diag, d_from_susp_{cond} as from_susp from meta_{plat}'.format(cond=condition,
																																															 plat=platform,
																																															 plat_long=platform_long,
																																															 fields=fields),
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
					'created_date':{'id':		'count',
									'target':	'first',
									'has_url':	'mean',
									'is_rt':	'mean',
									'is_reply':	'mean',
									'from_diag':'first',
									'from_susp':'first',
									'text':		' '.join,
									'diag_date':'first',
									'user_id':	'first',
									'word_count':'mean',
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
						'means':['total_words',
								   'LIWC_num_words', 'LIWC_happs', 'LabMT_num_words', 'LabMT_happs',
								   'ANEW_num_words', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
								   'LIWC_total_count', 'LIWC_funct', 'LIWC_pronoun', 'LIWC_ppron',
								   'LIWC_i', 'LIWC_we', 'LIWC_you', 'LIWC_shehe', 'LIWC_they',
								   'LIWC_ipron', 'LIWC_article', 'LIWC_verb', 'LIWC_auxverb',
								   'LIWC_past', 'LIWC_present', 'LIWC_future', 'LIWC_adverb',
								   'LIWC_preps', 'LIWC_conj', 'LIWC_negate', 'LIWC_quant',
								   'LIWC_number', 'LIWC_swear', 'LIWC_social', 'LIWC_family',
								   'LIWC_friend', 'LIWC_humans', 'LIWC_affect', 'LIWC_posemo',
								   'LIWC_negemo', 'LIWC_anx', 'LIWC_anger', 'LIWC_sad', 'LIWC_cogmech',
								   'LIWC_insight', 'LIWC_cause', 'LIWC_discrep', 'LIWC_tentat',
								   'LIWC_certain', 'LIWC_inhib', 'LIWC_incl', 'LIWC_excl',
								   'LIWC_percept', 'LIWC_see', 'LIWC_hear', 'LIWC_feel', 'LIWC_bio',
								   'LIWC_body', 'LIWC_health', 'LIWC_sexual', 'LIWC_ingest',
								   'LIWC_relativ', 'LIWC_motion', 'LIWC_space', 'LIWC_time',
								   'LIWC_work', 'LIWC_achieve', 'LIWC_leisure', 'LIWC_home',
								   'LIWC_money', 'LIWC_relig', 'LIWC_death', 'LIWC_assent',
								   'LIWC_nonfl', 'tweet_count', 'word_count', 'has_url',
								   'is_rt', 'is_reply'],
						'model':['total_words',
								   'LIWC_num_words', 'LIWC_happs', 'LabMT_num_words', 'LabMT_happs',
								   'ANEW_num_words', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
								   'LIWC_total_count', 'LIWC_funct', 'LIWC_pronoun', 'LIWC_ppron',
								   'LIWC_i', 'LIWC_we', 'LIWC_you', 'LIWC_shehe', 'LIWC_they',
								   'LIWC_ipron', 'LIWC_article', 'LIWC_verb', 'LIWC_auxverb',
								   'LIWC_past', 'LIWC_present', 'LIWC_future', 'LIWC_adverb',
								   'LIWC_preps', 'LIWC_conj', 'LIWC_negate', 'LIWC_quant',
								   'LIWC_number', 'LIWC_swear', 'LIWC_social', 'LIWC_family',
								   'LIWC_friend', 'LIWC_humans', 'LIWC_affect', 'LIWC_posemo',
								   'LIWC_negemo', 'LIWC_anx', 'LIWC_anger', 'LIWC_sad', 'LIWC_cogmech',
								   'LIWC_insight', 'LIWC_cause', 'LIWC_discrep', 'LIWC_tentat',
								   'LIWC_certain', 'LIWC_inhib', 'LIWC_incl', 'LIWC_excl',
								   'LIWC_percept', 'LIWC_see', 'LIWC_hear', 'LIWC_feel', 'LIWC_bio',
								   'LIWC_body', 'LIWC_health', 'LIWC_sexual', 'LIWC_ingest',
								   'LIWC_relativ', 'LIWC_motion', 'LIWC_space', 'LIWC_time',
								   'LIWC_work', 'LIWC_achieve', 'LIWC_leisure', 'LIWC_home',
								   'LIWC_money', 'LIWC_relig', 'LIWC_death', 'LIWC_assent',
								   'LIWC_nonfl', 'tweet_count', 'word_count', 'has_url',
								   'is_rt', 'is_reply'],
						'full':	['tweet_id', 'user_id', 'total_words',
								   'LIWC_num_words', 'LIWC_happs', 'LabMT_num_words', 'LabMT_happs',
								   'ANEW_num_words', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
								   'LIWC_total_count', 'LIWC_funct', 'LIWC_pronoun', 'LIWC_ppron',
								   'LIWC_i', 'LIWC_we', 'LIWC_you', 'LIWC_shehe', 'LIWC_they',
								   'LIWC_ipron', 'LIWC_article', 'LIWC_verb', 'LIWC_auxverb',
								   'LIWC_past', 'LIWC_present', 'LIWC_future', 'LIWC_adverb',
								   'LIWC_preps', 'LIWC_conj', 'LIWC_negate', 'LIWC_quant',
								   'LIWC_number', 'LIWC_swear', 'LIWC_social', 'LIWC_family',
								   'LIWC_friend', 'LIWC_humans', 'LIWC_affect', 'LIWC_posemo',
								   'LIWC_negemo', 'LIWC_anx', 'LIWC_anger', 'LIWC_sad', 'LIWC_cogmech',
								   'LIWC_insight', 'LIWC_cause', 'LIWC_discrep', 'LIWC_tentat',
								   'LIWC_certain', 'LIWC_inhib', 'LIWC_incl', 'LIWC_excl',
								   'LIWC_percept', 'LIWC_see', 'LIWC_hear', 'LIWC_feel', 'LIWC_bio',
								   'LIWC_body', 'LIWC_health', 'LIWC_sexual', 'LIWC_ingest',
								   'LIWC_relativ', 'LIWC_motion', 'LIWC_space', 'LIWC_time',
								   'LIWC_work', 'LIWC_achieve', 'LIWC_leisure', 'LIWC_home',
								   'LIWC_money', 'LIWC_relig', 'LIWC_death', 'LIWC_assent',
								   'LIWC_nonfl', 'time_unit', 'tweet_count', 'word_count', 'has_url',
								   'is_rt', 'is_reply', 'target','before_diag','before_susp',
								   'created_date','diag_date']
						},
					'user_id':{
						'means':['LIWC_happs', 'LabMT_happs',
								   'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
								   'LIWC_total_count', 'LIWC_funct', 'LIWC_pronoun', 'LIWC_ppron',
								   'LIWC_i', 'LIWC_we', 'LIWC_you', 'LIWC_shehe', 'LIWC_they',
								   'LIWC_ipron', 'LIWC_article', 'LIWC_verb', 'LIWC_auxverb',
								   'LIWC_past', 'LIWC_present', 'LIWC_future', 'LIWC_adverb',
								   'LIWC_preps', 'LIWC_conj', 'LIWC_negate', 'LIWC_quant',
								   'LIWC_number', 'LIWC_swear', 'LIWC_social', 'LIWC_family',
								   'LIWC_friend', 'LIWC_humans', 'LIWC_affect', 'LIWC_posemo',
								   'LIWC_negemo', 'LIWC_anx', 'LIWC_anger', 'LIWC_sad', 'LIWC_cogmech',
								   'LIWC_insight', 'LIWC_cause', 'LIWC_discrep', 'LIWC_tentat',
								   'LIWC_certain', 'LIWC_inhib', 'LIWC_incl', 'LIWC_excl',
								   'LIWC_percept', 'LIWC_see', 'LIWC_hear', 'LIWC_feel', 'LIWC_bio',
								   'LIWC_body', 'LIWC_health', 'LIWC_sexual', 'LIWC_ingest',
								   'LIWC_relativ', 'LIWC_motion', 'LIWC_space', 'LIWC_time',
								   'LIWC_work', 'LIWC_achieve', 'LIWC_leisure', 'LIWC_home',
								   'LIWC_money', 'LIWC_relig', 'LIWC_death', 'LIWC_assent',
								   'LIWC_nonfl', 'tweet_count', 'word_count', 'has_url',
								   'is_rt', 'is_reply'],
						'model':['LIWC_happs', 'LabMT_happs',
								   'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
								   'LIWC_funct', 'LIWC_pronoun', 'LIWC_ppron',
								   'LIWC_i', 'LIWC_we', 'LIWC_you', 'LIWC_shehe', 'LIWC_they',
								   'LIWC_ipron', 'LIWC_article', 'LIWC_verb', 'LIWC_auxverb',
								   'LIWC_past', 'LIWC_present', 'LIWC_future', 'LIWC_adverb',
								   'LIWC_preps', 'LIWC_conj', 'LIWC_negate', 'LIWC_quant',
								   'LIWC_number', 'LIWC_swear', 'LIWC_social', 'LIWC_family',
								   'LIWC_friend', 'LIWC_humans', 'LIWC_affect', 'LIWC_posemo',
								   'LIWC_negemo', 'LIWC_anx', 'LIWC_anger', 'LIWC_sad', 'LIWC_cogmech',
								   'LIWC_insight', 'LIWC_cause', 'LIWC_discrep', 'LIWC_tentat',
								   'LIWC_certain', 'LIWC_inhib', 'LIWC_incl', 'LIWC_excl',
								   'LIWC_percept', 'LIWC_see', 'LIWC_hear', 'LIWC_feel', 'LIWC_bio',
								   'LIWC_body', 'LIWC_health', 'LIWC_sexual', 'LIWC_ingest',
								   'LIWC_relativ', 'LIWC_motion', 'LIWC_space', 'LIWC_time',
								   'LIWC_work', 'LIWC_achieve', 'LIWC_leisure', 'LIWC_home',
								   'LIWC_money', 'LIWC_relig', 'LIWC_death', 'LIWC_assent',
								   'LIWC_nonfl', 'word_count', 'has_url',
								   'is_rt', 'is_reply'],
						'full':	['tweet_id', 'user_id', 
								   'LIWC_happs', 'LabMT_happs',
								   'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
								   'LIWC_total_count', 'LIWC_funct', 'LIWC_pronoun', 'LIWC_ppron',
								   'LIWC_i', 'LIWC_we', 'LIWC_you', 'LIWC_shehe', 'LIWC_they',
								   'LIWC_ipron', 'LIWC_article', 'LIWC_verb', 'LIWC_auxverb',
								   'LIWC_past', 'LIWC_present', 'LIWC_future', 'LIWC_adverb',
								   'LIWC_preps', 'LIWC_conj', 'LIWC_negate', 'LIWC_quant',
								   'LIWC_number', 'LIWC_swear', 'LIWC_social', 'LIWC_family',
								   'LIWC_friend', 'LIWC_humans', 'LIWC_affect', 'LIWC_posemo',
								   'LIWC_negemo', 'LIWC_anx', 'LIWC_anger', 'LIWC_sad', 'LIWC_cogmech',
								   'LIWC_insight', 'LIWC_cause', 'LIWC_discrep', 'LIWC_tentat',
								   'LIWC_certain', 'LIWC_inhib', 'LIWC_incl', 'LIWC_excl',
								   'LIWC_percept', 'LIWC_see', 'LIWC_hear', 'LIWC_feel', 'LIWC_bio',
								   'LIWC_body', 'LIWC_health', 'LIWC_sexual', 'LIWC_ingest',
								   'LIWC_relativ', 'LIWC_motion', 'LIWC_space', 'LIWC_time',
								   'LIWC_work', 'LIWC_achieve', 'LIWC_leisure', 'LIWC_home',
								   'LIWC_money', 'LIWC_relig', 'LIWC_death', 'LIWC_assent',
								   'LIWC_nonfl', 'time_unit', 'tweet_count', 'word_count', 'has_url',
								   'is_rt', 'is_reply', 'target']
						},
					'created_date':{
						'means':['total_words','LIWC_num_words', 'LIWC_happs', 'LabMT_num_words', 'LabMT_happs',
								   'ANEW_num_words', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
								   'LIWC_total_count', 'LIWC_funct', 'LIWC_pronoun', 'LIWC_ppron',
								   'LIWC_i', 'LIWC_we', 'LIWC_you', 'LIWC_shehe', 'LIWC_they',
								   'LIWC_ipron', 'LIWC_article', 'LIWC_verb', 'LIWC_auxverb',
								   'LIWC_past', 'LIWC_present', 'LIWC_future', 'LIWC_adverb',
								   'LIWC_preps', 'LIWC_conj', 'LIWC_negate', 'LIWC_quant',
								   'LIWC_number', 'LIWC_swear', 'LIWC_social', 'LIWC_family',
								   'LIWC_friend', 'LIWC_humans', 'LIWC_affect', 'LIWC_posemo',
								   'LIWC_negemo', 'LIWC_anx', 'LIWC_anger', 'LIWC_sad', 'LIWC_cogmech',
								   'LIWC_insight', 'LIWC_cause', 'LIWC_discrep', 'LIWC_tentat',
								   'LIWC_certain', 'LIWC_inhib', 'LIWC_incl', 'LIWC_excl',
								   'LIWC_percept', 'LIWC_see', 'LIWC_hear', 'LIWC_feel', 'LIWC_bio',
								   'LIWC_body', 'LIWC_health', 'LIWC_sexual', 'LIWC_ingest',
								   'LIWC_relativ', 'LIWC_motion', 'LIWC_space', 'LIWC_time',
								   'LIWC_work', 'LIWC_achieve', 'LIWC_leisure', 'LIWC_home',
								   'LIWC_money', 'LIWC_relig', 'LIWC_death', 'LIWC_assent',
								   'LIWC_nonfl', 'tweet_count', 'word_count', 'has_url',
								   'is_rt', 'is_reply'],
						'model':['total_words','LIWC_num_words', 'LIWC_happs', 'LabMT_num_words', 'LabMT_happs',
								   'ANEW_num_words', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
								   'LIWC_total_count', 'LIWC_funct', 'LIWC_pronoun', 'LIWC_ppron',
								   'LIWC_i', 'LIWC_we', 'LIWC_you', 'LIWC_shehe', 'LIWC_they',
								   'LIWC_ipron', 'LIWC_article', 'LIWC_verb', 'LIWC_auxverb',
								   'LIWC_past', 'LIWC_present', 'LIWC_future', 'LIWC_adverb',
								   'LIWC_preps', 'LIWC_conj', 'LIWC_negate', 'LIWC_quant',
								   'LIWC_number', 'LIWC_swear', 'LIWC_social', 'LIWC_family',
								   'LIWC_friend', 'LIWC_humans', 'LIWC_affect', 'LIWC_posemo',
								   'LIWC_negemo', 'LIWC_anx', 'LIWC_anger', 'LIWC_sad', 'LIWC_cogmech',
								   'LIWC_insight', 'LIWC_cause', 'LIWC_discrep', 'LIWC_tentat',
								   'LIWC_certain', 'LIWC_inhib', 'LIWC_incl', 'LIWC_excl',
								   'LIWC_percept', 'LIWC_see', 'LIWC_hear', 'LIWC_feel', 'LIWC_bio',
								   'LIWC_body', 'LIWC_health', 'LIWC_sexual', 'LIWC_ingest',
								   'LIWC_relativ', 'LIWC_motion', 'LIWC_space', 'LIWC_time',
								   'LIWC_work', 'LIWC_achieve', 'LIWC_leisure', 'LIWC_home',
								   'LIWC_money', 'LIWC_relig', 'LIWC_death', 'LIWC_assent',
								   'LIWC_nonfl', 'tweet_count', 'word_count', 'has_url',
								   'is_rt', 'is_reply'],
						'full':	['tweet_id', 'user_id', 'total_words',
								   'LIWC_num_words', 'LIWC_happs', 'LabMT_num_words', 'LabMT_happs',
								   'ANEW_num_words', 'ANEW_happs', 'ANEW_arousal', 'ANEW_dominance',
								   'LIWC_total_count', 'LIWC_funct', 'LIWC_pronoun', 'LIWC_ppron',
								   'LIWC_i', 'LIWC_we', 'LIWC_you', 'LIWC_shehe', 'LIWC_they',
								   'LIWC_ipron', 'LIWC_article', 'LIWC_verb', 'LIWC_auxverb',
								   'LIWC_past', 'LIWC_present', 'LIWC_future', 'LIWC_adverb',
								   'LIWC_preps', 'LIWC_conj', 'LIWC_negate', 'LIWC_quant',
								   'LIWC_number', 'LIWC_swear', 'LIWC_social', 'LIWC_family',
								   'LIWC_friend', 'LIWC_humans', 'LIWC_affect', 'LIWC_posemo',
								   'LIWC_negemo', 'LIWC_anx', 'LIWC_anger', 'LIWC_sad', 'LIWC_cogmech',
								   'LIWC_insight', 'LIWC_cause', 'LIWC_discrep', 'LIWC_tentat',
								   'LIWC_certain', 'LIWC_inhib', 'LIWC_incl', 'LIWC_excl',
								   'LIWC_percept', 'LIWC_see', 'LIWC_hear', 'LIWC_feel', 'LIWC_bio',
								   'LIWC_body', 'LIWC_health', 'LIWC_sexual', 'LIWC_ingest',
								   'LIWC_relativ', 'LIWC_motion', 'LIWC_space', 'LIWC_time',
								   'LIWC_work', 'LIWC_achieve', 'LIWC_leisure', 'LIWC_home',
								   'LIWC_money', 'LIWC_relig', 'LIWC_death', 'LIWC_assent',
								   'LIWC_nonfl', 'time_unit', 'tweet_count', 'word_count', 'has_url',
								   'is_rt', 'is_reply', 'target','before_diag','before_susp',
								   'created_date','diag_date']
						}
				}
					
			},
			'fields_to_merge':{
				'tw':{
					'weekly':		['user_id','has_url','is_rt','is_reply','target','text','tweet_count','word_count','before_diag','before_susp','created_date','diag_date'],
					'created_date':	['user_id','has_url','is_rt','is_reply','target','text','tweet_count','word_count','before_diag','before_susp','created_date','diag_date'],
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
	return params 


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
		print 'TARGET :: {cond} / {plat} total:'.format(cond=t.upper(),plat=plat_long.upper()),samp.ix[samp.platform=='{}'.format(plat_long),:].shape[0]
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
	print 'Total attempted:', d.shape[0]
	print 'Disqualified for share_sm:', np.sum(d.share_sm=='No')
	print '% disq for share_sm:', np.sum(d.share_sm=='No')/float(d.shape[0])


def urls_per_user(data):
# avg num urls rated per user 

	targ_url_ct = data['target']['all'].groupby(['username','url']).count().reset_index().groupby('username').count()['url']
	control_url_ct = data['control']['all'].groupby(['username','url']).count().reset_index().groupby('username').count()['url']
	all_url_ct = pd.concat([targ_url_ct,control_url_ct])
	return targ_url_ct, control_url_ct, all_url_ct

def urls_rated_by_pop(data):
	c = data['control']['ratings'].url.unique().shape[0]
	t = data['target']['ratings'].url.unique().shape[0]
	return (c,t)

def subj_data_by_pop(data, target, platform, conn):
	
	output = {'control':{},'target':{}}
	popdata = {}

	popdata['control'] = pd.Series(data['control']['all'].username.unique())
	popdata['target'] = pd.Series(data['target']['all'].username.unique())

	output['control']['ct'] = popdata['control'].shape[0]
	output['target']['ct'] = popdata['target'].shape[0]

	for pop in ['control','target']:
		tup = tuple(popdata[pop].astype(str).values)

		if pop == 'control':
			q = 'select username, year_born, gender from control where {}="No" and username in {}'.format(target,tup)
			d = pd.read_sql_query(q,conn)
			d.drop_duplicates(subset=['username'], inplace=True)

			output['control']['femprop'] = round(np.sum(d.gender=="Female")/float(d.shape[0]),2)
			output['control']['perc_diag_within_13_15'] = None

		elif pop == 'target':
			q = 'select username, year_born, diag_date from {} where platform="{}" and username in {}'.format(target,platform,tup)
			d = pd.read_sql_query(q,conn)
			d.drop_duplicates(subset=['username'], inplace=True)
			output['target']['femprop'] = None

			ts = pd.to_datetime(d.diag_date)
			ct_within_2013_2015 = ts[(ts < pd.to_datetime('2015-12-31')) & (ts > pd.to_datetime('2013-01-01'))].shape[0]
			perc_within_2013_2015 = ct_within_2013_2015/float(ts.shape[0])
			output['target']['perc_diag_within_13_15'] = round(perc_within_2013_2015,2)
			
		age = pd.Series(2016 - d.year_born)
		output[pop]['age'] = {'min':age.min(),'max':age.max(),'mean':round(age.mean(),2),'std':round(age.std(),2)}

	return output

def get_descriptives(data, target, platform, additional_data, conn, return_output=False, doPrint=True):
	''' Reports on descriptive statistics of overall dataset used for analysis.
		Also includes stats broken down by target/control pop.
		Meant for printing, but also returns data from function call '''

	pop_data = subj_data_by_pop(data,target,platform,conn)
	ct = {}
	ct['target'], ct['control'], url_ct = urls_per_user(data)

	if additional_data:
		urls_rated_c, urls_rated_t = urls_rated_by_pop(data)
	
	if doPrint:
		print 'Mean posts per participant:', round(url_ct.mean(),2), '(SD={})'.format(round(url_ct.std(),2))
		print 'Median posts per participant:', round(url_ct.median(),2)
		if additional_data:
			print 'Photos rated: TARGET population ::', urls_rated_t
			print 'Photos rated: CONTROL population ::', urls_rated_c
		print
		
		for pop in ['target','control']:
			print 'POPULATION: {}'.format(pop.upper())
			print 'Total participants analyzed:', pop_data[pop]['ct']
			print 'Mean posts per participant:', round(ct[pop].mean(),2), '(SD={})'.format(round(ct[pop].std(),2))
			print 'Median posts per participant:', round(ct[pop].median(),2)
			if pop == 'control': 
				print 'Proportion female (control only):', pop_data[pop]['femprop']
			if pop == 'target': 
				print 'Proportion diagnosed between 2013-2015 (target only):', pop_data[pop]['perc_diag_within_13_15']
			print 'Average age:', pop_data[pop]['age']['mean'], '(SD={})'.format(pop_data[pop]['age']['std'])
			print 'Min age:', pop_data[pop]['age']['min']
			print 'Max age:', pop_data[pop]['age']['max']
			print
			print

	output = {'pop_data':pop_data, 'url_ct':url_ct}

	if additional_data:
		output['urls_rated'] = {'c':urls_rated_c,'t':urls_rated_t}
		
	output['url_ct'].plot(kind='hist', 
						  bins=50, 
						  title='Instagram posts per participant, {} (target + control)'.format(target))
	plt.show()

	if return_output:
		return output


def out_of_100(prev, prec, spec, rec, N=100):
	''' Reports model accuracy based on a hypothetical sample of 100 data points.
		Eg. "out of 100 samples, there were X positive IDs, Y false alarms, and Z false negatives..." '''

	print 'Out of {} random observations...'.format(N)
	print
	n_1 = round(N*prev)
	n_0 = N - n_1
	pos_id = round(n_1 * prec)
	neg_id = round(n_0 * spec)
	f_alarm = n_1 - pos_id
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
		
		if doPrint:
			print 'Number of users: get_meta yes addtl get_additional_data ::', d2.username.unique().shape[0]
			print d2.username.unique()
		
		# Now merge the meta_ig data with the photo_ratings_depression data
		consolidate_data(d2, d, platform, pop_long, kind, data)
		
		if doPrint:
			print 'Number of users: consolidate_date yes addtl get_additional_data ::', data[pop_long][kind].username.unique().shape[0]

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

		if doPrint:
			print 'Number of users: second consolidate yes addtl data get_addtl_data ::', data[pop_long]['all'].username.unique().shape[0]
	else:
		d = data[pop_long]['hsv']
		d2 = get_meta(params, conn, pop)
		tmp = d2.copy()
		
		if doPrint:
			print 'Number of users: get_meta no addtl get_additional_data ::', d2.username.unique().shape[0]
		
		# Now merge the meta_ig data with the photo_ratings_depression data
		consolidate_data(d2, d, platform, pop_long, 'all', data)
		
		if doPrint:
			print 'Number of users: consolidate_data no addtl get_additional_data ::', data[pop_long]['all'].username.unique().shape[0]
			print 'usernames from meta_ig that did not pass the merge between hsv and meta_ig:'
			print tmp.username[~tmp.username.isin(data[pop_long]['all'].username)].unique()

	if include_filter:
		data[pop_long]['all']['has_filter'] = 1
		data[pop_long]['all'].ix[ data[pop_long]['all']['filter'] == 'Normal','has_filter'] = 0

		if doPrint:
			print 'Prop {} with filter:'.format( pop_long ), data[pop_long]['all']['has_filter'].mean()


def cut_low_posters(data, pop_long, std_frac=0.5, doPrint=True):
	''' Cuts all data from participants with a low number of posts.
		'Low' is defined as fewer than (mean_posts_within_pop - 0.5*std_posts_within_pop) '''

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


def prepare_raw_data(data, platform, params, conn, gb_types, condition, periods, turn_points,
					 posting_cutoff=False, additional_data=False, 
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

		# aggregate data by groupby types (url, username, created_date)
		make_groupby(data[pop_long], platform, pop_long, params, gb_types, 
					 conn, condition, additional_data,
					 doPrint=True)

		if pop_long == 'target':
			# creates before/after subsets for target pop
			# subsets stored in: data[m]['target'][period][turn_point]['all']
			make_timeline_subsets(data, platform, periods)
		
			for period in periods:
				for turn_point in turn_points:
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
	''' 'is_reply' field, a la DeChoudhury, that looks to see whether there's an @ tag in the tweet '''

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


def get_tweet_metadata(data, m, params, conn, pop, pop_long, limit_date_range=False, 
					   t_maxback=-(365 + 60), t_maxfor=365, c_maxback=365*2, doPrint=False,
					   key='tweets'):
	''' Collects all Twitter metadata for a given population, and does some cleaning.
		If limit_date_range == True, then posts are limited to within maxback/maxfor days.
		For target populations, these limits are based on diag/susp date. 
		For control populations, limits extend back from present day (although they should really start 
		from the date they were collected...at some point we will be much father into the future than 
		the last date we collected their data!'''

	unames = get_pop_unames(params, m, conn, pop)
	meta = get_meta(params, conn, pop, doPrint)
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


def make_timeline_subsets(data, m, periods=['before','after'], doPrint=True):
	''' Creates subset of target observations based on diagnosis date.

		Currently (Apr 28 2016) we make the following subsets.  Note that "before/after" means two separate sets.
		Instagram:
		  - all data from before/after DIAG_date
		  #- only observations with photo ratings, before/after DIAG_date (in conditions where we have ratings)
		  - all data from before/after SUSP_date
		  #- only observations with photo ratings, before/after SUSP_date
		Twitter:
		  - all data from before/after DIAG_date
		  - all data from before/after SUSP_date '''

	if m == 'ig':
		base = 'all'
	elif m == 'tw':
		base = 'tweets'

	for period in periods:
		data['target'][period] = {}

		for turn_point in ['from_diag','from_susp']:
			if period == 'before':
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

		if gb_type == 'post':
			gb_list = ['username', post_unit]
			to_group_df = df[base] # for instagram, this is the 'outermost' groupby we do, on original df

		elif (gb_type == 'created_date') and (m=='ig'):
			gb_list = ['username', 'created_date']
			to_group_df = df['gb'][grouped] # this groupby is acting on the gb-url aggregate df

		elif (gb_type == 'created_date') and (m=='tw'):
			gb_list = ['user_id', 'created_date']
			to_group_df = df[base] # this groupby is acting on the gb-url aggregate df

		elif gb_type == 'weekly': # for tweets only
			gb_list = ['user_id', pd.Grouper(key='created_date',freq='W')]
			to_group_df = df[base] # this groupby is acting on the gb-url aggregate df

		elif (gb_type == 'username') and (m=='ig'):
			gb_list = ['username']
			to_group_df = df['gb'][username_gb] # this groupby is acting on the gb-created_date aggregate df

		elif (gb_type == 'user_id') and (m=='tw'):
			gb_list = ['user_id']
			to_group_df = df[base] # this groupby is acting on the gb-url aggregate df
			
		# testing
		#print 
		#print 'ROUND:', m, pop, gb_type 
		#print 'to_group_df shape:', to_group_df.shape
		#print to_group_df.columns
		#print 'agg cols:'
		#print params['agg_func'][m][gb_type]
		#print to_group_df.head()
		#print 

		df['gb'][gb_type] = (to_group_df.groupby(gb_list)
										.agg(params['agg_func'][m][gb_type])
							)
		# testing
		#print 'gb df shape:', df['gb'][gb_type].shape
		#print 'columns:', df['gb'][gb_type].columns

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
			#		df['gb'][gb_type].shape )
		
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
				df = df.ix[df.target==pval,:]
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


def importance_wrapper(fits, ctype, model_feats, title, tall_plot=False):
	# Plot the feature importances of the forest
	imp_cutoff = 0.01 # below this level of importance suggests a variable really doesn't matter
	fimpdf = pd.DataFrame(fits[ctype]['clf'].feature_importances_, index=model_feats, columns=['importance'])
	
	if tall_plot:
		fsize = (5,11)
	else:
		fsize = (5,7)
	plt.figure() 
	(fimpdf.sort_values('importance', ascending=True)
		   .ix[fimpdf.importance > imp_cutoff,:]
		   .plot(kind='barh', figsize=fsize, fontsize=14)
	 )
	plt.title("Best predictors (Random Forests, subset:{})".format(title), fontsize=16)
	plt.show()


def cleanX(X, doPrint=False):
	''' Drops columns which are all NaN 
		Imputer should take care of most issues, this function is for when an entire column is NaN '''

	to_drop = X.columns[X.isnull().all()]
	if doPrint:
		print 'Columns to drop:', to_drop

	X.drop(to_drop,1,inplace=True)


def pca_explore(pca, X):
	''' Shows variance accounted for by principal components '''

	plt.figure()
	X_reduced = pca.fit_transform(scale(X))
	print 'Total vars:', X.shape[1]
	print 'Num components selected by Minka MLE:', pca.components_.shape[0]
	print 'Cumulative % variance explained per component:'
	cumvar = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
	print cumvar
	plt.plot(cumvar)
	_=plt.ylim([0,100])
	plt.show()
	return X_reduced, pca.n_components_


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


def pca_report(fit, feats, N=3):
	''' Prints components loadings for the top N components.
		Three separate printouts are made, each is sorted with the highest loading variables per component on top
	'''
	loaddf = (pd.DataFrame(fit.components_.T[:,0:N], 
						   columns=['PCA_{}'.format(x) for x in np.arange(N)], 
						   index=feats)
			  .abs()
			  )
	for n in np.arange(N):
		comp_ix = 'PCA_{}'.format(n)
		print loaddf.sort_values(comp_ix,ascending=False).ix[0:10,comp_ix]
		print 


def make_models(d, test_size=0.3, clf_types=['lr','rf','svc'], excluded_set=None, 
				tall_plot=False, n_est=100, kernel='rbf', use_pca=False, num_pca_comp=None,
				labels={'known_0':'known_control',
						'known_1':'known_target',
						'pred_0':'pred_control',
						'pred_1':'pred_target'}):

	''' NOTE: clf_types can include: 'lr' (logistic reg.),'rf' (random forests),'svc' (support vec. clf.) '''

	mdata = d['data']
	title = d['name']
	unit = d['unit']
	target = d['target'] 
	feats = d['features']
	acc_avg = d['acc_avg']

	if 'tall_plot' in d.keys():
		tall_plot = d['tall_plot']
	
	X = mdata[feats].copy()
	y = mdata[target]

	cleanX(X)

	model_feats = X.columns

	imp = Imputer(strategy='median')
	X = imp.fit_transform(X,y)
	
	if use_pca:
		# http://stats.stackexchange.com/questions/82050/principal-component-analysis-and-regression-in-python
		# http://nxn.se/post/36838219245/loadings-with-scikit-learn-pca
		# http://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ration-in-pca-with-sklearn
		pca = PCA(n_components='mle')
		X_reduced, num_pca_comp = pca_explore(pca, X)
		num_pca_comp = pca_model(pca, X_reduced, y, num_pca_comp)
		pca_report(pca, model_feats)

		# we do all subsequent modeling with the best PCA component vectors
		if num_pca_comp + 1 > X_reduced.shape[1]:
			max_ix = X_reduced.shape[1] 
		else:
			max_ix = num_pca_comp+1

		X = X_reduced[:,0:max_ix].copy() 
		model_feats = pd.Series(['pca_{}'.format(x) for x in np.arange(max_ix)])
		#testing
		#print 'num_pca_comp:', num_pca_comp
		#print 'X_reduced shape:', X_reduced.shape 
		#print np.arange(0,max_ix)
		#print np.arange(1,max_ix)
		#print 'PCA X shape:', X.shape
		#print 'PCA model_feats shape:', model_feats.shape 
		#print 'PCA model_feats:', model_feats

	else:
		pca = None

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
	
	fits = {}

	fits['lr'] = {'name':'Logistic Regression','clf':logregcv(class_weight='auto')}
	fits['rf'] = {'name':'Random Forests','clf':RFC(n_estimators=n_est, class_weight='balanced')}
	fits['svc'] = {'name':'Support Vector Machine','clf':SVC(class_weight='auto', kernel=kernel, probability=True)}
	
	for ctype in clf_types:
		fits[ctype]['clf'].fit(X_train,y_train)
			
	if use_pca:
		print 
		print 'NOTE: ALL MODELS ON THIS RUN USE PCA COMPONENTS!'
		print

	print 'UNIT OF OBSERVATION:', unit.upper()
	total_obs = X_test.shape[0]
	total_neg = np.sum(y_test==0)

	prop_neg = round((total_neg)/float(total_obs),3)

	if prop_neg > 0.5:
		majority_class = 'control'
		pos_label = 0
		naive_acc = prop_neg
	else:
		majority_class = 'target'
		pos_label = 1
		naive_acc = 1 - prop_neg

	print 'NAIVE ACCURACY ALL MAJORITY:', naive_acc
	print "  *'ALL MAJORITY' = all observations are predicted as majority class (here: {})".format(majority_class)
	print 
	print
	
	output = {}

	output['X_test'] = X_test
	output['y_test'] = y_test

	for ctype in clf_types:
		
		print_model_summary(fits[ctype], ctype,
							target, title, X_test, y_test, labels, average=acc_avg, pos_label=pos_label)

		roc_wrapper(fits, ctype, y_test, X_test, d['platform'])
		
		if ctype == 'rf':
			importance_wrapper(fits, ctype, model_feats, title, tall_plot)
	
		output[ctype] = fits[ctype]['clf']

	return output, pca


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


def compare_filters(data, conn, level, gb_type, show_figs=True):
	''' Chi2, plotting comparisons of Instagram filter use between target and control pops '''

	metaig = pd.read_sql_query('select username, filter, d_from_diag_depression as ddiag from meta_ig', conn)

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
	filtct_chi2expect = pd.DataFrame(chi2[3], columns=['control','target'], index=filtct_chi2.index)
	filtct_chi2offset = filtct_chi2 - filtct_chi2expect 

	print 'chi2 stats comparing Instagram filters:'
	print 'chi2 value:',chi2[0]
	print 'p-value:',chi2[1]
	print 'deg.f.:',chi2[2]

	if show_figs:
		plt.figure()
		filtct_chi2.sort_values('target',ascending=False).plot(kind='bar', figsize=(16,8), fontsize=14)
		plt.title('Instagram filter frequency counts (target=depressed)', fontsize=18)

		plt.figure()
		filtct_chi2offset.sort_values('target',ascending=True).plot(kind='bar', figsize=(16,8), fontsize=14)
		plt.title('Instagram filter frequency count Chi2 offset (observed-expected)', fontsize=18)

		plt.figure()
		filtct_chi2offset.ix[filtct_chi2offset.index!='Normal',:].sort_values('target',ascending=True).plot(kind='bar', figsize=(16,8), fontsize=14)
		plt.title('Instagram filter frequency count Chi2 offset (observed-expected), Normal excluded', fontsize=18)

	return filts 


def logreg_output(dm, resp):
	''' Performs frequentist logistic regression, returns model stats '''
	logit = sm.Logit(resp, dm)
	# fit the model
	result = logit.fit()
	print result.summary()
	print 
	print 'Odds ratios:'
	print np.exp(result.params)


def logreg_wrapper(master, gb_type, varset, additional_data, target='target'):
	''' formatting wrapper for logreg_output() '''

	print 'UNIT OF MEASUREMENT:', gb_type
	print

	if additional_data:
		vlist = 'means'
	else:
		vlist = 'no_addtl_means'

	#print 'vars:', varset[gb_type][vlist]
	predictors = varset[gb_type][vlist] 
	response = master[gb_type][target]
	design_matrix = smtools.add_constant(master[gb_type][predictors]) # adds bias term, ie. vector of ones

	logreg_output(design_matrix, response)


def master_actions(master, target, control, condition, m, params, gb_type, 
				   report, aparams, clfs, additional_data, posting_cutoff,
				   use_pca=False):
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

	if aparams['save_to_file']:
		# save csv of target-type // groupby-type
		addtl = 'addtl_data' if additional_data else 'no_addtl_data'
		postcut = 'post_cut' if posting_cutoff else 'post_uncut'
		fpath = '/'.join(['data-files',condition,m,gb_type])
		fname = '_'.join([condition,m,gb_type,report,addtl,postcut])
		csv_name = '{}/{}.csv'.format(fpath,fname)
		master[gb_type].to_csv(csv_name, encoding='utf-8', index=False)

	if aparams['ml']:
		print 'Building ML models...'
		print 
		if additional_data:
			vlist = 'means'
		else:
			vlist = 'no_addtl_means'
		model_df = {'name':'Models: {} {}'.format(report, gb_type),
					'unit':gb_type,
					'data':master[gb_type],
					'features':params['vars'][m][gb_type][vlist],
					'target':'target',
					'platform':m,
					'tall_plot':aparams['tall_plot'],
					'acc_avg':aparams['acc_avg']
				   }

		output, pca = make_models(model_df, clf_types=clfs, 
								  excluded_set=params['model_vars_excluded'][m][gb_type],
								  use_pca=use_pca)

		for k in output.keys():
			master['model'][gb_type][k] = output[k]

	if aparams['nhst']:
		print
		print 'TTEST'
		ttest_out = ttest_wrapper(master, gb_type, params['vars'][m], additional_data)
		master['model'][gb_type]['ttest'] = ttest_out[0]
		master['model'][gb_type]['ttest_pvals'] = ttest_out[1]

		print
		print 'LOGISTIC REGRESSION'
		print logreg_wrapper(master, gb_type, params['vars'][m], additional_data)



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
			create_word_feats(data[pop_long], tunit, condition, conn, write_to_db=True, testing=False)


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
		#testingprint 'current cols:'
		#print df['word_feats'][tunit].columns
		#print
		#print 'table fields:'
		#print table_fields
		#print
		#print q
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

def get_local_time_data(metatw):
	metatw['local_timestamp'] = tz.local_timestamp
	metatw['local_time'] = metatw.local_timestamp.apply(get_time_only)

def tag_insomnia(x):
	
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
	