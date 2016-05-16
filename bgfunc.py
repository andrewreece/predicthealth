import sqlite3
import pandas as pd
import numpy as np
from scipy import linalg
import itertools
import datetime

from matplotlib import pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('white')

import pickle
from skimage import io, data, color
import re
from re import findall,UNICODE

from sklearn.cross_validation import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

from sklearn.preprocessing import scale
from sklearn import cross_validation
from sklearn import mixture

from statsmodels.sandbox.stats.multicomp import multipletests

from labMTsimple.speedy import *


def define_params(condition, test_name, test_cutoff, 
				  platform, platform_long, fields,
				  photos_rated=False, has_test=False, ratings_min=3):
	''' creates params dict for queries and other foundational parameters used throughout analysis '''

	if photos_rated and (platform=='ig'):
		ratings_clause = ' and ratings_ct >= {}'.format(ratings_min)
	else:
		ratings_clause = ''

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
					'unames':		'select username from {cond} where platform="{plat_long}" and {test_name} > {cutoff} and username is not null and disqualified=0'.format(cond=condition, 
																																											 test_name=test_name, 
																																											 cutoff=test_cutoff,
																																											 plat_long=platform_long)
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
						'happy':		['mean','var'],
						'sad':			['mean','var'],
						'interesting':	['mean','var'],
						'likable':		['mean','var'],
						'one_word':		' '.join,
						'description':	'    '.join,
						'rater_id':		'count',
						'username':		'first',
						'created_date':	'first',
						'target':		'first',
						'hue':			'first',
						'saturation':	'first',
						'brightness':	'first',
						'before_diag':	'first',
						'before_susp':	'first'
					},
					'username':{'url':				'count',
								'likable|mean':		['mean','var'],
								'interesting|mean':	['mean','var'],
								'happy|mean':		['mean','var'],
								'sad|mean':			['mean','var'],
								'one_word':			'_|_'.join,
								'description':		'__|__'.join,
								'target':			'first',
								'hue':				'mean',
								'saturation':		'mean',
								'brightness':		'mean'
					},
					'created_date':{'url':				'count',
									'likable|mean':		['mean','var'],
									'interesting|mean':	['mean','var'],
									'happy|mean':		['mean','var'],
									'sad|mean':			['mean','var'],
									'one_word':			'_|_'.join,
									'description':		'__|__'.join,
									'target':			'first',
									'hue':				'mean',
									'saturation':		'mean',
									'brightness':		'mean',
									'before_diag':		'first',
									'before_susp':		'first'
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
						'means':['likable|mean','interesting|mean','sad|mean','happy|mean',
								 'hue','brightness','saturation'],
						'full':	['likable|mean','interesting|mean','sad|mean','happy|mean',
								 'likable|var','interesting|var','sad|var','happy|var',
								 'hue','saturation','brightness','before_diag','before_susp','target']
						},
					'username':{
						'means':['likable|mean|mean','interesting|mean|mean','sad|mean|mean','happy|mean|mean',
								 'hue|mean','saturation|mean','brightness|mean'],
						'full':	['likable|mean|mean','interesting|mean|mean','sad|mean|mean','happy|mean|mean',
								 'likable|mean|var','interesting|mean|var','sad|mean|var','happy|mean|var',
								 'hue|mean','saturation|mean','brightness|mean','one_word|join','description|join','target']
						},
					'created_date':{
						'means':['likable|mean|mean','interesting|mean|mean','sad|mean|mean','happy|mean|mean',
								 'hue|mean','saturation|mean','brightness|mean'],
						'full':['likable|mean|mean','interesting|mean|mean','sad|mean|mean','happy|mean|mean',
								'likable|mean|var','interesting|mean|var','sad|mean|var','happy|mean|var',
								'hue|mean','saturation|mean','brightness|mean','one_word|join','description|join',
								'before_diag','before_susp','target']
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
					'post':['username','username|first','url','rater_id|count','likable|var',
							'description|join', 'interesting|var',
							'sad|var', 'one_word|join', 'happy|var','before_diag','before_susp','target'],
					'username':['username','url|count','interesting|mean|var','sad|mean|var',
								'happy|mean|var','likable|mean|var', 
								'one_word|join','description|join', 'target'],
					'created_date':['username','url|count','interesting|mean|var','sad|mean|var',
									'happy|mean|var','likable|mean|var', 
									'one_word|join','description|join','before_diag','before_susp','target']
				},
				'tw':{
					'weekly':['user_id','before_diag','before_susp','target'],
					'user_id':['user_id', 'target','total_words', 'tweet_count',
							   'LIWC_num_words', 'LabMT_num_words', 
							   'ANEW_num_words','LIWC_total_count'],
					'created_date':['before_diag','before_susp','target']
				}
			},
			'master_drop_subset':{
				'ig':{'post':['hue','saturation','brightness'],
					  'username':['hue|mean','saturation|mean','brightness|mean'],
					  'created_date':['hue|mean','saturation|mean','brightness|mean']
					  },
				'tw':{'created_date':[],
					  'user_id':[],
					  'weekly':[]
					  }
			},
			'rated':photos_rated,
			'has_test':has_test 
		}

	return params 


def report_sample_sizes(params, conn, cond, plat_long, test_cutoff, 
						test='', show_all=False,
						table=['depression','pregnancy','ptsd','cancer']):
	''' Printout of sample sizes across conditions and thresholds '''

	print 'SAMPLE SIZES: target populations'
	print

	if not show_all:
		table = [cond] # if not show_all, then just show current condition being analyzed

	for t in table:
		extra_field = '{}'.format(test)
			
		q = 'select platform, {} from {} where username is not null and diag_year is not null and disqualified=0'.format(extra_field, t)
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


def prepare_raw_data(data, platform, params, conn, gb_types, condition, periods, turn_points):
	''' Pulls data from db, cleans, and aggregates. Also creates subsets based on diag/susp date '''

	# collect target and control data
	for pop, pop_long in [('t','target'),('c','control')]:
		
		print '{} DATA:'.format(pop_long.upper())
		
		# get hsv data
		get_basic_data(data, platform, params, conn, pop, pop_long)
		
		# aggregate data by groupby types (url, username, created_date)
		make_groupby(data[pop_long], platform, pop_long, params, gb_types, 
					 conn, condition, 
					 doPrint=True)

		if pop_long == 'target':
			# creates before/after subsets for target pop
			# subsets stored in: data[m]['target'][period][turn_point]['all']
			make_timeline_subsets(data, platform, periods)
		
			for period in periods:
				for turn_point in turn_points:
					# aggregate data by groupby types (url, username, created_date)
					make_groupby(data[pop_long][period][turn_point], platform, pop_long, params, gb_types, 
								 conn, condition, period = period, turn_point=turn_point)  


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


def get_basic_data(data, m, params, conn, pop, pop_long, doPrint=False):
	''' gets the set of variables that all observations have filled (eg. hsv for photos, tweet data for twitter)
		this is basically just a wrapper for get_hsv() and get_tweet_metadata() '''

	if m == 'ig':
		get_hsv(data, m, params, conn, pop, pop_long)
	elif m == 'tw':
		get_tweet_metadata(data, m, params, conn, pop, pop_long, doPrint)


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


def get_tweet_metadata(data, m, params, conn, pop, pop_long,
					   doPrint=False, key='tweets'):
	''' Collects all Twitter metadata for a given population, and does some cleaning '''

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

	if pop_long == 'target':
		max_backward = -(365 + 60) # one year back plus 60 days to make sure we're including a full year back from suspected 
		max_forward = 365
		mask = (all_tweets.from_diag >= max_backward) & (all_tweets.from_diag <= max_forward)
	
	elif pop_long == 'control':
		day_span = 365*2 # to approximate going one year back and forwards from diag_date in target pop
		dfp = [(datetime.datetime.now() - d).days for d in all_tweets.created_date]
		all_tweets['days_from_present'] = dfp
		mask = (all_tweets.groupby('username')['days_from_present']
						  .apply(lambda x: x.nsmallest(day_span))
						  .reset_index()['level_1'] # level_1 gets row indices from all_tweets to use as mask
				)

	tweets = all_tweets.ix[ mask, : ].copy()

	
	if doPrint:
		print 'Num tweets in {}, before dropping duplicates:'.format(pop_long.upper()), tweets.shape[0]
		print 'Num tweets in {}, after dropping duplicates:'.format(pop_long.upper()), data[pop_long][key].shape[0]

	data[pop_long]['tweets'] = tweets 


def get_hsv(data, m, params, conn, pop, pop_long, doPrint=False):
	''' Gets HSV values for Instagram photos '''

	hsv = pd.read_sql_query(params['q']['all_hsv'],conn)
	hsv.dropna(inplace=True)
	
	if doPrint:
		print 'Number photos with HSV ratings (all conditions):', hsv.shape[0]

	unames = get_pop_unames(params, m, conn, pop)
	metaig = get_meta(params, conn, pop)
	urls = metaig.ix[metaig.username.isin(unames.username),'url'].values

	if doPrint:
		print 'Num HSV-rated photos with URL in {}, before dropping duplicates:'.format(pop_long.upper()), hsv.ix[hsv.url.isin(urls),:].shape[0]

	data[pop_long]['hsv'] = hsv.ix[hsv.url.isin(urls),['url','hue','saturation','brightness']].drop_duplicates(subset='url').copy()

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


def consolidate_data(d, d2, m, pop_long, kind, data):
	''' merges dfs, adds 0/1 class indicator variable '''

	data[pop_long][kind] = d.merge(d2, how='left',on='url')

	if pop_long == 'target':
		cl = 1
	elif pop_long == 'control':
		cl = 0

	data[pop_long][kind]['target'] = cl
	mark_before_after(data[pop_long][kind], pop_long)

	print 'Shape of consolidated {} {} data:'.format(pop_long.upper(), kind.upper()), data[pop_long][kind].shape 



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



def print_coltype(data, condition, m, pop):
	''' Get data type of all columns in data frame '''
	x = data[pop]['ratings']
	print 'Data column types for medium: {}, population: {}, condition: {}'.format(m, pop, condition)
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
			'sad':{'Not':0,'None':0,'Not sad':0,'Not at all':0,'Very little':1,'Not really':1,'Slightly':2,'Kind of ':3,'Kind of':3,'Sure':4,'Very':5},
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
				 conn=None, condition=None, 
				 period=None, turn_point=None, doPrint=False):
	''' Create aggregated datasets 

		We collect mean and variance for each numerical measure, 
		and we combine all of the text descriptions

		Note: This kind of groupby results in a MultiIndex colum format (at least where we have [mean,var] aggs).  
		The code after the actual groupby flattens the column hierarchy.'''

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
			to_group_df = df['gb'][grouped] # this groupby is acting on the gb-url aggregate df

		elif (gb_type == 'user_id') and (m=='tw'):
			gb_list = ['user_id']
			to_group_df = df[base] # this groupby is acting on the gb-url aggregate df
			
		# testing
		#print 
		#print 'ROUND:', m, pop, gb_type 
		#print 'to_group_df shape:', to_group_df.shape
		#print to_group_df.columns

		df['gb'][gb_type] = (to_group_df.groupby(gb_list)
										.agg(params['agg_func'][m][gb_type])
							)
		# testing
		#print 'gb df shape:', df['gb'][gb_type].shape
		#print 'columns:', df['gb'][gb_type].columns

		if m == 'ig':
			# collapses multiindex columns into |-separated names
			new_colname = ['%s%s' % (a, '|%s' % b if b else '') for a, b in df['gb'][gb_type].columns]
			
			df['gb'][gb_type].columns = new_colname
			# bring url into column set (instead of as row index)
			df['gb'][gb_type].reset_index(inplace=True)

			if 'created_date|first' in df['gb'][gb_type].columns:
				df['gb'][gb_type].rename(columns={'created_date|first':'created_date',
												  'one_word|join':'one_word',
												  'description|join':'description',
												  'hue|first':'hue',
												  'saturation|first':'saturation',
												  'brightness|first':'brightness'}, inplace=True)
			for field in ['target|first','before_diag|first','before_susp|first']:
				if field in df['gb'][gb_type].columns:
					df['gb'][gb_type].rename(columns={field:field.split("|")[0]}, inplace=True)

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


def merge_to_master(master, target, control, m, varset, gb_type, doPrint=False):
	''' merge target and control dfs into master df '''

	c = control[gb_type]
	t = target[gb_type]
	#subset = params['master_subset'][m][gb_type]
	master[gb_type] = pd.concat([c,t], axis=0)#.dropna(subset=subset)
	master[gb_type] = master[gb_type][ varset[m][gb_type]['full'] ]

	if doPrint:
		print 'Master {} {} nrows:'.format(m.upper(), gb_type.upper()), master[gb_type].shape[0]


def compare_density(df, m, gbtype, varset, ncols=4):
	''' Overlays density plots of target vs control groups for selected aggregation type '''

	print 'target vs control for {} {}-groupby:'.format(m.upper(), gbtype.upper())
	plt.figure()
	gb = df[gbtype].groupby('target')

	numvars = float(len(varset[m][gbtype]['means']))
	nrows = np.ceil(numvars/ncols).astype(int)
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 3*nrows), tight_layout=True)

	for ax, p in zip(axes.ravel(), varset[m][gbtype]['means']):
		for k, v in gb[p]:
			try:
				sns.kdeplot(v, ax=ax, label=str(k)+":"+v.name)
			except Exception, e:
				print 'Error or all zeros for variable:', v.name
	plt.show()


def corr_plot(df, m, gb_type, varset, print_corrmat=False):

	print 'Correlation matrix:'

	metrics = varset[m][gb_type]['means']
	corr = df[gb_type][metrics].corr().round(2)
	corr.columns = [x.split("|")[0] for x in corr.columns]
	corr.index = [x.split("|")[0] for x in corr.index]

	# Generate a mask for the upper triangle
	mask = np.zeros_like(corr, dtype=np.bool)
	mask[np.triu_indices_from(mask)] = True

	plt.figure()
	# Set up the matplotlib figure
	f, ax = plt.subplots(figsize=(11, 9))

	# Generate a custom diverging colormap
	cmap = sns.diverging_palette(220, 10, as_cmap=True)

	# Draw the heatmap with the mask and correct aspect ratio
	sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0,
				square=True, linewidths=.5, 
				cbar_kws={"shrink": .5}, ax=ax)
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


def print_model_summary(fit,ctype,target,title,X_test,y_test,labels):
	''' formats model output in a notebook-friendly print layout '''

	print 'MODEL: {} {} ({}):'.format(fit['name'],target,title)
	print 'NAIVE ACCURACY:'.format(ctype), round(fit['clf'].score(X_test,y_test),3)

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
	''' Shows cross-validated F1 scores using PCA components in logistic regression '''

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
	print 'num pca comp displayed:', num_pca_comp
	print 'optimal number of components:', new_num_pca_comp # optimal num components based on max F1 score
	
	plt.figure()
	fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))
	ax1.plot(f1, '-v')
	ax2.plot(np.arange(num_pca_comp), f1[1:num_pca_comp+1], '-v')
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
	mdata = d['data']
	title = d['name']
	unit = d['unit']
	target = d['target'] 
	feats = d['features']
	if 'tall_plot' in d.keys():
		tall_plot = d['tall_plot']
	
	X = mdata[feats]
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
		X = X_reduced[:,0:num_pca_comp+1].copy() # we do all subsequent modeling with the best PCA component vectors
		model_feats = pd.Series(['pca_{}'.format(x) for x in np.arange(num_pca_comp+1)])
		
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
	
	fits = {}

	fits['lr'] = {'name':'Logistic Regression','clf':logregcv(class_weight='auto')}
	fits['rf'] = {'name':'Random Forests','clf':RFC(n_estimators=n_est)}
	fits['svc'] = {'name':'Support Vector Machine','clf':SVC(class_weight='auto', kernel=kernel, probability=True)}
	
	for ctype in clf_types:
		fits[ctype]['clf'].fit(X_train,y_train)
			
	if pca:
		print 
		print 'NOTE: ALL MODELS ON THIS RUN USE PCA COMPONENTS!'
		print

	print 'UNIT OF OBSERVATION:', unit.upper()
	total_obs = X_test.shape[0]
	total_incomp = np.sum(y_test==0)
	print 'NAIVE ACCURACY ALL NULL:', round((total_incomp)/float(total_obs),3)
	print "  *'ALL NULL' means if all observations are predicted as uncompleted assessments"
	print 
	print
	
	output = {}

	output['X_test'] = X_test
	output['y_test'] = y_test

	for ctype in clf_types:
		
		print_model_summary(fits[ctype], ctype,
							target, title, X_test, y_test, labels)

		roc_wrapper(fits, ctype, y_test, X_test, d['platform'])
		
		if ctype == 'rf':
			importance_wrapper(fits, ctype, model_feats, title, tall_plot)
	
		output[ctype] = fits[ctype]['clf']

	return output, pca


def ttest_output(a, b, varset, ttype, correction=True, alpha=0.05, method='bonferroni'):
	''' performs independent-samples t-tests with bonferroni correction '''

	pvals = []

	for metric in varset:
		if metric[0:5] != 'LIWC': # this controls for 0s on happs scores and other vars that shouldn't have 0
			a = a.ix[a[metric] != 0, :]
			b = b.ix[b[metric] != 0, :]
		print 'RATING: {}'.format(metric)
		if ttype == 'ind':
			test = ttest(a[metric], b[metric])
		elif ttype == 'dep':
			test = ttest_rel(a[metric], b[metric])
		print test
		pvals.append( test.pvalue )
		print
		
	if correction:
		corrected = multipletests(pvals,alpha=alpha,method=method)
	
		print '{}-corrected alpha of {}:'.format(method,alpha),corrected[3]
		for i, metric in enumerate(varset):
			print '{} significant post-correction? {} ({})'.format(metric, corrected[0][i], corrected[1][i])
			
	return test, pvals


def ttest_wrapper(master, gb_type, varset, split_var='target', ttype='ind'):
	''' formatting wrapper for ttest_output() '''

	print 'UNIT OF MEASUREMENT:', gb_type
	print

	a = master[gb_type].ix[master[gb_type][split_var]==0,:]
	b = master[gb_type].ix[master[gb_type][split_var]==1,:]

	return ttest_output(a, b, varset[gb_type]['means'], ttype)


def master_actions(master, target, control, condition, m, params, gb_type, report, aparams, clfs, 
				   use_pca=False):
	''' Performs range of actions on master data frame, including plotting, modeling, and saving to disk. 
		
		Note: "Master" may refer to any set of target+control data, including timeline subsets. 
		In other words, both the full dataset and subsets of full dataset may be passed as "master" argument.'''

	if aparams['create_master']:

		# merge target, control, into master
		print 
		print 'Merge to master: {} {}'.format(report, gb_type)
		merge_to_master(master, target, control, m, params['vars'], gb_type)

		print 'master {} shape:'.format(gb_type), master[gb_type].shape
		print

		master['model'][gb_type] = {}
	
	if aparams['density']:
		compare_density(master, m, gb_type, params['vars'])

	if aparams['corr']:
		corr_plot(master, m, gb_type, params['vars'], aparams['print_corrmat'])

	if aparams['save_to_file']:
		# save csv of target-type // groupby-type
		csv_name = '{cond}_{m}_{gbt}_{r}.csv'.format(cond=condition,m=m,gbt=gb_type,r=report)
		master[gb_type].to_csv(csv_name, encoding='utf-8', index=False)

	if aparams['ml']:
		print 'Building ML models...'
		print 
		model_df = {'name':'Models: {} {}'.format(report, gb_type),
					'unit':gb_type,
					'data':master[gb_type],
					'features':params['vars'][m][gb_type]['means'],
					'target':'target',
					'platform':m,
					'tall_plot':aparams['tall_plot']
				   }

		output = make_models(model_df, clf_types=clfs, 
							 excluded_set=params['model_vars_excluded'][m][gb_type],
							 use_pca=use_pca)

		for k in output.keys():
			master['model'][gb_type][k] = output[k]

	if aparams['nhst']:
		ttest_out = ttest_wrapper(master, gb_type, params['vars'][m])
		master['model'][gb_type]['ttest'] = ttest_out[0]
		master['model'][gb_type]['ttest_pvals'] = ttest_out[1]


def before_vs_after(df, gb_type, m, condition, varset, aparams, ncols=4):
	for date in ['diag','susp']:
		print
		print 'before vs after (target: {}) for {}-groupby, based on {}_date:'.format(condition, gb_type, date)

		splitter = 'before_{}'.format(date)
		gb = df[gb_type].groupby(splitter)

		if aparams['density']:
			plt.figure()
			numvars = float(len(varset[gb_type]['means']))
			nrows = np.ceil(numvars/ncols).astype(int)
			fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 3*nrows), tight_layout=True)

			for ax, p in zip(axes.ravel(), varset[gb_type]['means']): # don't use gb_type here, it's only for url
				for k, v in gb[p]:
					sns.kdeplot(v, ax=ax, label=str(k)+":"+v.name)
			plt.show()

		if aparams['nhst']:
			ttest_wrapper(df, gb_type, varset, split_var=splitter, ttype='ind')


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

	# print(len(my_LIWC.data))
	# print(len(my_LIWC.scorelist))
	my_word_vec = my_LIWC_stopped.wordVecify(word_dict)
	# print(len(my_word_vec))
	# print(sum(my_word_vec))
	happs = my_LIWC_stopped.score(word_dict)
	# print(len(my_LIWC.data))
	# print(len(my_LIWC.scorelist))
	# print(happs)
	result['LIWC_num_words'] = sum(my_word_vec)
	result['LIWC_happs'] = happs

	my_word_vec = my_LabMT.wordVecify(word_dict)
	happs = my_LabMT.score(word_dict)
	# print(len(my_word_vec))
	# print(sum(my_word_vec))
	# print(happs)
	result['LabMT_num_words'] = sum(my_word_vec)
	result['LabMT_happs'] = happs
	my_word_vec = my_ANEW.wordVecify(word_dict)
	happs = my_ANEW.score(word_dict)
	# print(len(my_word_vec))
	# print(sum(my_word_vec))
	# print(result)
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
		q = 'insert into word_features{} values ('.format(word_features) + ",".join(['?']*len(feats_to_write)) +")"
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
	