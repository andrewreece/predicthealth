import pandas as pd
import numpy as np 
import re

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression as logreg
from sklearn.linear_model import LogisticRegressionCV as logregcv
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import Imputer

from scipy.stats import ttest_ind as ttest
from scipy.stats import ttest_rel
from statsmodels.sandbox.stats.multicomp import multipletests

from matplotlib import pyplot as plt 
import seaborn as sns
sns.set_style('white')

def define_params(condition, test_name, test_cutoff, 
				  photos_rated=False, has_test=False):
	''' creates params dict for queries and other foundational parameters used throughout analysis '''

	if photos_rated:
		rt_cond = ' and ratings_ct > 0'
	else:
		rt_cond = ''

	params = {
			'ig':{
				'q':{
					't':{
						'meta_ig':		'select url, username, instagram_user_id, created_date, d_from_diag_{cond} as from_diag, d_from_susp_{cond} as from_susp from meta_ig where d_from_diag_{cond} is not null{ratings_ct_cond}'.format(cond=condition,ratings_ct_cond=rt_cond),
						'photo_ratings':'select url, rater_id, happy, sad, interesting, likable, one_word, description from photo_ratings_{}'.format(condition),
						'before_diag':	'select url, username, instagram_user_id, created_date from meta_ig where d_from_diag_{} < 0 and ratings_ct > 0'.format(condition),
						'after_diag':	'select url, username, instagram_user_id, created_date from meta_ig where d_from_diag_{} >= 0 and ratings_ct > 0'.format(condition),
						'unames':		'select username from {cond} where platform="instagram" and {test_name} > {cutoff} and username is not null and disqualified=0'.format(cond=condition, test_name=test_name, cutoff=test_cutoff)
					},
					'c':{
						'photo_ratings':'select url, rater_id, happy, sad, interesting, likable, one_word, description from photo_ratings_{}_control'.format(condition),
						'unames':		'select username from control where platform="instagram" and username is not null and {}="No" and disqualified=0'.format(condition)
					},
					'all_meta_ig':	'select url, username, instagram_user_id, created_date, d_from_diag_{cond} as from_diag, d_from_susp_{cond} as from_susp from meta_ig'.format(cond=condition),
					'all_hsv':		'select url, hue, saturation, brightness, username from hsv'
				},
				'agg_func':{
					'url':{
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
				}
			},
			'tw':{
				'agg_func':{},
				'q':{
					't':{
						'unames':'select username from {cond} where platform="twitter" and {test_score} > {cutoff} and username is not null and disqualified=0'.format(cond=condition, test_score=test_name, cutoff=test_cutoff)
					},
					'c':{},
					'all_meta_tw':			'select text, username, twitter_user_id, created_date, d_from_diag_{cond} as from_diag, d_from_susp_{cond} as from_susp from meta_tw'.format(cond=condition),
					'unames_above_cutoff':	'select username from {} where {} > {} and username is not null and disqualified=0 and platform="twitter"'.format(condition, test_name, test_cutoff)
				}
			},
			'vars':{
					'url':{ 
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
						},
					'before_after':	['likable|mean|mean','interesting|mean|mean','sad|mean|mean','happy|mean|mean',
									'hue|mean','saturation|mean','brightness|mean']
			},
			'excluded':{
				'url':[	'username','username|first','url','rater_id|count','likable|var',
						'description|join', 'interesting|var',
						'sad|var', 'one_word|join', 'happy|var','before_diag','before_susp','target'],
				'username':['username','url|count','interesting|mean|var','sad|mean|var',
							'happy|mean|var','likable|mean|var', 
							'one_word|join','description|join', 'target'],
				'created_date':['username','url|count','interesting|mean|var','sad|mean|var',
								'happy|mean|var','likable|mean|var', 
								'one_word|join','description|join','before_diag','before_susp','target']
			},
			'rated':photos_rated,
			'has_test':has_test 
		}

	return params 


def report_sample_sizes(params, conn, table=['depression','pregnancy','ptsd','cancer']):
	''' Printout of sample sizes across conditions and thresholds '''

	print 'SAMPLE SIZES: target populations'
	print

	for t in table:
		if t == 'depression':
			extra_field = 'cesd,'
		elif t == 'ptsd':
			extra_field = 'tsq,'
		else:
			extra_field = ''
			
		q = 'select username, {} platform from {} where username is not null and diag_year is not null and disqualified=0'.format(extra_field, t)
		samp = pd.read_sql_query(q,conn)
		print '{} / TWITTER total:'.format(t.upper()),samp.ix[samp.platform=='twitter',:].shape[0]
		print '{} / INSTAGRAM total:'.format(t.upper()),samp.ix[samp.platform=='instagram',:].shape[0]
		if t == 'depression':
			print
			print '{} / TWITTER cesd >= 22:'.format(t.upper()), samp.ix[(samp.platform=='twitter') & (samp.cesd > 21), :].shape[0]
			print '{} / INSTAGRAM cesd >= 22:'.format(t.upper()), samp.ix[(samp.platform=='instagram') & (samp.cesd > 21), :].shape[0]
		if t == 'ptsd':
			print
			print '{} / TWITTER tsq >= 6:'.format(t.upper()), samp.ix[(samp.platform=='twitter') & (samp.tsq > 5), :].shape[0]
			print '{} / INSTAGRAM tsq >= 6:'.format(t.upper()), samp.ix[(samp.platform=='instagram') & (samp.tsq > 5), :].shape[0]
		print
		print

	print 'SAMPLE SIZES: control populations'
	print

	for t in table:
		q = 'select username, platform from control where {}="No" and disqualified=0'.format(t)
		samp = pd.read_sql_query(q,conn)
		print '{} / TWITTER total:'.format(t.upper()),samp.ix[samp.platform=='twitter',:].shape[0]
		print '{} / INSTAGRAM total:'.format(t.upper()),samp.ix[samp.platform=='instagram',:].shape[0]
		print


def make_data_dict(params, condition, test_name, conn, doPrint=False):
	''' defines basic nested data structure for entire analysis '''

	data = {}
	for m in ['ig','tw']:

		data[m] = {'target':{}, 'control':{}, 'master':{}, 'before':{}, 'after':{}}

		if params['has_test']:
			data[m]['cutoff'] = {}
			data[m]['cutoff']['unames'] = pd.read_sql_query(params[m]['q']['t']['unames'],conn)

			if doPrint:
				print  ('Number {med} {cond} subjects above {test} cutoff:'.format(med=m,
																				   cond=condition,
																				   test=test_name),
						data[m]['cutoff']['unames'].shape[0])
	return data 


def get_pop_unames(params, m, conn, pop):
	''' Get all usernames for a given population '''

	return pd.read_sql_query(params[m]['q'][pop]['unames'], conn)


def get_photo_ratings(params, conn, pop, doPrint=False):
	''' Gets set of all Instagram photo ratings for either target or control condition ''' 

	d = pd.read_sql_query(params['ig']['q'][pop]['photo_ratings'], conn)
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


def get_meta_ig(params, conn, pop, doPrint=False):
	''' get data from meta_ig table '''

	if pop == 't':
		d = pd.read_sql_query(params['ig']['q'][pop]['meta_ig'], conn)

	elif pop == 'c':
		unames = pd.read_sql_query(params['ig']['q'][pop]['unames'],conn)
		all_meta_ig = pd.read_sql_query(params['ig']['q']['all_meta_ig'],conn)
		d = all_meta_ig.ix[all_meta_ig.username.isin(unames.username),:]
	
	if doPrint:
		print 'Num META_IG {} entries:'.format(pop.upper()), d.shape[0]
	return d


def get_hsv(data, m, params, conn, pop, pop_long, doPrint=False):

	hsv = pd.read_sql_query(params[m]['q']['all_hsv'],conn)
	hsv.dropna(inplace=True)
	
	if doPrint:
		print 'Number photos with HSV ratings (all conditions):', hsv.shape[0]

	unames = get_pop_unames(params, m, conn, pop)
	metaig = get_meta_ig(params, conn, pop)
	urls = metaig.ix[metaig.username.isin(unames.username),'url'].values

	if doPrint:
		print 'Num HSV-rated photos with URL in {}, before dropping duplicates:'.format(pop_long.upper()), hsv.ix[hsv.url.isin(urls),:].shape[0]

	data[m][pop_long]['hsv'] = hsv.ix[hsv.url.isin(urls),['url','hue','saturation','brightness']].drop_duplicates(subset='url').copy()

	if doPrint:
		print 'Num HSV-rated photos with URL in {}, after dropping duplicates:'.format(pop_long.upper()), data[m][pop_long]['hsv'].shape[0]


def consolidate_data(d, d2, m, pop_long, kind, data):
	''' merges dfs, adds 0/1 class indicator variable '''

	data[m][pop_long][kind] = d.merge(d2, how='left',on='url')

	if pop_long == 'target':
		cl = 1
	elif pop_long == 'control':
		cl = 0

	data[m][pop_long][kind]['target'] = cl

	if pop_long == 'target':
		for date in ['diag','susp']:
			data[m][pop_long][kind]['before_{}'.format(date)] = 0
			data[m][pop_long][kind].ix[data[m][pop_long][kind]['from_{}'.format(date)] < 0, 'before_{}'.format(date)] = 1
	else:
		for date in ['diag','susp']:
			data[m][pop_long][kind]['before_{}'.format(date)] = np.nan 

	print 'Shape of consolidated {} {} data:'.format(pop_long.upper(), kind.upper()), data[m][pop_long][kind].shape 



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
	x = data[m][pop]['ratings']
	print 'Data column types for medium: {}, population: {}, condition: {}'.format(m, pop, condition)
	for c in x.columns:
		print x[c].dtype


def find_chartype(data,col):
	''' find the rows where raters used strings as photo ratings '''

	x = data['ig']['target']['ratings']
	ixs = np.where(x[col].apply(type)==unicode)[0]
	print 'Indices where raters used strings for ratings instead of numeric values:', ixs 
	print 'Total cases:', ixs.shape[0]
	print 'Actual cases:'
	print data['ig']['target']['ratings'].ix[ixs,['happy','sad','interesting','likable']]


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

	data['ig']['target']['ratings'].replace(to_replace=key,inplace=True)


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

	for period in periods:
		data[m]['target'][period] = {}

		for turn_point in ['from_diag','from_susp']:
			if period == 'before':
				subset = (data[m]['target']['all'][turn_point] < 0)
			elif period == 'after':
				subset = (data[m]['target']['all'][turn_point] >= 0)
				
			data[m]['target'][period][turn_point] = {'all':data[m]['target']['all'].ix[subset,:]}
		
			if doPrint:
				print 'Subset shape for {} {} {}:'.format(m.upper(),period.upper(),turn_point.upper()), data[m]['target'][period][turn_point]['all'].shape
	print 


def add_class_indicator(gbdf, pop, doPrint=False):
	''' Add a target class column indicator variable, for later merging '''

	if pop == 'target':
		cl = 1
	elif pop == 'control':
		cl = 0

	gbdf['target'] = cl


def make_groupby(df, m, pop, params, gb_types, 
				 period=None, turn_point=None, doPrint=False):
	''' Create aggregated datasets 

		We collect mean and variance for each numerical measure, 
		and we combine all of the text descriptions

		Note: This kind of groupby results in a MultiIndex colum format.  
		The code after the actual groupby flattens the column hierarchy.'''

	df['gb'] = {}
	
	for gb_type in gb_types:

		if gb_type == 'url':
			gb_list = ['username','url']
			to_group_df = df['all'] # in a sense, this is the 'outermost' groupby we do, on original df
		elif gb_type == 'username':
			gb_list = [gb_type]
			to_group_df = df['gb']['url'] # this groupby is acting on the gb-url aggregate df
		elif gb_type == 'created_date':
			gb_list = ['username','created_date']
			to_group_df = df['gb']['url'] # this groupby is acting on the gb-url aggregate df

			
		# testing
		#print 'ROUND:', m, pop, gb_type 
		#print 'to_group_df shape:', to_group_df.shape
		#print to_group_df.columns

		df['gb'][gb_type] = (to_group_df.groupby(gb_list)
										.agg(params[m]['agg_func'][gb_type])
							)
		# testing
		#print 'gb df shape:', df['gb'][gb_type].shape

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

		if doPrint:
			print 'final gb shape [{} {} {} {} {}]:'.format(m, pop, gb_type, period, turn_point), df['gb'][gb_type].shape
		
		# testing
		#print 'final gb columns:'
		#print df['gb'][gb_type].columns

	if doPrint:
		print 


def merge_to_master(master, target, control, m, varset, gb_type, doPrint=False):
	''' merge target and control dfs into master df '''

	c = control[gb_type]
	t = target[gb_type]
	if gb_type == 'url':
		subset = ['hue','saturation','brightness']
	else:
		subset = ['hue|mean','saturation|mean','brightness|mean']
	master[gb_type] = pd.concat([c,t], axis=0).dropna(subset=subset)
	master[gb_type] = master[gb_type][ varset[gb_type]['full'] ]

	if doPrint:
		print 'Master {} {} nrows:'.format(m.upper(), gb_type.upper()), master[gb_type].shape[0]


def compare_density(df, m, gbtype, varset, ncols=4):
	''' Overlays density plots of target vs control groups for selected aggregation type '''

	print 'target vs control for {} {}-groupby:'.format(m.upper(), gbtype.upper())
	plt.figure()
	gb = df[gbtype].groupby('target')

	numvars = float(len(varset[gbtype]['means']))
	nrows = np.ceil(numvars/ncols).astype(int)
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 3*nrows), tight_layout=True)

	for ax, p in zip(axes.ravel(), varset[gbtype]['means']):
		for k, v in gb[p]:
			sns.kdeplot(v, ax=ax, label=str(k)+":"+v.name)
	plt.show()


def corr_plot(df, m, gb_type, varset):

	print 'Correlation matrix:'

	metrics = varset[gb_type]['means']
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


def print_model_summary(fit,ctype,target,title,X_test,y_test,labels):
	''' formats model output in a notebook-friendly print layout '''

	print 'MODEL: {} {} ({}):'.format(fit['name'],target,title)
	print 'NAIVE ACCURACY:'.format(ctype), round(fit['clf'].score(X_test,y_test),3)
	print
	print 'CONFUSION MATRIX ({}):'.format(ctype)
	
	# for confusion matrix
	known_0 = labels['known_0']
	known_1 = labels['known_1']
	pred_0 = labels['pred_0']
	pred_1 = labels['pred_1']

	cm_df = pd.DataFrame(confusion_matrix(y_test, fit['clf'].predict(X_test)), 
						 columns=[pred_0,pred_1], 
						 index=[known_0,known_1])
	print cm_df
	print 
	print 'Proportion of {} in {}:'.format(pred_1,known_0), round(cm_df.ix[known_0,pred_1] / float(cm_df.ix[known_0,:].sum()),3)
	print 'Proportion of {} in {}:'.format(pred_1,known_1), round(cm_df.ix[known_1,pred_1] / float(cm_df.ix[known_1,:].sum()),3)
	print
	print


def roc_wrapper(fits, ctype, y_test, X_test):
	proba = True
	skip = 0
	labe = 5
	
	if ctype == 'lr':
		skip = 1
		labe = 100
	elif ctype == 'svc':
		skip = 1
		labe = 100
		proba = False
	elif ctype == 'rf':
		skip = 1
		labe = 100
		
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


def make_models(d, test_size=0.3, clf_types=['lr','rf','svc'], 
				include=True, tall_plot=False, n_est=100, kernel='rbf',
				labels={'known_0':'known_control',
						'known_1':'known_target',
						'pred_0':'pred_control',
						'pred_1':'pred_target'}):
	mdata = d['data']
	title = d['name']
	unit = d['unit']
	target = d['target'] 
	feats = d['features']
	
	if include:
		mask = mdata.columns.isin(feats)
	else:
		mask = ~mdata.columns.isin(feats)
		
	model_feats = mdata.columns[mask]
	
	X = mdata[model_feats]
	y = mdata[target]

	imp = Imputer(strategy='median')
	X = imp.fit_transform(X,y)
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
	
	fits = {}
	
	fits['lr'] = {'name':'Logistic Regression','clf':logregcv(class_weight='auto')}
	fits['rf'] = {'name':'Random Forests','clf':RFC(n_estimators=n_est)}
	fits['svc'] = {'name':'Support Vector Machine','clf':SVC(class_weight='auto', kernel=kernel, probability=True)}
	
	for ctype in clf_types:
		fits[ctype]['clf'].fit(X_train,y_train)
			
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

		roc_wrapper(fits, ctype, y_test, X_test)
		
		if ctype == 'rf':
			importance_wrapper(fits, ctype, model_feats, title, tall_plot)
	
		output[ctype] = fits[ctype]['clf']

	return output


def ttest_output(a, b, varset, ttype, correction=True, alpha=0.05, method='bonferroni'):
	''' performs independent-samples t-tests with bonferroni correction '''

	pvals = []

	for metric in varset:
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


def master_actions(master, target, control, condition, m, params, gb_types, report,
				   save_to_file = False, density = True, corr = False, ml = False, nhst = True):
	''' Performs range of actions on master data frame, including plotting, modeling, and saving to disk. 
		
		Note: "Master" may refer to any set of target+control data, including timeline subsets. 
		In other words, both the full dataset and subsets of full dataset may be passed as "master" argument.'''

	master['model'] = {}

	for gb_type in gb_types:
		
		# merge target, control, into master
		print 
		print 'Merge to master: {} {}'.format(report, gb_type)
		merge_to_master(master, target, control, m, params['vars'], gb_type)

		master['model'][gb_type] = {}
		
		if density:
			compare_density(master, m, gb_type, params['vars'])

		if corr:
			corr_plot(master, m, gb_type, params['vars'])

		if save_to_file:
			# save csv of target-type // groupby-type
			csv_name = '{cond}_{m}_{gbt}_{r}.csv'.format(cond=condition,m=m,gbt=gb_type,r=report)
			master[gb_type].to_csv(csv_name, encoding='utf-8')

		if ml:
			model_df = {'name':'Models: {} {}'.format(report, gb_type),
						'unit':gb_type,
						'data':master[gb_type],
						'features':params['excluded'][gb_type],
						'target':'target'
					   }

			output = make_models(model_df, test_size=0.3, include=False)

			for k in output.keys():
				master['model'][gb_type][k] = output[k]

		if nhst:
			ttest_out = ttest_wrapper(master, gb_type, params['vars'])
			master['model'][gb_type]['ttest'] = ttest_out[0]
			master['model'][gb_type]['ttest_pvals'] = ttest_out[1]


def before_vs_after(df, gb_type, condition, params):
	for date in ['diag','susp']:
		print
		print 'before vs after (target: {}) for {}-groupby, based on {}_date:'.format(condition, gb_type, date)

		splitter = 'before_{}'.format(date)
		gb = df[gb_type].groupby(splitter)

		plt.figure()
		fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(14, 3), 
								 tight_layout=True)
		for ax, p in zip(axes.ravel(), params['vars'][gb_type]['means']): # don't use gb_type here, it's only for url
			for k, v in gb[p]:
				sns.kdeplot(v, ax=ax, label=str(k)+":"+v.name)
		plt.show()
		
		ttest_wrapper(df, gb_type, params['vars'], split_var=splitter, ttype='ind')


