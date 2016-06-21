from flask import Flask, jsonify, request, current_app, session, redirect, make_response, current_app

import numpy as np
import pandas as pd
import time, datetime, re, os, ssl, json
from datetime import timedelta
from functools import wraps, update_wrapper
from random import randint
from urllib import unquote

from os.path import join, dirname
from dotenv import load_dotenv

from twython import Twython, TwythonError
from instagram.client import InstagramAPI

from nocache import nocache
import util 
from collect import collect_instagram, collect_twitter
from verify import verify_instagram, verify_twitter

## the core flask app
app = Flask(__name__)

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

## GLOBALS ##
base_path = os.environ.get("BASE_PATH")
cert_path = os.environ.get("CERT_PATH")
cert_key_path = os.environ.get("CERT_KEY_PATH")
data_directory = os.environ.get("DATA_PATH")
oauth_url_base = os.environ.get("OAUTH_URL_BASE")
acquire_url_base = os.environ.get("ACQUIRE_URL_BASE")

## we need this to run https, which we need for CORS issues on Qualtrics (verify() function)
context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.load_cert_chain(cert_path,cert_key_path)

## gets table name / field / datatype for all tables as a Pandas data frame
table_data = util.get_table_data()



def crossdomain(origin=None, methods=None, headers=None,
				max_age=21600, attach_to_all=True, automatic_options=True):
	''' Super-helpful decorator to avoid CORS issues when calling from Qualtrics (or anywhere else)
		Source: https://blog.skyred.fi/articles/better-crossdomain-snippet-for-flask.html 
		Also: http://stackoverflow.com/questions/22181384/javascript-no-access-control-allow-origin-header-is-present-on-the-requested '''
	
	if methods is not None:
		methods = ', '.join(sorted(x.upper() for x in methods))
	if headers is not None and not isinstance(headers, basestring):
		headers = ', '.join(x.upper() for x in headers)
	if not isinstance(origin, basestring):
		origin = ', '.join(origin)
	if isinstance(max_age, timedelta):
		max_age = max_age.total_seconds()

	def get_methods():
		if methods is not None:
			return methods

		options_resp = current_app.make_default_options_response()
		return options_resp.headers['allow']

	def decorator(f):
		def wrapped_function(*args, **kwargs):
			if automatic_options and request.method == 'OPTIONS':
				resp = current_app.make_default_options_response()
			else:
				resp = make_response(f(*args, **kwargs))
			if not attach_to_all and request.method != 'OPTIONS':
				return resp

			h = resp.headers
			h['Access-Control-Allow-Origin'] = origin
			h['Access-Control-Allow-Methods'] = get_methods()
			h['Access-Control-Max-Age'] = str(max_age)
			h['Access-Control-Allow-Credentials'] = 'true'
			h['Access-Control-Allow-Headers'] = \
				"Origin, X-Requested-With, Content-Type, Accept, Authorization"
			if headers is not None:
				h['Access-Control-Allow-Headers'] = headers
			return resp

		f.provide_automatic_options = False
		return update_wrapper(wrapped_function, f)
	return decorator


def register_user(medium, userid, username, uid, post_ct, conn, token, secret):
	''' on successful verification, enter username into 'usernames' table '''

	tstamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
	log_dir = 'registeruser/{med}__{u}__{t}.log'.format(med=medium,u=username,t=tstamp)
	log_msgs = []
	log_msgs.append('\nStarting register_user for username {} [service id: {}] [total posts: {}]'.format(username,userid,post_ct))

	try:
		table  = "usernames"
		fields = table_data.ix[ table_data.table == table, "field"].values
		fields_after_tableid = fields[1:]
		# INSERT OR IGNORE: http://stackoverflow.com/questions/15535355/insert-only-if-id-doesnt-exist
		query  = "INSERT OR IGNORE INTO {table}{cols} VALUES(".format(table=table, cols=tuple(fields_after_tableid)) + ('?,' *len(fields_after_tableid))[:-1] + ")"
		
		# defaults
		collected = 0
		validated = 0
		collect_error = ''

		with conn:
			cur = conn.cursor()
			cur.execute(query,(uid, userid, username, post_ct, medium, collected, collect_error, validated)) # 'uid' is a unique index
			conn.commit()

		# store oauth creds for this user in `tokens` table
		query = "INSERT OR IGNORE INTO tokens (user_id, username, service, access_key, access_secret) VALUES ('{}','{}','{}','{}','{}')".format(userid, username, medium, token, secret)
		with conn:
			cur = conn.cursor()
			cur.execute(query)
		conn.commit()

		log_msgs.append('register success for user: {} [medium:{}]'.format(username,medium))
		
		util.log(log_msgs,log_dir,full_path_included=True)
		return final_report

	except Exception,error:
		log_msgs.append('Error in registering user: {}'.format(error))
		util.log(log_msgs,log_dir,full_path_included=True)
		return error 

##  support_jsonp: wraps JSONified output for JSONP
##  we need this for returning values to Qualtrics - part of the CORS precautions that modern browsers implement
def support_jsonp(f):
	@wraps(f)
	def decorated_function(*args, **kwargs):
		callback = request.args.get('callback', False)
		if callback:
			content = str(callback) + '(' + str(f(*args,**kwargs).data) + ')'
			return current_app.response_class(content, mimetype='application/javascript')
		else:
			return f(*args, **kwargs)
	return decorated_function


@app.route('/verify2/<medium>/<uname>')
@support_jsonp
def verify2(medium, uname):
	try:
		medium = medium.lower() 
		conn   = util.connect_db()

		query = "SELECT username FROM usernames WHERE username='{}' AND medium='{}'".format(uname,medium)
		cur   = conn.cursor()
		cur.execute(query)
		rows  = cur.fetchall() #fetchone?
		verified = (rows[0][0] == uname)
		return jsonify({"verified":verified})

	except Exception,e:
		return str(e)


##  create_table: creates new db table, using parameters stored in table_data df (global)
##  this is only called directly via URI - it's not part of the verify/collect pipeline
@app.route("/createtable/<conn>/<table>")
@nocache
def create_table(conn, table):
	try:
		if conn == "new": conn = util.connect_db()

		this_table = table_data.ix[ table_data.table == table, :]
		this_table.reset_index(drop=True,inplace=True)

		query = "CREATE TABLE {}(".format(table)

		for ix in xrange(this_table.shape[0]):
			query = query + str(this_table.field[ix]) + " " + str(this_table.type[ix]) + ", "

		query = query[:-2] + ")" # drops trailing comma, adds closing parenthesis
		drop_command = "DROP TABLE IF EXISTS {}".format(table)

		with conn:
			cur = conn.cursor()
			cur.execute(drop_command)
			cur.execute(query)

		conn.commit()
		conn.close()
		return query
	except Exception, e:
		return str(e)+query


''' Called by qualtrics survey for photo raters, gets instagram photos to be rated
selects from urls which have not been rated at least N times (currently N=3)
upon successful completion of rating survey, N is incremented by 1 for each photo rated
photo urls are found in meta_ig '''
@app.route("/getphoto/<cond>/<kind>")
@nocache
def get_photo(kind, cond):

	try:
		conn = util.connect_db()
		max_ratings_ct = 3
		sample_size = 20
		query = 'select * from photo_ratings_{}_{}'.format(cond, kind)
		db = pd.read_sql_query(query, conn)
		gb = db.groupby('url').count()
		# we use 'rater_id' because we need to pick a column that keeps track of the groupby count. happy, sad, etc. also work
		urls_to_rate = gb.ix[gb.rater_id < max_ratings_ct,'rater_id'].sample(sample_size).index.tolist()
		urls = {}
		for i,row in enumerate(urls_to_rate):
			urls["url"+str(i)] = row
		return jsonify(urls)
	except Exception,e:
		return jsonify({"url0":str(e)})


@app.route("/addrating/<cond>/<kind>/<rater_id>/<happy>/<sad>/<likable>/<interesting>/<one_word>/<description>/<encoded_url>", methods=['POST', 'GET', 'OPTIONS'])
@crossdomain(origin='*')
def add_rating(cond,kind,rater_id,happy,sad,likable,interesting,one_word,description,encoded_url,table="photo_ratings"):
	''' Called from Qualtrics after a user has rated a photo
		- Writes ratings data to photo_ratings db
		- Increments ratings_ct in meta_ig for that URL '''

	tstamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
	target_table = "_".join([table, cond, kind])

	''' The reason for the regex sub on url below is because Flask parses forward slash (/) 
		(as well as the encoded %2F) before we can even access incoming parameter strings programmatically.  
		Since we're passing it a url, it barfs when it discovers slashes in the parameter.
		So we converted / to _____ (5 underscores) on the Qualtrics side, and we back-translate here.
		5 underscores may be overkill, but _ or even __ is not out of the question in a url...just being safe. '''

	try:

		url = unquote(encoded_url).replace("_____","/")

		valid_ratings = False if description == "_" else True 

		fields = ['rater_id','happy','sad','likable','interesting','one_word','description']
		values = map(unquote, [rater_id, happy, sad, likable, interesting, one_word, description])
		
		log_dir = 'ratings/{rid}__{t}.log'.format(rid=values[0],url=url,t=tstamp)
		log_msgs = []
		log_msgs.append('\nStarting add_rating for photo url {} [rater id: {}]'.format(url,values[0]))

		conn = util.connect_db()

		if valid_ratings:

			with conn:
				query1 = "UPDATE meta_ig SET ratings_ct=ratings_ct+1 WHERE url='{}'".format(url)
				cur = conn.cursor()
				cur.execute(query1)

				try:
					query2 = 'insert into {}(url, rater_id, happy, sad, likable, interesting, one_word, description) values (?,?,?,?,?,?,?,?)'.format(target_table)
					qvals = (url,)+tuple(values)
					cur.execute(query2,qvals)
				except Exception,e:
					return query2+"__"+str(e)
			conn.commit()

			log_msgs.append('\nRating for url: {} [rater id: {}] stored successfully!'.format(url,values[0]))
			util.log(log_msgs,log_dir,full_path_included=True)
			return query2

		else:

			with conn:
				# getting to this part of the logic branch means a user said that no photo appeared for this url
				# we flag it with the code 999 so we can review later to determine whether it really is a dead url
				# UPDATE (APR 08 2016): This is legacy code now, we /sunsetted/ the user-inputted no-photo marker
				query1 = "UPDATE meta_ig SET valid_url=999 WHERE url='{}'".format(url)
				cur = conn.cursor()
				cur.execute(query1)
			conn.commit()

			log_msgs.append('\nFAILED TO LOAD url: {} [rater id: {}]'.format(url,values[0]))
			util.log(log_msgs,log_dir,full_path_included=True)
			return 'failed to load photo url in survey'

	except Exception,e:
		log_msgs.append('\nError recording rating for url: {} [rater id: {}] [ERROR: {}]'.format(url,values[0],str(e)))
		util.log(log_msgs,log_dir,full_path_included=True)
		return str(e)

@app.route("/ratingsreport/<cond>/<kind>")
def ratings_report(cond, kind):

	tstamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
	log_dir = 'ratings/REPORT__{t}.log'.format(t=tstamp)
	log_msgs = []

	conn = util.connect_db()
	max_ratings_ct = 3
	sample_size = 20
	query = 'select * from photo_ratings_{}_{}'.format(cond,kind)
	db = pd.read_sql_query(query, conn)
	gb = db.groupby('url').count()
	log_msgs = ['Photos in {}-{} with NO ratings: {}'.format(cond, kind, np.sum(gb['rater_id'] == 0))]
	max_ratings_on_record = 12
	for i in range(max_ratings_on_record):
	    ct = np.sum(gb['rater_id'] > i)
	    log_msgs.append('Photos with more than {} ratings: {}'.format(i,ct))

	try:
		util.log(log_msgs,log_dir,full_path_included=True)
		return '<br />'.join(log_msgs)
	except Exception,e:
		return str(e)

@app.route("/oauth/<medium>/<username>")
def get_auth(medium,username):
	try:
		conn = util.connect_db()
		callback = acquire_url_base+'?medium={}&username={}'.format(medium,username)

		tokens = util.get_tokens(conn, medium)

		if medium == "twitter":

			session['APP_KEY'] = tokens[0]
			session['APP_SECRET'] = tokens[1]

			twitter = Twython(session['APP_KEY'], session['APP_SECRET'])

			
			auth = twitter.get_authentication_tokens(callback_url=callback)

			session['OAUTH_TOKEN'] = auth['oauth_token']
			session['OAUTH_TOKEN_SECRET'] = auth['oauth_token_secret']

			return redirect(auth['auth_url'])

		elif medium == "instagram":
			CONFIG = {
			'client_id': tokens[2],
			'client_secret': tokens[3],
			'redirect_uri': callback
			}

			api = InstagramAPI(**CONFIG)
			
			session['APP_KEY'] = tokens[2]
			session['APP_SECRET'] = tokens[3]

			url = api.get_authorize_url(scope=["basic"])

			return redirect(url)

	except Exception, e:
		return str(e)

@app.route("/acquire")
def acquire():

	try:
		alleged_user = request.args.get('username')
		medium = request.args.get('medium')
		oauth_url = oauth_url_base+'{}/{}'.format(medium, alleged_user)
	except Exception,e:
		return "Could not access request.args.get [ERROR: {}]".format(str(e))

	try:
		if medium == "twitter":
			oauth_verifier = request.args.get('oauth_verifier')

			twitter = Twython(session['APP_KEY'], session['APP_SECRET'],
						  session['OAUTH_TOKEN'], session['OAUTH_TOKEN_SECRET'])

			final_step = twitter.get_authorized_tokens(oauth_verifier)

			OAUTH_TOKEN = final_step['oauth_token']
			OAUTH_TOKEN_SECRET = final_step['oauth_token_secret']

			twitter2 = Twython(session['APP_KEY'], session['APP_SECRET'],
						  OAUTH_TOKEN, OAUTH_TOKEN_SECRET)

			ucreds = twitter2.verify_credentials()
			username = ucreds["screen_name"]
			userid = ucreds["id_str"]
			post_ct = ucreds["statuses_count"]

		elif medium == "instagram":
			try:
				callback = acquire_url_base+'?medium={}&username={}'.format(medium,alleged_user)
				CONFIG = {
					'client_id': session['APP_KEY'], 
					'client_secret': session['APP_SECRET'],
					'redirect_uri': callback
				}

				api = InstagramAPI(**CONFIG)

				code = request.args.get('code')

				if code:


					access_token, user_info = api.exchange_code_for_access_token(code)
					if not access_token:
						return 'Could not get access token'

					# Sessions are used to keep this data 
					OAUTH_TOKEN = access_token

					api = InstagramAPI(access_token=access_token, client_secret=CONFIG['client_secret'])
					userid = user_info['id'] 
					username = user_info['username'] 	
					post_ct = api.user().counts['media']
				else:
					return "Uhoh no code provided"
			except Exception,e:
				return "Error in acquire step 1: "+str(e)

		try:
			
			if username == alleged_user:
				unique_id = np.random.randint(1e10)
				conn = util.connect_db()

				if medium=="twitter":
					register_user(medium, userid, username, unique_id, post_ct, conn, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
				elif medium=="instagram":
					register_user(medium, userid, username, unique_id, post_ct, conn, OAUTH_TOKEN,'')

				return '<span style="font-size:24pt;color:green;">USERNAME {} CONFIRMED!</span>'.format(username)

			else:
				return 'The username you just used to grant access to this app (<b>{actual}</b>) is not the same username you provided in the study survey (<b>{alleged}</b>). <br />Please go back to <a href="{oauth}">the app authorization page</a> and make sure you are logged in as the correct user, and try again. <br />(You may need to log out of your account first in a separate window.)'.format(actual=username,alleged=alleged_user,oauth=oauth_url)
		except Exception,e:
			return "Error in acquire step 2:"+str(e)

	except Exception,e:
		return 'There was an error, please go back to {} and retry. [ERROR: {}]'#.format(oauth_url,str(e))


@app.route("/ping")
def ping():
	''' For keeping server alive and responsive, runs once an hour on cron '''
	return "ping ok"
	
@app.route("/testoauth/<medium>/<username>")
def test_oauth(medium, username):
	return 'For testing only.'


app.secret_key = os.environ.get("SECRET_KEY")

## run flask
if __name__ == '__main__':
	
	app.run(host='127.0.0.1',port=12340, 
		debug = True, ssl_context=context)

## setting default values for optional paths:
## http://stackoverflow.com/questions/14032066/flask-optional-url-parameters
