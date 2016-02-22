from flask import Flask, jsonify, request, current_app, session, redirect
import ssl
import numpy as np
import pandas as pd
import re
import time, datetime
import os
from functools import wraps
from random import randint
from os.path import join, dirname
from dotenv import load_dotenv

import tweepy
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

## register_user: on successful verification, enter username into 'usernames' table
def register_user(medium, userid, username, uid, post_ct, conn, token, secret):
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

## get_photo: called by qualtrics survey for photo raters, gets instagram photos to be rated
## - selects from urls which have not been rated at least N times (currently N=3)
## - upon successful completion of rating survey, N is incremented by 1 for each photo rated
## - photo urls are found in meta_ig
@app.route("/getphoto/<max_ratings>/<photo_ct>")
def get_photo(max_ratings, photo_ct):
	conn  = util.connect_db()
	query = "SELECT url FROM meta_ig WHERE ratings_ct < {N} LIMIT {ct}".format(N=max_ratings,ct=photo_ct) 
	urls  = {}
	cur   = conn.cursor()
	cur.execute(query)
	rows  = cur.fetchall()

	for i,row in enumerate(rows):
		urls["url"+str(i)] = row[0]

	return jsonify(urls)

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

## add_rating: called from Qualtrics after a user has rated a photo; increments ratings_ct
## note: the reason for the regex sub is because Flask parses / (as well as the encoded %2F) before we can even access
##       incoming parameter strings.  since we're passing it a url, it barfs when it discovers slashes in the parameter
##       so we just converted / to _____ (5 underscores) on the Qualtrics side, and we back-translate here.
##       * 5 underscores may be overkill, but _ or even __ is not out of the question, just wanted to be safe.
@app.route("/addrating/<rater_id>/<happy>/<sad>/<likable>/<interesting>/<one_word>/<description>/<encoded_url>")
def add_rating(rater_id,happy,sad,likable,interesting,one_word,description,encoded_url,table="photo_ratings"):

	try:
		url    = re.sub("_____","/",encoded_url)
		fields = ['rater_id','happy','sad','likable','interesting','one_word','description']
		values = [rater_id, happy, sad, likable, interesting, one_word, description]
		conn   = util.connect_db()

		with conn:
			query1 = "UPDATE meta_ig SET ratings_ct=ratings_ct+1 WHERE url='{}'".format(url)
			cur = conn.cursor()
			cur.execute(query1)

			for i,field in enumerate(fields):
				try:
					val = values[i]
					query2 = "UPDATE {table} SET {f}='{val}' WHERE url='{url}'".format(table=table, f=field, val=val, url=url)
					cur.execute(query2)
				except Exception,e:
					return query2+"__"+str(e)
		conn.commit()

		return query2
	except Exception,e:
		return str(e)

@app.route("/do/<action>")
def invite(action):
	try:

		inv = util.Inviter(data_directory)
		if action == "test":
			inv.conditions = ["TEST"] # comment out when you go live
			action = "invite"

		for cond in inv.conditions:
			inv.do(action, cond)

		return "Invitations sent successfully!"
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
