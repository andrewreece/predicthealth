# -*- coding: utf-8 -*-

from flask import jsonify
import boto
import boto.s3.connection
import sqlite3
import pandas as pd
from StringIO import StringIO 
import numpy as np
import requests 
import time, datetime
from os.path import join, dirname
from dotenv import load_dotenv
from random import randint
from twython import Twython

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

## GLOBALS ##
table_data_path = os.environ.get("TABLE_DATA_PATH")
table_goog_path = os.environ.get("TABLE_GOOG_PATH")
data_path = os.environ.get("DATA_PATH")
log_path = os.environ.get("LOG_PATH")
db_path = os.environ.get("DB_PATH")
s3_path = os.environ.get("S3_PATH")
bucket_name = os.environ.get("BUCKET_NAME")

def log(msgs,path,full_path_included=False,path_head=log_path):
	
	if full_path_included:
		log_path = path_head+path
	else:
		tstamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
		log_path = path_head+path+'{}.log'.format(tstamp)
		
	f = open(log_path,'w')
	f.write('\n'.join(msgs)+'\n\n')
	f.close()


def get_table_data(path=table_goog_path):
	r    = requests.get(table_goog_path)
	data = r.content
	return pd.read_csv(StringIO(data))

##  connect_db: initiates sqlite3 connection
def connect_db():	
	return sqlite3.connect(db_path)

##  get_tokens: gets api tokens for twitter, instagram, s3 (specified by 'medium')
def get_tokens(conn, medium, username='MASTER'):
	if username == 'MASTER':
		with conn:
			query = "SELECT consumer_key, consumer_secret, access_key, access_secret FROM tokens WHERE service='{}' AND username='{}'".format(medium,username)
			cur = conn.cursor()
			cur.execute(query)
			row = cur.fetchone()
			return (row[0], row[1], row[2], row[3])
	else:
		with conn:
			query = "SELECT consumer_key, consumer_secret FROM tokens WHERE service='{}' AND username='{}'".format(medium,'MASTER')
			cur = conn.cursor()
			cur.execute(query)
			row = cur.fetchone()
			ckey = row[0]
			csec = row[1]

			query = "SELECT access_key, access_secret FROM tokens WHERE service='{}' AND username='{}'".format(medium,username)
			cur = conn.cursor()
			cur.execute(query)
			row = cur.fetchone()
			akey = row[0]
			asec = row[1]
		return (ckey, csec, akey, asec)

## cache Twitter/Instagram data to s3 
## accessible at s3://predicthealth/<medium>/<username>
def s3_cache(conn, medium, data, username,path_head=data_path):
	try:
		fname = path_head+medium+'/'+username
		np.savetxt(fname, data, fmt="%s")
		_, _, s3_access_key, s3_secret_key = get_tokens(conn,"s3")
		s3_conn = boto.connect_s3( aws_access_key_id = s3_access_key, aws_secret_access_key = s3_secret_key)
		s3_bucket = s3_conn.get_bucket(bucket_name)
		k = s3_bucket.new_key(medium+'/'+username)
		k.set_contents_from_filename(fname)
		return jsonify({"success":"all "+medium+" posts for user: "+username+" were collected and cached. retrieve at: "+s3_path+medium+"/"+username})
	except Exception,error:
		return error 

## on data collection attempt, update user in 'usernames' table as either 'collected' or save the error in case of fail
def update_user_status(conn, medium, username, status):
	if status == "success":
		conn  = connect_db()
		query = "UPDATE usernames SET collected=1 WHERE username='{}' AND medium='{}'".format(username,medium)
		## deletes any previous recorded errors that prevented successful collection
		query2 = "UPDATE usernames SET collect_error='' WHERE username='{}' AND medium='{}'".format(username,medium)

		with conn:
			cur = conn.cursor()
			cur.execute(query)
			cur.execute(query2)
		conn.commit()
	else:
		conn  = connect_db()
		query = "UPDATE usernames SET collect_error='{}' WHERE username='{}' AND medium='{}'".format(status,username,medium)
		with conn:
			cur = conn.cursor()
			cur.execute(query)
		conn.commit()

## registers photo in photo_ratings table with default values
def register_photo(conn, url, uid, tname="photo_ratings"):
	try:
		cols = tuple(['uid', 'rater_id', 'url', 'happy', 'sad', 'likable', 'interesting', 'one_word', 'description'])
		vals = tuple([ uid,      '',      url,     0,      0,       0,           0,           '',          ''      ])
		query = "INSERT OR IGNORE INTO {table}{cols} VALUES{vals}".format(table=tname,cols=cols,vals=vals)
		with conn:
			cur = conn.cursor()
			cur.execute(query)
		conn.commit()
		return "register_ok"
	except Exception, e:
		return "reg_photo__"+str(e)
