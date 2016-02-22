from flask import jsonify
from skimage import io, data, color
import numpy as np
import collections
import datetime
from dateutil import parser

##  get_hsv: converts image from rgb to hsv (Hue, Saturation, V=Brightness)
def get_hsv(im):
	im = color.rgb2hsv(im) # http://scikit-image.org/docs/dev/api/skimage.color.html#rgb2hsv
	hsv = []
	for i in xrange(len(im.shape)):
		if len(im.shape) == 2: #b/w photo
			imdata = im[:,i]
		elif len(im.shape) == 3: #color photo
			imdata = im[:,:,i]
		avg = np.mean(imdata)
		hsv.append( avg )
	return hsv

##  extract_hsv: reads image, converts to hsv, stores in db table "hsv"
def extract_hsv(conn, url, uid, username):
	try:
		img = io.imread(url)
		h,s,v = get_hsv(img)
		query = "INSERT INTO hsv(uid, url, hue, saturation, brightness, username) VALUES ('{}','{}','{}','{}','{}','{}')".format(uid, url, h, s, v, username)
		with conn:
			cur = conn.cursor()
			cur.execute(query)
			conn.commit()
	except Exception,e:
		return 'extract_hsv error: {}'.format(str(e))
## gets metadata from instagram posts, stores on sqlite3 db table "meta_ig"
## NOTES:
## - Instagram API calls return a hodgepodge of dicts, lists, and custom objects.  It's not as simple as Twitter.
## - In a few cases, we've hard coded exactly the features we're looking for (image url, caption text)
## - We use an ordered dict to keep track of relevant key/value pairs for entering into SQL
def extract_meta_instagram(conn, medium, data, username, user_id, unique_id, table_data, log_msgs):
	try:
		table    = "meta_ig"
		fields   = table_data.ix[ table_data.table == table, "field"].values
		features = collections.OrderedDict()

		features['uid'] = unique_id
		features['instagram_user_id'] = user_id
		features['username'] = username
		features['ratings_ct'] = 0

		for key, val in vars(data).iteritems():
			if key == "images":
				features["url"] = data.images["low_resolution"].url

			elif key == "caption":
				try:
					features["caption"] = data.caption.text
				except:
					#print 'caption fail!'
					features["caption"] = ''

			elif key in fields:
				if isinstance(vars(data)[key], list):

					if key == "comments":
						try:

							features[key] =  '___'.join([com.text if com is not None else '' for com in vars(data)[key]]) 
							#print key
							#print features[key]
						except Exception,e:
							log_msgs.append('Failed on joining list of {}: {}'.format(key,str(e)))
					elif key == "tags":
						try:
							features[key] =  '___'.join([tag.name for tag in vars(data)[key]]) 
							#print key
							#print features[key]
						except Exception,e:
							log_msgs.append('Failed on joining list of {}: {}'.format(key,str(e)))
					else:
						try:
							features[key] =  '___'.join(vars(data)[key]) 
							#print key
							#print features[key]
						except Exception,e:
							log_msgs.append( 'Failed on joining list of keys: {}'.format(str(e)) )
				else:
					features[key] = vars(data)[key]
					
					#print key
					#print features[key]
			if key == "created_time":
				timestamp = vars(data)[key]
				yyyy_mm_dd = timestamp.strftime('%Y-%m-%d')
				features['created_date'] = yyyy_mm_dd	
				#print 'created_date:',yyyy_mm_dd
		
		query = "INSERT OR IGNORE INTO {table}{cols} VALUES(".format(table=table,cols=tuple(features.keys())) + ('?,' *len(features))[:-1] + ")"
	except Exception,e:
		print 'extract_meta_instagram error before db write: {}'.format(str(e))
	try:
		with conn:
			cur = conn.cursor()
			cur.executemany(query, (features.values(),))
			conn.commit()	
		return features["url"]
	except Exception, e:
		log_msgs.append( 'Error: {}, Query: {}'.format(str(e),query) )
		return '\n'.join(log_msgs)

## Gets metadata from Twitter posts, stores to db table "meta_tw" 
## NOTES: 
## - Unlike Instagram, it's easy to pass in the entire Tweet corpus and iterate
## - Some of the data we want is nested. The table_data df keeps track of the level each datum resides on.
## - Each iteration checks subsequent levels to see if the given key is available
## - We use an ordered dict to keep track of relevant key/value pairs for entering into SQL
def extract_meta_twitter(conn, medium, data, username, user_id, unique_id, table_data, log_msgs):

	table = "meta_tw"
	fields = table_data.ix[ table_data.table == table, "field"].values
	
	log_msgs.append("Total rows to extract for {}: {}".format(username, len(data)))
	for i,row in enumerate(data):
		if i%50==0:
			print 'Extracting metadata from row {}'.format(i)
			log_msgs.append('Extracting metadata from row {}'.format(i))
		features = collections.OrderedDict()
		features['uid'] = unique_id
		features['twitter_user_id'] = user_id
		features['username'] = username

		for k in row.keys():
			if k in fields:
				features[str(k)] = row[k]
				
			elif isinstance(row[k],dict):
				for k2 in row[k].keys():
					if (k2 in fields) and (table_data.level[(table_data.table==table) & (table_data.field==k2)].values == 2):
						if not row[k][k2]:
							features[str(k2)] = "None"
						elif isinstance(row[k][k2],dict):
							k3s = row[k][k2].keys()
							features[str(k3s[0])] = str(row[k][k2][k3s[0]]) 
						else:
							features[str(k2)] = "not_found"
			if str(k)=="created_at":
				timestamp = parser.parse(row[k])
				yyyy_mm_dd = timestamp.strftime('%Y-%m-%d')
				features['created_date'] = yyyy_mm_dd
				#print 'created_date:',yyyy_mm_dd

		query = "INSERT OR IGNORE INTO {table}{cols} VALUES(".format(table=table,cols=tuple(features.keys())) + ('?,' *len(features))[:-1] + ")"

		with conn:
			cur = conn.cursor()
			cur.executemany(query, (features.values(),))
			conn.commit()


## wrapper for extracting metadata
def extract_meta(conn, medium, data, username, user_id, unique_id, table_data, log_msgs):
	if medium == "twitter":
		return extract_meta_twitter(conn, medium, data, username, user_id, unique_id, table_data, log_msgs)
	elif medium == "instagram":
		return extract_meta_instagram(conn, medium, data, username, user_id, unique_id, table_data, log_msgs)
	else:
		return "Incorrect or missing social medium."