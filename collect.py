from flask import jsonify
from extract import extract_meta, extract_hsv
import util
import time


def collect_instagram(api,username,unique_id,conn,table_data):
	medium = "instagram"
	sleep_cutoff = 10
	warning_count = 0
	collect_error = False
	min_posts_per_call = 5

	log_dir = 'collect/{}/{}.log'.format(medium,username)
	log_msgs = []

	# get diagnosis date and suspected date
	query = "SELECT diag_year, diag_month, diag_day, days_suspected"
	try:

		uid = api.user().id
		num_statuses = api.user().counts['media']

		if num_statuses == 0:
			log_msgs.append('USER: {} [ID: {}] has zero posts!'.format(username,uid))
		else:
			log_msgs.append('START LOG FOR USER: {} | ID: {} | TOTAL POSTS: {}\n'.format(username, uid, num_statuses))
			
			user_posts = []

			recent_media, next_ = api.user_recent_media()

			for media in recent_media:
				mobj = {}
				raw = api.media(media.id)
				# we return photo_url for use in extract_hsv()
				# (we could just run extract_hsv inside extract_meta, but this is better separability)	
				photo_url = extract_meta(conn, medium, raw, username, uid, unique_id, table_data, log_msgs) # extract metadata, store in meta_ig
				#print 'extract_meta return:'
				#print photo_url
				#print
				extract_hsv( conn, photo_url, unique_id, username ) # store in hsv table		
				util.register_photo(conn, photo_url, unique_id) # register photo in photo_ratings table
				user_posts.append(mobj)

				stolen = len(user_posts)
				calls_until_rate_limit = api.x_ratelimit_remaining

			while next_ and stolen < num_statuses and (not collect_error):

				if len(user_posts) < min_posts_per_call:
					warning_count += 1
				else:
					warning_count = 0
				if warning_count > 5:
					collect_error = True
					log_msgs.append('We hit a collect error, stolen: {}, calls_until_rate_limit: {}'.format(stolen,calls_until_rate_limit))
				
				more_media, next_ = api.user_recent_media(with_next_url=next_)
				for media in more_media:
					mobj = {}
					try:
						raw = api.media(media.id)

						# from extract_meta we return photo_url for use in extract_hsv()
						# (we could just run extract_hsv inside extract_meta, but this is better separability)
						photo_url = extract_meta(conn, medium, raw, username, uid, unique_id, table_data, log_msgs) # extract metadata, store in meta_ig

						extract_hsv( conn, photo_url, unique_id, username ) # store in hsv table
						util.register_photo(conn, photo_url, unique_id) # register photo in photo_ratings table
						user_posts.append(mobj)
					except Exception,error:
						log_msgs.append('Failed collection for media id: {}, Error: {}'.format(media.id,str(error)))
				stolen = len(user_posts)
				calls_until_rate_limit = api.x_ratelimit_remaining
				
				if stolen%50==0:
					print 'Num statuses collected for user: {}: {}'.format(username,stolen)
					print 'Total API calls left before rate limit: {}'.format(calls_until_rate_limit)
					log_msgs.append('Num statuses collected for user: {}: {}'.format(username,stolen))
					log_msgs.append('Total API calls left before rate limit: {}'.format(calls_until_rate_limit))

			log_msgs.append('Num statuses collected for user: {}: {}'.format(username,stolen))
			log_msgs.append('Total API calls left before rate limit: {}'.format(calls_until_rate_limit))

		data = user_posts
		util.update_user_status(conn, medium, username, "success") # update username as "collected"
		util.s3_cache(conn, medium, data, username) # cache raw blob in s3

		util.log(log_msgs, log_dir, full_path_included=True)
		
	except Exception,error:
		util.update_user_status(conn, medium, username, str(error))
		log_msgs.append("Error collecting {} for user: {} [ERROR: {}]".format(medium,username,str(error)))
		util.log(log_msgs, log_dir, full_path_included=True)


def collect_twitter(api,username,unique_id,conn,table_data):
	medium = "twitter"
	sleep_cutoff = 10
	warning_count = 0
	collect_error = False
	min_tweets_per_call = 50

	log_dir = 'collect/{}/{}.log'.format(medium,username)
	log_msgs = []

	try:
		user_info = api.verify_credentials()
		num_statuses = user_info["statuses_count"]
		user_id = user_info["id"]
		log_msgs.append('START LOG FOR USER: {} | ID: {} | TOTAL POSTS: {}'.format(username, user_id, num_statuses))
	except Exception,error:
		util.update_user_status(conn, medium, username, error)
	
	try:
		alltweets = api.get_user_timeline(id=user_id,count=200)
		tweets = alltweets
		stolen = len(alltweets)

		if num_statuses > 0:
			min_id = alltweets[-1]["id"]

		while stolen < 3100 and stolen < num_statuses and (not collect_error):
			# for some accounts, the API starts collecting one tweet at a time after a certain period.
			# not sure why this happens - it happens well before rate limits kick in. 
			# warning_count tracks how many times we get a return well below 200 (anything less than min_tweets_per_call tweets back)
			# if that happens more than 5 times in a row, we stop collection and store what we've got
			#print 'len(tweets): {} | less than min?: {} | collect error: {}'.format(len(tweets), len(tweets) < min_tweets_per_call, collect_error)
			if len(tweets) < min_tweets_per_call:
				warning_count += 1
			else:
				warning_count = 0
			if warning_count > 5:
				collect_error = True

			calls_until_rate_limit = api.get_application_rate_limit_status(resources=['statuses'])['resources']['statuses']['/statuses/user_timeline']['remaining']
			
			print 'Num statuses collected for user: {}: {}'.format(username,stolen)
			print 'Total API calls left before rate limit: {}'.format(calls_until_rate_limit)
			log_msgs.append('Num statuses collected for user: {}: {}'.format(username,stolen))
			log_msgs.append('Total API calls left before rate limit: {}'.format(calls_until_rate_limit))
			
			if calls_until_rate_limit > sleep_cutoff:
				tweets = api.get_user_timeline(id=user_id,count=200,max_id=min_id)
				alltweets.extend(tweets)
				min_id = alltweets[-1]["id"]
				stolen = len(alltweets)
			else:
				print 'Close to API rate limit ({current} left) (this happened while working on username: {uname})...sleeping for 1 minute\n'.format(current=calls_until_rate_limit, uname=username)
				log_msgs.append('Close to API rate limit ({current} left) (this happened while working on username: {uname})...sleeping for 1 minute\n'.format(current=calls_until_rate_limit, uname=username))
				time.sleep(60)

		data = alltweets
		extract_meta(conn, medium, data, username, user_id, unique_id, table_data, log_msgs) # extract metadata, store in meta_tw
		util.update_user_status(conn, medium, username, "success") # update username as "collected"
		util.s3_cache(conn, medium, data, username) # cache raw blob in s3

		log_msgs.append('Collect for {} successful, collected {} tweets'.format(username, len(alltweets)))
		if collect_error:
			log_msgs.append('We hit a collect error, some kind of slowdown around Tweet ID: {}'.format(min_id))
		
		util.log(log_msgs,log_dir,full_path_included=True)
	except Exception,error:	
		try:
			util.update_user_status(conn, medium, username, error)
			log_msgs.append('There was an error with collect: [{}]'.format(error))
			util.log(log_msgs,log_dir,full_path_included=True)
			return "There was an error.  Check logs."
		except Exception,e:
			log_msgs.append(str(e)+str(error))
			util.log(log_msgs,log_dir,full_path_included=True)
			return str(e)+str(error)
