import pandas as pd
import numpy as np 
import re, datetime, os

from dateutil import parser
from os.path import join, dirname
from dotenv import load_dotenv
import pytz

from twython import Twython, TwythonError
from instagram.client import InstagramAPI

import util 

conn = util.connect_db()

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

data_head = os.environ.get("DATA_PATH")

''' Collects timezone information to generate local timestamps for postings '''

# Twitter uses Rails tz descriptors...this is the mapping to standardized TZInfo IDs
tz_mapping = { "International Date Line West" : "Pacific/Midway", "Midway Island" : "Pacific/Midway", "American Samoa" : "Pacific/Pago_Pago", "Hawaii" : "Pacific/Honolulu", "Alaska" : "America/Juneau", "Pacific Time (US & Canada)" : "America/Los_Angeles", "Tijuana" : "America/Tijuana", "Mountain Time (US & Canada)" : "America/Denver", "Arizona" : "America/Phoenix", "Chihuahua" : "America/Chihuahua", "Mazatlan" : "America/Mazatlan", "Central Time (US & Canada)" : "America/Chicago", "Saskatchewan" : "America/Regina", "Guadalajara" : "America/Mexico_City", "Mexico City" : "America/Mexico_City", "Monterrey" : "America/Monterrey", "Central America" : "America/Guatemala", "Eastern Time (US & Canada)" : "America/New_York", "Indiana (East)" : "America/Indiana/Indianapolis", "Bogota" : "America/Bogota", "Lima" : "America/Lima", "Quito" : "America/Lima", "Atlantic Time (Canada)" : "America/Halifax", "Caracas" : "America/Caracas", "La Paz" : "America/La_Paz", "Santiago" : "America/Santiago", "Newfoundland" : "America/St_Johns", "Brasilia" : "America/Sao_Paulo", "Buenos Aires" : "America/Argentina/Buenos_Aires", "Montevideo" : "America/Montevideo", "Georgetown" : "America/Guyana", "Greenland" : "America/Godthab", "Mid-Atlantic" : "Atlantic/South_Georgia", "Azores" : "Atlantic/Azores", "Cape Verde Is." : "Atlantic/Cape_Verde", "Dublin" : "Europe/Dublin", "Edinburgh" : "Europe/London", "Lisbon" : "Europe/Lisbon", "London" : "Europe/London", "Casablanca" : "Africa/Casablanca", "Monrovia" : "Africa/Monrovia", "UTC" : "Etc/UTC", "Belgrade" : "Europe/Belgrade", "Bratislava" : "Europe/Bratislava", "Budapest" : "Europe/Budapest", "Ljubljana" : "Europe/Ljubljana", "Prague" : "Europe/Prague", "Sarajevo" : "Europe/Sarajevo", "Skopje" : "Europe/Skopje", "Warsaw" : "Europe/Warsaw", "Zagreb" : "Europe/Zagreb", "Brussels" : "Europe/Brussels", "Copenhagen" : "Europe/Copenhagen", "Madrid" : "Europe/Madrid", "Paris" : "Europe/Paris", "Amsterdam" : "Europe/Amsterdam", "Berlin" : "Europe/Berlin", "Bern" : "Europe/Berlin", "Rome" : "Europe/Rome", "Stockholm" : "Europe/Stockholm", "Vienna" : "Europe/Vienna", "West Central Africa" : "Africa/Algiers", "Bucharest" : "Europe/Bucharest", "Cairo" : "Africa/Cairo", "Helsinki" : "Europe/Helsinki", "Kyiv" : "Europe/Kiev", "Riga" : "Europe/Riga", "Sofia" : "Europe/Sofia", "Tallinn" : "Europe/Tallinn", "Vilnius" : "Europe/Vilnius", "Athens" : "Europe/Athens", "Istanbul" : "Europe/Istanbul", "Minsk" : "Europe/Minsk", "Jerusalem" : "Asia/Jerusalem", "Harare" : "Africa/Harare", "Pretoria" : "Africa/Johannesburg", "Kaliningrad" : "Europe/Kaliningrad", "Moscow" : "Europe/Moscow", "St. Petersburg" : "Europe/Moscow", "Volgograd" : "Europe/Volgograd", "Samara" : "Europe/Samara", "Kuwait" : "Asia/Kuwait", "Riyadh" : "Asia/Riyadh", "Nairobi" : "Africa/Nairobi", "Baghdad" : "Asia/Baghdad", "Tehran" : "Asia/Tehran", "Abu Dhabi" : "Asia/Muscat", "Muscat" : "Asia/Muscat", "Baku" : "Asia/Baku", "Tbilisi" : "Asia/Tbilisi", "Yerevan" : "Asia/Yerevan", "Kabul" : "Asia/Kabul", "Ekaterinburg" : "Asia/Yekaterinburg", "Islamabad" : "Asia/Karachi", "Karachi" : "Asia/Karachi", "Tashkent" : "Asia/Tashkent", "Chennai" : "Asia/Kolkata", "Kolkata" : "Asia/Kolkata", "Mumbai" : "Asia/Kolkata", "New Delhi" : "Asia/Kolkata", "Kathmandu" : "Asia/Kathmandu", "Astana" : "Asia/Dhaka", "Dhaka" : "Asia/Dhaka", "Sri Jayawardenepura" : "Asia/Colombo", "Almaty" : "Asia/Almaty", "Novosibirsk" : "Asia/Novosibirsk", "Rangoon" : "Asia/Rangoon", "Bangkok" : "Asia/Bangkok", "Hanoi" : "Asia/Bangkok", "Jakarta" : "Asia/Jakarta", "Krasnoyarsk" : "Asia/Krasnoyarsk", "Beijing" : "Asia/Shanghai", "Chongqing" : "Asia/Chongqing", "Hong Kong" : "Asia/Hong_Kong", "Urumqi" : "Asia/Urumqi", "Kuala Lumpur" : "Asia/Kuala_Lumpur", "Singapore" : "Asia/Singapore", "Taipei" : "Asia/Taipei", "Perth" : "Australia/Perth", "Irkutsk" : "Asia/Irkutsk", "Ulaanbaatar" : "Asia/Ulaanbaatar", "Seoul" : "Asia/Seoul", "Osaka" : "Asia/Tokyo", "Sapporo" : "Asia/Tokyo", "Tokyo" : "Asia/Tokyo", "Yakutsk" : "Asia/Yakutsk", "Darwin" : "Australia/Darwin", "Adelaide" : "Australia/Adelaide", "Canberra" : "Australia/Melbourne", "Melbourne" : "Australia/Melbourne", "Sydney" : "Australia/Sydney", "Brisbane" : "Australia/Brisbane", "Hobart" : "Australia/Hobart", "Vladivostok" : "Asia/Vladivostok", "Guam" : "Pacific/Guam", "Port Moresby" : "Pacific/Port_Moresby", "Magadan" : "Asia/Magadan", "Srednekolymsk" : "Asia/Srednekolymsk", "Solomon Is." : "Pacific/Guadalcanal", "New Caledonia" : "Pacific/Noumea", "Fiji" : "Pacific/Fiji", "Kamchatka" : "Asia/Kamchatka", "Marshall Is." : "Pacific/Majuro", "Auckland" : "Pacific/Auckland", "Wellington" : "Pacific/Auckland", "Nuku'alofa" : "Pacific/Tongatapu", "Tokelau Is." : "Pacific/Fakaofo", "Chatham Is." : "Pacific/Chatham", "Samoa" : "Pacific/Apia" }


def tz_collect(conn, test=False, max_collect=1000):
	''' Collects time zone of social media accounts '''

	# gets table name / field / datatype for all tables as a Pandas data frame
	table_data = util.get_table_data()

	log_msgs = []
	log_msgs.append('Starting collect\n')

	user_tz = []
	try:
		query = "SELECT username, user_id, uid, medium FROM usernames WHERE medium='twitter' LIMIT {}".format(max_collect)
		cur   = conn.cursor()
		cur.execute(query)
		rows  = cur.fetchall()
		ct = 0
		for row in rows:

			username, user_id, unique_id, medium = row
			#log_msgs.append('Collect time zone for {} user: {} [ID: {}]'.format(medium, username, user_id))

			CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_SECRET = util.get_tokens(conn, medium, username)

			if medium == "twitter": # big thanks to Andy Reagan here: https://github.com/andyreagan/tweet-stealing
				
				try:
					twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_SECRET)
					user_info = twitter.verify_credentials()
					tz = user_info['time_zone']
					if tz is not None:
						print username,'::',tz_mapping[tz]
						ct += 1
						user_tz.append( (tz_mapping[tz],username) )
					else:
						print username, '::', 'No time zone info'
					#tz_collect_twitter(twitter, username, unique_id, conn, table_data)
				except Exception,e:
					print username, ":: ERROR:", str(e)

		#log_msgs.append('Collect log completed without top-level errors (check individual logs for per-basis errors).')

	except Exception, error:
		log_msgs.append('Collect error: {}'.format(str(error)))

	log_dir = 'collect/batch/'

	for msg in log_msgs:
		print msg
	print '{} out of {} users had time zone info'.format(ct, len(rows))
	#util.log(log_msgs,log_dir)

	try:
		q = 'update meta_tw set tz=? where username=?'
		with conn:
			cur = conn.cursor()
			cur.executemany(q, user_tz)
			conn.commit()
	except Exception,e:
		print 'sqlite3 error:', str(e)

	return user_tz
