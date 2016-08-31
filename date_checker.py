import numpy as np
import pandas as pd
import os, re, time, datetime, sqlite3
from dateutil import parser

from functools import wraps

from nocache import nocache

import util, datetime
from dateutil import parser

conn = util.connect_db()
conditions = ['pregnancy','cancer','ptsd','depression']

# this code fixes existing database entries from qualtrics where the month is a string instead of a zero-padded integer
months = [('January','01'),('February','02'),('March','03'),('April','04'),('May','05'),('June','06'),('July','07'),('August','08'),('September','09'),('October','10'),('November','11'),('December','12')]
month_dict = {k:v for k,v in months}
conditions = ['pregnancy','cancer','ptsd','depression']
for condition in conditions:
	query = 'select uid, diag_month from {} where diag_month is not null and diag_monthnum is null'.format(condition)
	with conn:
		cur = conn.cursor()
		cur.execute(query)
		rows = cur.fetchall()
	params = []
	for row in rows:
		uid = row[0] 
		dmonth = row[1] 
		params.append([dmonth, month_dict[dmonth], uid])
	query = "update {} set diag_monthnum = replace(diag_month,?,?) where uid=? and diag_month is not null".format(condition)
	with conn:
		cur = conn.cursor()
		cur.executemany(query, params)
	conn.commit()


for condition in conditions:

	query = ("SELECT platform, {cond}.username, diag_day, diag_monthnum, diag_year, days_suspected ".format(cond=condition) + 
			 "FROM {} ".format(condition) + 
			 "WHERE ({cond}.username_tw is not null OR {cond}.username_ig is not null) AND ({cond}.diag_year is not null)".format(cond=condition)
			 )

	with conn:
		cur = conn.cursor()
		cur.execute(query)
		rows = cur.fetchall()
	print 
	print
	print "CONDITION: {}".format(condition)
	params = []
	for r in rows:
		platform, utw, uig, day, month, year, dsusp = r
		month = '0'+str(month) if len(str(month))==1 else month
		day = '0'+str(day) if len(str(day))==1 else day 
		dates = {}
		ddate = '-'.join([str(year),str(month),str(day)])
		dates['diag'] = ddate
		try:
			ddate_obj = parser.parse(ddate)
		except Exception, e:
			print str(e)
			print 'Problem with date for user: tw: {} ig: {}, date = {}'.format(utw, uig,ddate) 
		if dsusp and str(dsusp)[-1]!='+':
			sdate_obj = ddate_obj - datetime.timedelta(days=dsusp)
			dates['susp'] = sdate_obj.strftime('%Y-%m-%d')
		else:
			dates['susp'] = None
		if utw:
			table_name = 'meta_tw'
			field_name = 'username_tw'
			uname = utw
		elif uig:
			table_name = 'meta_ig'
			field_name = 'username_ig'
			uname = uig
		else:
			table_name = None
		if table_name:
			for date_type in ['diag','susp']:
				if dates[date_type] is not None:

					# count number of days difference between post and diag/susp date
					# -int = post date is earlier than diag/susp date, +int=later
					query = "UPDATE {table} SET d_from_{date_type}_{cond}=julianday(created_date)-julianday('{date}') WHERE username='{uname}'".format(table=table_name,cond=condition,date=dates[date_type],date_type=date_type,uname=uname)
					
					with conn:
						cur = conn.cursor()
						cur.execute(query)
						#rows = cur.fetchall()
					conn.commit()
					print 'Commit complete for {} [TABLE: {}, COND: {}, DTYPE: {}]'.format(uname, table_name, condition, date_type)
					#conn.commit()
				#query = "update {cond} set usable_data_points_{dtype} = (select count(*) from {tname} where {tname}.within_90_{dtype}_{cond}=1 and {tname}.username={cond}.{fname}) where exists (select * from {tname} where {tname}.username={cond}.{fname})".format(cond=condition,dtype=date_type,tname=table_name,fname=field_name)
				#with conn:
				#	cur = conn.cursor()
				#	cur.execute(query)
				#conn.commit()

conditions = ['pregnancy','cancer','ptsd','depression']
for condition in conditions:
	for date_type in ['diag','susp']:
		for med in ['tw','ig']:

			query = "update {cond} set usable_data_points_{dtype} = (select count(*) from meta_{med} where abs(meta_{med}.d_from_{dtype}_{cond})<=90 and meta_{med}.username={cond}.username_{med}) where exists (select * from meta_{med} where meta_{med}.username={cond}.username_{med} limit 1)".format(cond=condition,dtype=date_type,med=med)

			with conn:
				cur = conn.cursor()
				cur.execute(query)
			conn.commit()

			print 'Finished {med} {c} {dt}'.format(med=med, c=condition, dt=date_type)


