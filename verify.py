def verify_twitter(api,username,followed):

	verified = False
	uid = api.get_user(username).id
	follow_ids = api.followers_ids(followed)

	if uid in follow_ids:
		verified = True

	return verified

def verify_instagram(api,username):

	try:
		verified  = False
		followers = []
		follower_data, next_ = api.user_followed_by()

		for f in follower_data:
			followers.append(str(f).split(" ")[1])
		while next_: # pagination 
		    more_follows, next_ = api.user_followed_by(with_next_url=next_)
		    for f in more_follows:
				followers.append(str(f).split(" ")[1])
		    followers.extend(more_follows)

		if username in followers:
			verified = True

		return verified

	except Exception, e:
		return str(e)