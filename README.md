<tt>run.py</tt> handles all of the RESTful/outward-facing requests (eg. /verify/\<username\>).
<br /><br />
<tt>collect.py</tt> and <tt>verify.py</tt> have medium-specific methods for verifying that a participant has followed us, and collecting their data.
<ul>
<li><tt>/collect</tt> is run as a cron job every 15 minutes (when turned on)</li>
<li>it looks for any usernames with collected=0 in the <tt>usernames</tt> table</li>
<li>cron job settings are set through Andrew's Dreamhost account.</li>
</ul>
<tt>extract.py</tt> feeds subroutines to the <tt>/collect</tt> regime for feature extraction.
<br /><br />
<tt>util.py</tt> has various database and other helper functions.