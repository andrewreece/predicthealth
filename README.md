**Data Collection**  
<tt>run.py</tt> handles all of the RESTful/outward-facing requests (eg. /verify/\<username\>).
<br /><br />
<tt>collect.py</tt> and <tt>verify.py</tt> have medium-specific methods for verifying that a participant has followed us, and collecting their data.
<ul>
<li><tt>/collect</tt> is run either manually or as a cron job every 15 minutes (when turned on)</li>
<li>it looks for any usernames with collected=0 in the <tt>usernames</tt> table</li>
</ul>
<tt>extract.py</tt> feeds subroutines to the <tt>/collect</tt> regime for feature extraction.
<br /><br />
<tt>util.py</tt> has various database and other helper functions.
  
**Analysis**
<ul>
<li><tt>eda-instagram</tt> and <tt>eda-twitter</tt> are the analytical frames except for Bayesian stuff</li>
<li><tt>*-bayes.R</tt> for Bayesian logistic regression and convergence checks</li>
<li><tt>bgfunc</tt> contains all the heavy lifting for analysis and processing</li>
<li><tt>face-detection</tt> compiles, trains, and verifies face detection algo</li>
<li><tt>mixture-models</tt> has code for Kalman filter, GMM, and HMM</li>
</ul>