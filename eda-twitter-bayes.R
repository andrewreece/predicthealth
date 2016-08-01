#
# 22 June 2016
# Dissertation: Bayesian analysis (instagram)
# Author: Andrew Reece
#

setwd('~/Google Drive/_dissertation/backups/db_backup/')

require(stringr)
require(arm)
source('twbgfunc.R')

# mcmc model software params
use.jags <- FALSE
use.mcmc.pack <- TRUE
load.jags <- FALSE
load.coda <- FALSE
load.mcmc.pack <- FALSE
if (use.jags) require(rjags)
if (use.mcmc.pack) require(MCMCpack)

# mcmc params
n <- 200000
thin <- 10
burn.in <- 10000
b0 <- 0
B0 <- 0.0001

# analysis params
use.pca <- FALSE
total.pca.comp <- 40 # how many pca components are there?
pca.optimize <- FALSE
pca.frames <- c(2,3,39) # these values + 1 = num
opt.pca.comp <- NULL # after comparing different pca comp sets, what's best?
standardize.predictors <- TRUE
only.created.date <- TRUE
run.diagnostics <- TRUE
show.autocorr <- TRUE
show.gelman <- TRUE
show.geweke <- TRUE
show.bayes.p <- TRUE

bayes.factor <- TRUE
b.factor <- list()

# data params
if (use.pca) {
  varset <- list()
  if (pca.optimize) {
    for (i in pca.frames) varset[[paste0('pca_',i)]] <- paste('pca', seq(0,i-1), sep='_')
  } else {
    varset[[paste0('pca_',opt.pca.comp)]] <- paste('pca', seq(opt.pca.comp), sep='_')
  }
} else {
  varset <- readRDS('varset.rds')
}

condition <- 'depression' # depression, pregnancy, cancer, ptsd
medium <- 'tw' # ig, tw
kind <- 'MAIN' # MAIN, before-from-diag, before-from-susp
used.pca <- ifelse (use.pca, 'pca', 'no-pca')

model.complexities <- c('intercept_only')

if (pca.optimize) {
  for (name in names(varset)) model.complexities <- c(model.complexities, name)
} else {
  model.complexities <- c(model.complexities, 'basic_tweet_means', 'all_means')
}

if (only.created.date) {
  gb.types <- c('created_date')
} else {
  gb.types <- c('created_date','weekly','user_id')
}

# adjustments
#model.complexities <- c('ig_face_means') 
gb.type <- 'weekly'
m <- 'all_means'
model.complexities <- c('basic_tweet_means')


for (gb.type in gb.types) {
  
  # generate file path to data csv (created in python)
  fpath <- get.data.fpath(condition,medium,gb.type,kind,use.pca,total.pca.comp)
  df <- read.csv(fpath, header=T, stringsAsFactors=F)
  
  for (m in model.complexities) {
    print(paste('Running',medium,'analysis for Timeline:',kind,':: groupby type:',gb.type,':: Model:', m))
    
    if (m == 'intercept_only') {
      
      var.list <- build.var.list(NULL, NULL, df, standardize.predictors, intercept.only=TRUE)
      output <- write.jags.model(mdf, NULL, NULL, intercept.only=TRUE)
      model.jags <- output[['model']]
      var.names <- output[['var.names']]
      if (use.mcmc.pack) mdf <- data.frame(b.0=rep(1,nrow(df)))
      
    } else {
      
      means <- m
      mdata <- set.model.data(medium, gb.type, means, varset, df, m, use.pca)
      preds <- mdata[['preds']]
      mdf <- mdata[['mdf']]
      var.list <- build.var.list(preds, mdf, df, standardize.predictors)
      output <- write.jags.model(mdf, preds, var.list) 
      model.jags <- output[['model']]
      var.names <- output[['var.names']]
    }
    
    if (use.jags) {
      jags.full.path <- get.jags.path('jags','jags',condition,medium,kind,gb.type,m)
      coda.full.path <- get.jags.path('jags','coda_samples',condition,medium,kind,gb.type,m)
      
      if (load.jags) {
        load(jags.full.path)
        jags$recompile()
      } else { # otherwise, run the mcmc simulation
        #runjags1 <- run.jags(model.jags, n.chains=2)
        jags <- jags.model(textConnection(model.jags), data = var.list, n.chains = 2)
        update(jags, burn.in)                ## burn-in
        save(file=jags.full.path, list="jags")
      }
      
      if (load.coda) {
        load(coda.full.path)
      } else {
        # coda samples save to file inside get.coda.params()
        mcmc <- get.coda.params(jags, var.names, thin, n.iter, coda.full.path)
      }
    } else if (use.mcmc.pack) {
      pack.full.path <- get.jags.path('pack','mcmcpack',condition,medium,kind,gb.type,m)
      
      if (load.mcmc.pack && file.exists(pack.full.path)) {
        load(pack.full.path)
      } else {
        mdf['target'] <- df$target
        mcmc <- mcmc.pack.model(mdf, burn.in, n, thin, b0, B0, m)
        save(file=pack.full.path, list="mcmc")
      }
    }
    
    if (use.mcmc.pack) {
      var.names <- pack.var.names(var.names)
    }
    
    # stats on samples from the mcmc posterior
    if (m != 'intercept_only') print(summary(mcmc)[[1]][,c('Mean','SD')])
    report.hpdi(mcmc, var.names) # flags HPDIs containing zero, plots hist
    
    ## MCMC diagnositcs
    if (run.diagnostics) {
      # trace and posterior density
      for (field in var.names) { # plotting all at once is too big
        plot(mcmc[[1]][,field], main=field)
      }
      # autocorrelation
      if (show.autocorr) autocorr.plot(mcmc)
      # Gelman diagnostics
      if (show.gelman && m!='intercept_only') {
        try(print(gelman.diag(mcmc)))
        #try(gelman.plot(mcmc))
      }
      if (show.geweke && m!='intercept_only') {
        try(print(geweke.diag(mcmc)))
        #try(geweke.plot(mcmc))
        
      }
      if (show.bayes.p) bayes.p <- get.bayes.p(var.list, mcmc, print.stats=TRUE)
      
    }
    
    ## Model comparison
    if (bayes.factor && use.mcmc.pack) b.factor[[m]] <- mcmc
  }
  
  if (bayes.factor && use.mcmc.pack) b.factor.output <- compare.bayes.factor(b.factor, use.pca, pca.frames)
  
} # end gb.type loop

