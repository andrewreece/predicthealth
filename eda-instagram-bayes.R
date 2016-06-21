#
# 21 June 2016
# Dissertation: Bayesian analysis (instagram)
# Author: Andrew Reece
#

setwd('~/Google Drive/_dissertation//backups/db_backup/')

require(stringr)
source('bgfunc.R')

# mcmc model software params
use.jags <- FALSE
use.mcmc.pack <- TRUE
load.jags <- FALSE
load.coda <- FALSE
load.mcmc.pack <- FALSE
if (use.jags) require(rjags)
if (use.mcmc.pack) require(MCMCpack)

# mcmc params
n <- 100000
thin <- 10
burn.in <- 5000
dic.samp <- 1000
b0 <- 0
B0 <- 0.0001

# analysis params
analyze.addtl.data <- TRUE
standardize.predictors <- TRUE
posting.cutoff <- TRUE
run.diagnostics <- TRUE
  show.autocorr <- TRUE
  show.gelman <- TRUE
  show.geweke <- TRUE
  show.bayes.p <- TRUE

dic.comparison <- TRUE
model.dic <- list()

# data params
varset <- readRDS('varset.rds')
condition <- 'depression' # depression, pregnancy, cancer, ptsd
medium <- 'ig' # ig, tw
gb.types <- c('post','created_date','username') 
gb.type <- 'created_date'
kind <- 'MAIN'
addtl <- ifelse (analyze.addtl.data, 'addtl', 'no_addtl')
stdized <- ifelse (standardize.predictors, 'standardized', 'nonstandardized')
post.cut <- ifelse (posting.cutoff, 'post_cut', 'post_uncut')
model.complexities <- c('intercept_only', 'means') 
if (analyze.addtl.data) model.complexities <- c(model.complexities,'no_addtl_means')

# generate file path to data csv (created in python)
fpath <- get.data.fpath(condition,medium,gb.type,kind,addtl,post.cut)
df <- read.csv(fpath, header=T, stringsAsFactors=F)

print(paste('Running analysis for groupby type:',gb.type,':: Timeline:',kind))

for (m in model.complexities) {
  if (m == 'intercept_only') {
    var.list <- build.var.list(NULL, NULL, df, standardize.predictors, intercept.only=TRUE)
    output <- write.jags.model(mdf, NULL, NULL, intercept.only=TRUE)
    model.jags <- output[['model']]
    var.names <- output[['var.names']]
    if (use.mcmc.pack) {
      mdf <- data.frame(b.0=rep(1,nrow(df)))
    }
  } else {
    means <- m
    mdata <- set.model.data(medium, gb.type, means, varset, df)
    preds <- mdata[['preds']]
    mdf <- mdata[['mdf']]
    var.list <- build.var.list(preds, mdf, df, standardize.predictors)
    output <- write.jags.model(mdf, preds, var.list) 
    model.jags <- output[['model']]
    var.names <- output[['var.names']]
  }
  
  if (use.jags) {
   jags.full.path <- get.jags.path('jags','jags',medium,kind,gb.type,addtl,stdized,m)
   coda.full.path <- get.jags.path('jags','coda_samples',medium,kind,gb.type,addtl,stdized,m)
    
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
    pack.full.path <- get.jags.path('pack','mcmcpack',medium,kind,gb.type,addtl,stdized,m)
    
    if (load.mcmc.pack) {
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
  summary(mcmc)
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
      gelman.diag(mcmc) # should be around 1.0
      gelman.plot(mcmc) # red line should converge to black line
    }
    if (show.geweke && m!='intercept_only') {
      geweke.diag(mcmc)       ## z-scores; look good [-2, 2]
      geweke.plot(mcmc)
    }
    if (show.bayes.p) bayes.p <- get.bayes.p(var.list, mcmc, print.stats=TRUE)
      
  }

  ## Model comparison
  # DIC (D=deviance) model comparison statistic, lower is better
  if (dic.comparison && use.jags) model.dic[[m]] <- dic.samples(jags, dic.samp) 
}

if (dic.comparison && use.jags) compare.dic(model.dic, analyze.addtl.data)

