get.data.fpath <- function(condition,medium,gb.type,kind,addtl,post.cut) {
  dir.path <- paste('data-files',condition,medium,gb.type,sep='/')
  fname <- paste(condition,medium,gb.type,kind,addtl,'data',post.cut,sep='_')
  fpath <- paste0(dir.path,'/',fname,'.csv')
  return(fpath)
}

get.jags.path <- function(ftype,which.f,condition,medium,kind,gb.type,addtl,stdized,m,post.cut) {
  folder.path <- paste(paste0(ftype,"-mcmc-output"),condition,kind,gb.type,sep="/")
  fname <- paste(which.f,condition,medium,kind,gb.type,addtl,stdized,m,post.cut,".Rdata",sep="_")
  full.path <- paste(folder.path,fname,sep="/")
  return(full.path)
}


set.model.data <- function(medium, gb.type, means, varset, df){
  
  # which predictors do we use for this medium/groupby type?
  preds <- varset[[medium]][[gb.type]][[means]]
  # mdf is the subset of df we use for modeling
  mdf <- df[preds]
  
  return(list(preds=preds,mdf=mdf))
}

build.var.list <- function(preds, mdf, df, stdize, intercept.only=FALSE) {
  var.list <- list(n = nrow(df), y = df$target)
  
  if (!intercept.only) {
    for (field in preds) {
      if (stdize) {
        # we need as.vector because scale() returns extra attrs
        var.list[[field]] <- as.vector(scale(mdf[[field]]))
      } else {
        var.list[[field]] <- mdf[[field]]
      }
    }
  }
  return(var.list)
}


write.jags.model <- function(mdf, preds, var.list, intercept.only=FALSE, include.or=FALSE) {
  
  # this first part is common to all log.reg. models we use
  model.jags <- "model {
  
  # likelihood (Bernoulli)
  for (i in 1:n){
  y[i] ~ dbern(p[i])
  p[i] <- 1 / (1 + exp(-Xb[i]))"
  
  Xb <- "Xb[i] <- b.0"
  bs <- "b.0 ~ dnorm (0, 0.0001)"
  ors <- ""
  var.names <- c('b.0')
  if (!intercept.only) {
    
    for (field in preds) {
      Xb <- paste(Xb, paste0(' + ','b.',field,'*',field,'[i]'))
      # add prior
      bs <- paste(bs, (paste0('b.',field,' ~ ','dnorm (0, 0.0001)')), sep='\n')
      # add odds ratio
      if (include.or) {
        ors <- paste(ors, (paste0('or.',field,' <- ','exp(b.',field,')')), sep='\n')
        var.names <- append(var.names, paste0('or.',field))
      }
      var.names <- append(var.names, paste0('b.',field))
    }
  }
  
  #Xb <- substr(Xb,1,nchar(Xb)-2) # this is only for when ' + ' is on the end, but you switched it to the front
  model.jags <- paste(model.jags, Xb, '}', sep='\n')
  model.jags <- paste(model.jags, bs, sep='\n')
  if (include.or) {
    model.jags <- paste(model.jags,ors,sep='\n')
  }
  if (!intercept.only) {
    addtl.priors <- "sigma.state ~ dunif(0, 100)
    tau.state <- pow(sigma.state, -2)"
    model.jags <- paste(model.jags,addtl.priors,sep='\n')
  }
  model.jags <- paste(model.jags,'}',sep='\n')
  
  return(list(model=model.jags, var.names=var.names))
}


hpd.hunt <- function(params, field, old.prob) {
  prob <- ifelse(old.prob==0.99, 0.95, old.prob-0.05)
  hpd.obj <- HPDinterval(params, prob=prob)[[1]]

  if (hpd.obj[field,'lower'] * hpd.obj[field,'upper'] > 0) {
    print(paste0(prob,'% HPDI:'))
    print(hpd.obj[field,])
    print(paste(str_to_upper(field),':: GOOD AT prob:',prob,'% :: Does not contain zero'))
    print('____')
  } else {
    # if HPDI contains zero, we plot the histogram to see how close it is
    #print(paste(str_to_upper(field),':: NOT GOOD AT prob:',prob,'% :: Contains zero, see plot'))
    hpd.hunt(params, field, prob)
  }  
}


report.hpdi <- function(params, var.names, prob=0.99) {
  print('Highest Posterior Density Intervals:')
  not.goods <- c() # for keeping track of which HPDIs include 0
  
  for (field in var.names) {
    print(field)
    hpd.obj <- HPDinterval(params, prob=prob)[[1]]
    print('99% HPDI:')
    print(hpd.obj[field,])
    if (hpd.obj[field,'lower'] * hpd.obj[field,'upper'] > 0) {
      print(paste(str_to_upper(field),':: GOOD AT prob:',prob,'% :: Does not contain zero'))
      print('____')
    } else {
      # if HPDI contains zero, we plot the histogram to see how close it is
      print(paste(str_to_upper(field),':: NOT GOOD AT prob:',prob,'% :: Contains zero, see plot'))
      field.post <- params[[1]][,field]           ## extract happy posterior
      hist(field.post, main = paste("Histogram",field, "Posterior (jags)"), xlab = paste(field,"parameter samples (jags)"))
      abline(v = hpd.obj[field,], col = "gray", lty = 2, lwd = 2)  ## HPD
      abline(v = mean(field.post), col = "red", lwd = 2)         ## posterior mean
      hpd.hunt(params, field, 0.99)
    }      
  }
}

pack.var.names <- function(var.names) {
  for (i in 1:length(var.names)) {
    if (var.names[i] != "b.0") {
      var.names[i] <- substr(var.names[i], 3, nchar(var.names[i]))
    } else {
      var.names[i] <- "(Intercept)"
    }
  }
  return(var.names)
}


get.coda.params <- function(jags, var.names, thin, n.iter, coda.fname, save.file=TRUE) {
  
  params <- coda.samples(jags, variable.names = var.names, 
                         thin = thin, n.iter = n.iter)
  save(file=coda.fname, list="params")
  
  return(params)
}

get.bayes.p <- function(var.list, params, 
                        print.stats=FALSE, n.replica=1000) {
  
  # this makes a df of the standardized predictors, y, and n
  dm.df <- data.frame(var.list)
  # we don't need n
  dm.df$n <- NULL
  target <- dm.df$y
  # this gets our intercept in the design matrix
  design.mat <-  model.matrix(glm(y ~ ., data = dm.df, family=binomial))
  
  samp.ix.min <- nrow(as.matrix(params))-10000
  samp.ix.max <- nrow(as.matrix(params))
  
  set.seed(123)
  ixs <- sample(samp.ix.min:samp.ix.max, n.replica) # index for 1000 random beta vectors 
  betamat <- t(as.matrix(params)[ixs,]) # we transpose it since we need the parameters in the rows
  
  ypred <- invlogit(design.mat %*% betamat)     ## compute expected (or predicted) values
  ypred[ypred < 0.5] <- 0
  ypred[ypred >= 0.5] <- 1
  
  replicated.prop <- colSums(ypred)/nrow(ypred)
  obs.prop <- sum(target)/nrow(design.mat)
  prop.diff <- replicated.prop - obs.prop
  
  if (print.stats) {
    pred.df <- data.frame(pred=ypred,actual=dm.df$y)
    print(paste('observed proportion of 1s:',obs.prop))
    print(paste('prop.diff mean:',round(mean(prop.diff),3)))
    hist(replicated.prop, main='Proportion target=y in replicated samples')
    hist(prop.diff, main='Proportion target=y difference: replicated-observed')
  }
  
  p.value <- round(length(which(prop.diff > 0))/length(prop.diff), 3)  ## compute p-value
  
  print(paste('Bayes p-value:',p.value))
  return(p.value)
}

compare.dic <- function(model.dic, analyze.addtl.data) {
  x <- 'intercept_only'
  y <- 'all_means'
  z <- 'hsv_means'
  if (!analyze.addtl.data) {
    models <- c(x,y)
    ddic <- diffdic(model.dic[[x]], model.dic[[y]])
    print(paste('Model',ifelse((ddic < 0),x,y),'is best'))
  } else {
    models <- c(x,y,z)
    
    ddic <- diffdic(model.dic[[x]], model.dic[[y]])
    print(paste('Model',x,'vs',y))
    print(ddic)
    
    ddic <- diffdic(model.dic[[x]], model.dic[[z]])

    print(paste('Model',x,'vs',z))
    print(ddic)  

    ddic <- diffdic(model.dic[[y]], model.dic[[z]])

    print(paste('Model',y,'vs',z))
    print(ddic)  
  }
  for (model in models) {
    print(paste(model,'DIC:'))
    print(model.dic[[model]])
  }
}

mcmc.pack.model <- function(data, burnin, n, thin, b0, B0, m) {

  if (m == 'intercept_only') {
    # first chain
    mcmc.1 <- MCMClogit(target ~ 1, data = data,
                        burnin = burnin, mcmc = n, 
                        thin = thin, b0 = b0, B0 = B0, seed = 1234, marginal.likelihood = "Laplace")
    # second chain
    mcmc.2 <- MCMClogit(target ~ 1, data = data,
                        burnin = burnin, mcmc = n, 
                        thin = thin, b0 = b0, B0 = B0, seed = 5678, marginal.likelihood = "Laplace")
  } else {
    # first chain
    mcmc.1 <- MCMClogit(target ~ ., data = data,
                        burnin = burnin, mcmc = n, 
                        thin = thin, b0 = b0, B0 = B0, seed = 1234, marginal.likelihood = "Laplace")
    # second chain
    mcmc.2 <- MCMClogit(target ~ ., data = data,
                        burnin = burnin, mcmc = n, 
                        thin = thin, b0 = b0, B0 = B0, seed = 5678, marginal.likelihood = "Laplace")
  }
  
  mcmc <- mcmc.list(mcmc.1, mcmc.2)  ## merging both models into a mcmc object
  return(mcmc)
}

compare.bayes.factor <- function(fits, addtl, hsv.separate) {
  # computes bayes factors for each model and compares them 
  
  if (addtl) {
    bfactor <- BayesFactor(fits[['intercept_only']][[1]], fits[['ig_face_means']][[1]], 
                           fits[['ratings_means']][[1]])
    col.names <- c('intercept_only','metadata','ratings')
  } else {
    if (hsv.separate) {
      
      bfactor <- BayesFactor(fits[['intercept_only']][[1]], fits[['hsv_means']][[1]], 
                             fits[['all_ig_means']][[1]], fits[['ig_face_means']][[1]])
      col.names <- c('intercept_only','hsv','all_instagram','insta_plus_face')
    } else {
      bfactor <- BayesFactor(fits[['intercept_only']][[1]], fits[['ig_face_means']][[1]])
      col.names <- c('intercept_only','all_instagram')
    }
  }
  bf.mat <- bfactor$BF.log.mat
  colnames(bf.mat) <- col.names
  rownames(bf.mat) <- col.names
  print('Bayes Factor model matrix:')
  print(round(bf.mat, 1))
  
  return(bfactor)
}