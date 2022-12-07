options(warn=-1)
library(mvtnorm) # generates multivariate Gaussian sampels and calculate the densities
library(ggplot2) # plotting
library(reshape2) # data wrangling!
library(clusterGeneration) # generates the covariance matrices that we need for producing synthetic data.

# Synthetic Data Generation
# Set the parameters:
set.seed(12345) # save the random seed to make the results reproducble
N <- 1000 # number of samples
K <- 3    # number of clusters
D <- 2    # number of dimensions

# Initializations:
Phi <- runif(K); Phi <- Phi/sum(Phi)    # Phi(k) indicates the fraction of samples that are from cluster k
Nk <- matrix(0,nrow = K)    # initiate  the effective number of points assigned to each cluster
Mu <- matrix(runif(K*D, min=-1)*10,nrow = K, ncol = D)    # initiate the centriods (means) of the clusters (randomly chosen)
Sigma <- matrix(0,nrow = K, ncol = D^2)    # initiate the covariance matrix

# Create the covariance matrices:
for (k in 1:K){
    # For each cluster generate one sigma matrix
    Sigma[k,] <- genPositiveDefMat(D)$Sigma[1:D^2]
}

# Generate data:
data <- data.frame(K=integer(), X1=double(), X2=double()) # empty dataset
data[1:N,'K'] <- sample(1:K, N, replace = TRUE, prob = Phi) # geenrate labels (they will not be used in EM, just for validation)
## For each cluster k:
for (k in 1:K){
    ### calculate the effective number of points assigned to it:
    Nk[k] <- sum(data$K==k)
    ### generate the actual points:
    data[data$K==k, 2:3] <- rmvnorm(n = Nk[k], Mu[k,], matrix(Sigma[k,], ncol=D))
}

# Remove the lables! So, our GMM has no clue what are the real labels.
X <- as.matrix(data[,-1]) 

# Visualize the data (with the real labels)
ggplot(data=data, aes(x=X1, y=X2, color=factor(K))) + geom_point() +
    scale_color_discrete(guide = guide_legend(title = "Cluster")) + 
    ggtitle ('Dataset') + theme_minimal()
	
## We generated our data previously, so we know the real cluster labels. 
## However, we pretend we do not have the real lables. 
## Let's implement Soft Expectation Maximization for our GMM. Again we use .hat name convention to differentiate the real and estimated values


##====================================================================================================================================================
# Soft EM for GMM
# Setting the parameters:
eta.max <- 100      # maximum number of iteratins
epsilon <- 0.01     # termination threshold 

# Initialzations:
eta <- 1            # epoch counter
terminate <- FALSE  # termination condition

## Ramdom cluster initialization:
set.seed(123456) # save the random seed to make the results reproducble
Phi.hat <- 1/K                          # assume all clusters have the same size (we will update this later on)
Nk.hat <- matrix(N/K,nrow = K)          # refer to the above line!
Mu.hat <- as.matrix(X[sample(1:N, K), ]) # randomly  choose K samples as cluster means (any better idea?)
Sigma.hat <- matrix(,nrow = K, ncol = D^2) # create empty covariance matrices (we will fill them)
post <- matrix(,nrow=N, ncol=K)        # empty posterior matrix (the membership estimates will be stored here)

### for each cluster k:
for (k in 1:K){
    #### initiate the k covariance matrix as an identity matrix (we will update it later on)
    Sigma.hat[k,] <- diag(D) # initialize with identity covariance matrix
}

# Build the GMM model
Mu.hat.old <- Mu.hat # store the old estimated means
while (!terminate){
    
    # E step:    
    for (k in 1:K){
        ## calculate the posterior based on the estimated means,covariance and cluster size:
        post[,k] <- dmvnorm(X, Mu.hat[k,],  matrix(Sigma.hat[k,], ncol=D)) * Nk.hat[k]
    }
    post <- post/rowSums(post) # normalization (to make sure post(k) is in [0,1] and sum(post)=1)

    # M step:
    for (k in 1:K){
        ## recalculate the estimations:
        Nk.hat[k] <- sum(post[,k])        # the effective number of point in cluster k
        Phi.hat[k] <- sum(post[,k])/N     # the relative cluster size
        Mu.hat[k,] <- colSums(post[,k] *X)/Nk.hat[k] # new means (cluster cenroids)
        Sigma.hat[k,] <- (t(X-matrix(Mu.hat[k,],nrow = N, ncol=D, byrow = TRUE))%*%
                          (post[,k]*(X-matrix(Mu.hat[k,],nrow = N, ncol=D, byrow = TRUE))))/Nk.hat[k] # new covariance
    
    }

    if (eta %% 10 ==1) {
        print(ggplot(data=as.data.frame(X), aes(x=X1, y=X2)) + 
        geom_point(color=rgb(post), alpha=0.75) +
        ggtitle (paste('Soft EM Results (eta=', eta, ')')) + theme_minimal())
        }
    
    # increase the epoch counter
    eta <- eta+1
    
    # check the termination criteria
    terminate <- eta > eta.max | sum(abs(Mu.hat.old - Mu.hat)) <= epsilon
    
    # record the means (neccessary for checking the termination criteria)
    Mu.hat.old <- Mu.hat

}
# That's it! Let see how many iterations we had:
cat('maximum number of itterations:',eta,'\n')

##====================================================================================================================================================
##Hard EM for GMM
# Setting the parameters:
eta.max <- 100      # maximum number of iteratins
epsilon <- 0.01  # termination threshold 

# Initialzations:
eta <- 1            # epoch counter
terminate <- FALSE  # termination condition

## Ramdom cluster initialization:
set.seed(123456) # save the random seed to make the results reproducble
Phi.hat <- 1/K                          # assume all clusters have the same size (we will update this later on)
Nk.hat <- matrix(N/K,nrow = K)          # refer to the above line!
Mu.hat <- as.matrix(X[sample(1:N, K), ]) # randomly  choose K samples as cluster means (any better idea?)
Sigma.hat <- matrix(,nrow = K, ncol = D^2) # create empty covariance matrices (we will fill them)
post <- matrix(,nrow=N, ncol=K)        # empty posterior matrix (the membership estimates will be stored here)

### for each cluster k:
for (k in 1:K){
    #### initiate the k covariance matrix as an identity matrix (we will update it later on)
    Sigma.hat[k,] <- diag(D) # initialize with identity covariance matrix
}

# Build the GMM model
Mu.hat.old <- Mu.hat # store the old estimated means

# Main Loop
while (!terminate){

    # E step:   
    for (k in 1:K){
        ## calculate the posterior based on the estimated means,covariance and cluster size:
        post[,k] <- dmvnorm(X, Mu.hat[k,],  matrix(Sigma.hat[k,], ncol=D)) * Nk.hat[k]
    }
    
    # hard assignments:
    max.prob <- post==apply(post, 1, max) # for each point find the cluster with the maximum (estimated) probability
    post[max.prob] <- 1 # assign each point to the cluster with the highest probability
    post[!max.prob] <- 0 # remove points from clusters with lower probabilites


    # M step:
    for (k in 1:K){
        ## recalculate the estimations:
        Nk.hat[k] <- sum(post[,k])        # the effective number of point in cluster k
        Phi.hat[k] <- sum(post[,k])/N     # the relative cluster size
        Mu.hat[k,] <- colSums(post[,k] *X)/Nk.hat[k] # new means (cluster cenroids)
        Sigma.hat[k,] <- (t(X-matrix(Mu.hat[k,],nrow = N, ncol=D, byrow = TRUE))%*%
                          (post[,k]*(X-matrix(Mu.hat[k,],nrow = N, ncol=D, byrow = TRUE))))/Nk.hat[k] # new covariance
    
    }
    
    # visualization
    if (eta %% 10 == 1){
        print(ggplot(data=as.data.frame(X), aes(x=X1, y=X2)) + 
        geom_point(color=rgb(post), alpha=0.75) +
        ggtitle (paste('Hard EM (eta=', eta,')')) + theme_minimal())
        line <- readline()
    }
    par(new=FALSE)
    # increase the epoch counter
    eta <- eta+1
    
    # check the termination criteria
    terminate <- eta > eta.max | sum(abs(Mu.hat.old - Mu.hat)) <= epsilon
    
    # record the means (neccessary for checking the termination criteria)
    Mu.hat.old <- Mu.hat
}
# That's it! Let see how many iterations we had:
cat('maximum number of itterations:',eta,'\n')

