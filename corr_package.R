library(corrr) #install.packages("corrr")
library(MASS)

# Simulate three columns correlating about .7 with each other
mu <- rep(0, 3)
Sigma <- matrix(.7, nrow = 3, ncol = 3) + diag(3)*.3
seven <- mvrnorm(n = 1000, mu = mu, Sigma = Sigma)

# Simulate three columns correlating about .4 with each other
mu <- rep(0, 3)
Sigma <- matrix(.4, nrow = 3, ncol = 3) + diag(3)*.6
four <- mvrnorm(n = 1000, mu = mu, Sigma = Sigma)

# bind the columns together
d <- cbind(seven, four)
## create column names
colnames(d) <- paste0("v", 1:ncol(d))

# Insert some missing values
d[sample(1:nrow(d), 100, replace = TRUE), 1] <- NA
d[sample(1:nrow(d), 200, replace = TRUE), 5] <- NA

# create correlation matrix - what a beauty
correlate(d)

# since it's a tibble, filter v1 coefficients greater than .6
correlate(d) %>%
  filter(v1 > .6)

# can also take a raw dataframe and using corrr() functions to
# make correlation matrices on the spot
x <- datasets::mtcars %>%
  correlate() %>%
  focus(-cyl, -vs, mirror = TRUE) %>%
  rearrange() %>% # groups highly correlated variables together
  shave() # fill in NA's for upper triangle

# fills in NAs with blanks
fashion(x)

# visualization for correlation matrix
rplot(x)

# now, time to use a different dataset and make cool network visualizations
datasets::airquality %>%
  correlate() %>%
  network_plot(min_cor = .1)
