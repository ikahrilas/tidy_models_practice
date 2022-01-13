library(tidymodels)  # for the parsnip package, along with the rest of tidymodels

# Helper packages
library(readr)       # for importing data
library(broom.mixed) # for converting bayesian models to tidy tibbles
library(dotwhisker)  # for visualizing regression results
library(car)
library(psych)
library(rstanarm)

# read in the urchins data
urchins <-
  read_csv("https://tidymodels.org/start/models/urchins.csv") %>%
  # Change the names to be a little more verbose
  setNames(c("food_regime", "initial_volume", "width")) %>%
  # Factors are very helpful for modeling, so we convert one column
  mutate(food_regime = factor(food_regime, levels = c("Initial", "Low", "High")))

# check it out!
some(urchins)
describe(urchins)
glimpse(urchins)

# for the first step, let's visualize the data
ggplot(urchins,
       aes(x = initial_volume,
           y = width,
           group = food_regime,
           col = food_regime)) +
  geom_point() +
  geom_smooth(method = lm) +
  scale_color_viridis_d(option = "plasma", end = .7) +
  theme_minimal()

# initial modeling with ANOVA use linear regressoin
linear_reg() %>%
  set_engine("lm")

lm_mod <-
  linear_reg() %>%
  set_engine("lm")

lm_fit <-
  lm_mod %>%
  fit(width ~ initial_volume * food_regime, data = urchins)

# inspect the summary of the model
lm_fit %>% tidy()

# can use the tidy() output to plot the regression coefficients
tidy(lm_fit) %>%
  dwplot(dot_args = list(size = 2, color = "black"),
         whisker_args = list(color = "black"),
         vline = geom_vline(xintercept = 0, colour = "grey50", linetype = 2))



new_points <- expand.grid(initial_volume = 20,
                          food_regime = c("Initial", "Low", "High"))
new_points

# making predictions
mean_pred <- predict(lm_fit, new_data = new_points)
mean_pred

# now include confidence intervals with the prediction
conf_int_pred <- predict(lm_fit, new_data = new_points, type = "conf_int")
conf_int_pred

# Now combine:
plot_data <-
  new_points %>%
  bind_cols(mean_pred) %>%
  bind_cols(conf_int_pred)

# and plot:
ggplot(plot_data, aes(x = food_regime)) +
  geom_point(aes(y = .pred)) +
  geom_errorbar(aes(ymin = .pred_lower,
                    ymax = .pred_upper),
                width = .2) +
  labs(y = "urchin size")

# now fit the same model using the stan engine for bayesian anlaysis
## define prior cauchy distrubition
# set the prior distribution
prior_dist <- rstanarm::student_t(df = 1)

set.seed(123)

# make the parsnip model
bayes_mod <-
  linear_reg() %>%
  set_engine("stan",
             prior_intercept = prior_dist,
             prior = prior_dist)

# train the model
bayes_fit <-
  bayes_mod %>%
  fit(width ~ initial_volume * food_regime, data = urchins)

print(bayes_fit, digits = 5)

# produce some tidy output
bayes_fit %>%
  tidy(conf.int = TRUE)

# now, using the bayesian fitted model, predict values based on dataset created earlier
bayes_plot_data <-
  new_points %>%
  bind_cols(predict(bayes_fit, new_data = new_points)) %>%
  bind_cols(predict(bayes_fit, new_data = new_points, type = "conf_int"))

ggplot(bayes_plot_data, aes(x = food_regime)) +
  geom_point(aes(y = .pred)) +
  geom_errorbar(aes(ymin = .pred_lower, ymax = .pred_upper), width = .2) +
  labs(y = "urchin size") +
  ggtitle("Bayesian model with t(1) prior distribution")
