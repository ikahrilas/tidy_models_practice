#-- case study
# load packages
library(tidymodels)
library(readr)
library(vip)

# load in dataset on hotel bookings
## data dictionary here: https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-02-11#data-dictionary
hotels <-
  read_csv("https://tidymodels.org/start/case-study/hotels.csv") %>%
  mutate_if(is.character, as.factor) # mutate character-type variables to factors

# explore the data
glimpse(hotels)
car::some(hotels)
skimr::skim(hotels)

# we will build a model to predict whether hotel guests brought children,
# which is a binary variable
hotels %>% distinct(children)
hotels %>%
  count(children) %>%
  mutate(prop = n / sum(n))
## there is a huge class imbalace here, with only 8% of guests bringing children
## data will be analyzed as is, but upsampling and downsampling will be explored later

# create training and testing sets, stratified on the `children` variable
hotel_split <- initial_split(hotels, strata = children)

hotel_train <- training(hotel_split)
hotel_test <- testing(hotel_split)

# prepare for kfolds cross validation
set.seed(123)
hotel_cv <- vfold_cv(hotel_train)

# build penalized logistic regression model
lr_mod <-
  logistic_reg(
    penalty = tune(),
    mixture = 1       # pure lasso regression
  ) %>%
  set_engine("glmnet")

# create a recipe for preprocessing
holidays <- c("AllSouls", "AshWednesday", "ChristmasEve", "Easter",
              "ChristmasDay", "GoodFriday", "NewYearsDay", "PalmSunday")

lr_recipe <-
  recipe(children ~ ., data = hotel_train) %>%
  step_date(arrival_date) %>%
  step_holiday(arrival_date, holidays = holidays) %>%
  step_rm(arrival_date) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())

# create workflow
hotels_wf <-
  workflow() %>%
  add_model(lr_mod) %>%
  add_recipe(lr_recipe)

# create the tuning grid for the penalty hyperparameter
hotels_grid <- tibble(penalty = 10 ^ seq(-4, -1, length.out = 30))

hotels_grid %>% top_n(-5) # lowest penalty values
hotels_grid %>% top_n(5) # highest penalty values

# tune the model
hotels_res <-
  hotels_wf %>%
  tune_grid(
    resamples = hotel_cv,
    grid = hotels_grid,
    control = control_grid(save_pred = TRUE),
    metrics = metric_set(roc_auc))

# inspect roc_auc metrics for model performance
hotels_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(penalty, mean)) +
  geom_line(color = "grey") +
  geom_point(size = 0.5, alpha = 0.8) +
  ylab("Area under the ROC Curve") +
  scale_x_log10(labels = scales::label_number()) +
  theme_minimal()

# the model performs better when there is a smaller number of predictors, signified by the
# small penalty term having higher auc values

# inspect the best performing models
top_models <-
  hotels_res %>%
  show_best("roc_auc", n = 15) %>%
  arrange(penalty)
top_models

# find the best one
hotels_res %>% select_best("roc_auc")

hotels_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(penalty, mean)) +
  geom_line(color = "grey") +
  geom_point(size = 0.5, alpha = 0.8) +
  ylab("Area under the ROC Curve") +
  scale_x_log10(labels = scales::label_number()) +
  geom_vline(xintercept = .00137,
             color = "red")

# this model looks pretty good, and is right before the drop off in performance, so we have some sparsity
# in terms of # of predictors
# interestingly, the cv resampling gives the right choice immediately as opposed to the validation_split method.

# let's collect our victor and look at the roc curve
lr_best <-
  hotels_res %>%
  collect_metrics() %>%
  arrange(penalty) %>%
  slice(12)

lr_auc <-
hotels_res %>%
  collect_predictions(parameters = lr_best) %>%
  roc_curve(children, .pred_children) %>%
  mutate(model = "Logistic Regression")

autoplot(lr_auc)

# inspect the individual coefficients and test statistics
hotel_fit <- fit(hotels_wf, data = hotel_train)

hotel_fit %>%
  extract_fit_engine() %>%
  tidy()

tidy(hotel_fit, penalty = .00137) %>%
  filter(term != "(Intercept)") %>%
  arrange(abs(estimate) %>% desc())

# now, try a random forest model
## to save on computation time, use parallel processing
cores <- parallel::detectCores()

library(doParallel)
cl <- makePSOCKcluster(cores)
registerDoParallel(cl)

rf_mod <-
  rand_forest(mtry = tune(),
              min_n = tune(),
              trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_recipe <-
  recipe(children ~ ., data = hotel_train) %>%
  step_date(arrival_date) %>%
  step_holiday(arrival_date, holidays = holidays) %>%
  step_rm(arrival_date)

rf_workflow <-
  workflow() %>%
  add_model(rf_mod) %>%
  add_recipe(rf_recipe)

rf_mod %>% parameters()

# grid for tuning parameters
rf_res <-
  rf_workflow %>%
  tune_grid(resamples = hotel_cv,
            grid = 25,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc))

# look at the 5 best models
rf_res %>%
  show_best("roc_auc", n = 5)

## this is looking much more promising that the logistic regression model
autoplot(rf_res)

# Plotting the results of the tuning process highlights that both mtry number of
# predictors at each node) and min_n (minimum number of data points required to
# keep splitting) should be fairly small to optimize performance. However, the
# range of the y-axis indicates that the model is very robust to the choice of
# these parameter values — all but one of the ROC AUC values are greater than 0.90.

rf_best <-
  rf_res %>%
  select_best(metric = "roc_auc")

rf_best # mtry = 8 and min_n = 3

rf_auc <- rf_res %>%
  collect_predictions(parameters = rf_best) %>%
  roc_curve(children, .pred_children) %>%
  mutate(model = "Random Forest")

# compared curves between randforest model and lr
bind_rows(rf_auc, lr_auc) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) +
  geom_line(lwd = 1.5, alpha = 0.8) +
  geom_abline(lty = 3) +
  coord_equal() +
  scale_color_viridis_d(option = "plasma", end = .6)

# now, refit final model
last_rf_mod <-
  rand_forest(mtry = 8, min_n = 3, trees = 1000) %>%
  set_engine("ranger", num.threads = 1, importance = "impurity") %>% # importance arg to extract variable importance
  set_mode("classification")

last_rf_wf <-
  rf_workflow %>%
  update_model(last_rf_mod)

# the last fit
set.seed(234)
last_rf_fit <-
  last_rf_wf %>%
  last_fit(hotel_split)

last_rf_fit

# inspect the final metrics
last_rf_fit %>%
  collect_metrics()

# look at most important variables
last_rf_fit %>%
  pluck(".workflow", 1) %>%
  extract_fit_parsnip() %>%
  vip(num_featers = 20)

# The most important predictors in whether a hotel stay
# had children or not were the daily cost for the room, the
# type of room reserved, the type of room that was ultimately
# assigned, and the time between the creation of the reservation
# and the arrival date.

# look at roc curve
last_rf_fit %>%
  collect_predictions() %>%
  roc_curve(children, .pred_children) %>%
  autoplot()

# in light of these findings, we have good confidene that this model with these hyperparameters
# would do a good job predicting new data
