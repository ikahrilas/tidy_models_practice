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
