# load packages
library(tidymodels)
library(modeldata) # for cells data

# explore data
data(cells, package = "modeldata")
cells
glimpse(cells)
car::some(cells)

# there is some imbalance among the outcome variable, `class`
cells %>%
  count(class) %>%
  mutate(prop = n / sum(n))

# split the dataset
## creates information that can be used to split the dataset
set.seed(123)
cell_split <- initial_split(cells %>% select(-case),
                            strata = "class")     # stratified sampling - this ensures that,
# despite the imbalance we noticed in our
# class variable, our training and test data
# sets will keep roughly the same proportions
# of poorly and well-segmented cells as in the original data.

cell_train <- training(cell_split)
cell_test <- testing(cell_split)

nrow(cell_train)
nrow(cell_test)

# ensure that stratification worked
cell_train %>%
  count(class) %>%
  mutate(prop = n / sum(n))

cell_test %>%
  count(class) %>%
  mutate(prop = n / sum(n))

# fit random forest model
rf_mod <- rand_forest(trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_fit <- rf_mod %>%
  fit(class ~ ., data = cell_train)

# inspect model
rf_fit

# for classification, can use accuracy or ROC/AUC as model metrics
## BAD -- DON'T DO THIS
## look at model metrics for training set
rf_training_pred <- predict(rf_fit, cell_train) %>%
  bind_cols(predict(rf_fit, cell_train, type = "prob")) %>%
  bind_cols(cell_train %>% select(class))

## model does exceptionally well on the training data
rf_training_pred %>%
  roc_auc(truth = class, .pred_PS)

rf_training_pred %>%
  accuracy(truth = class, .pred_class)

# now predict values based on the testing data set
rf_testing_pred <-
  predict(rf_fit, cell_test) %>%
  bind_cols(predict(rf_fit, cell_test, type = "prob")) %>%
  bind_cols(cell_test %>% select(class))

rf_testing_pred %>%
  roc_auc(truth = class, .pred_PS)

rf_testing_pred %>%
  accuracy(truth = class, .pred_class)

# use resampling techniques to avoid overfitting to training data
set.seed(345)
folds <- vfold_cv(cell_train, v = 10) # for k-folds cross validation using 10 folds
folds

# create workflow with model and formula
rf_wf <-
  workflow() %>%
  add_model(rf_mod) %>%
  add_formula(class ~ .)

# conduct the resampling with the `folds` object
set.seed(456)
rf_fit_rs <-
  rf_wf %>%
  fit_resamples(folds)

rf_fit_rs

# this is similar to the folds object, except there is the added `.metrics` column
# can extract this information using specific functions from the tune package
collect_metrics(rf_fit_rs) # averaged metrics across all 10 folds

# this is now far more similar to our test predictions!
rf_testing_pred %>%
  roc_auc(truth = class, .pred_PS)

rf_testing_pred %>%
  accuracy(truth = class, .pred_class)
