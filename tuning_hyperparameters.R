library(tidymodels)

# Helper packages
library(rpart.plot)  # for visualizing a decision tree
library(vip)         # for variable importance plots

# load the data
data(cells, packages = "cellsdata")

# inspect
glimpse(cells)
car::some(cells)
summary(cells)

# split the data into training and test sets
cell_split <- initial_split(cells %>% select(-case),
                            prop = 3/4,
                            strata = class)
cell_train <- training(cell_split)
cell_test <- testing(cell_split)

cell_train %>% dim() #1514 rows
cell_test %>% dim() #505 rows

# define decision tree object with tuned hyperparameters
tune_spec <-
  decision_tree(
    cost_complexity = tune(), # tune() is like a placeholder. after the tuning process,
    tree_depth = tune()       # will will decide upon single numeric values for these parameters
                ) %>%
  set_engine("rpart") %>%
  set_mode("classification")

# define a grid of hyperparameters
tree_grid <- grid_regular(cost_complexity(),
                          tree_depth(),
                          levels = 5)

# `tree_grid` is a 2x25 tibble holding hyperparameter values to be used in the models
tree_grid

## The function grid_regular() is from the dials package. It chooses sensible values to try
## for each hyperparameter; here, we asked for 5 of each. Since we have two to tune, grid_regular()
## returns 5 × 5 = 25 different possible tuning combinations to try in a tidy tibble format

# create cross validation folds for tuning
set.seed(345)
cell_folds <- vfold_cv(cell_train)

# tune the model with grid
set.seed(345)

tree_wf <-
  workflow() %>%
  add_model(tune_spec) %>%
  add_formula(class ~ .)

tree_res <-
  tree_wf %>%
  tune_grid(
    resamples = cell_folds,
    grid = tree_grid
  )

# explore the results and choose the best result
tree_res %>%
  collect_metrics()

## visualize performance
tree_res %>%
  collect_metrics() %>%
  mutate(tree_depth = factor(tree_depth)) %>%
  ggplot(aes(cost_complexity, mean, color = tree_depth)) +
  geom_line(size = 1.5, alpha = 0.6) +
  geom_point(size = 2) +
  facet_wrap(~ .metric, scales = "free", nrow = 2) +
  scale_x_log10(labels = scales::label_number()) +
  scale_color_viridis_d(option = "plasma", begin = .9, end = 0) +
  theme_minimal()

# We can see that our “stubbiest” tree, with a depth of 1,
# is the worst model according to both metrics and across all candidate
# values of cost_complexity. Our deepest tree, with a depth of 15, did better.
# However, the best tree seems to be between these values with a tree depth of 4.
# The show_best() function shows us the top 5 candidate models by default:

tree_res %>% show_best(n = 10) # 10 best performing models

tree_res %>% select_best("accuracy") # model with best accuracy

## These are the values for tree_depth and cost_complexity that
## maximize accuracy in this data set of cell images.


