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

tune_spec
