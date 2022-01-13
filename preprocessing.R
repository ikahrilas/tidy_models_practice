library(tidymodels)

# load packages as well
library(nycflights13)
library(skimr) # for skimming the data

set.seed(123)

flight_data <-
  flights %>%
  mutate(
    # Convert the arrival delay to a factor
    arr_delay = ifelse(arr_delay >= 30, "late", "on_time"),
    arr_delay = factor(arr_delay),
    # We will use the date (not date-time) in the recipe below
    date = lubridate::as_date(time_hour)
  ) %>%
  # Include the weather data
  inner_join(weather, by = c("origin", "time_hour")) %>%
  # Only retain the specific columns we will use
  select(dep_time, flight, origin, dest, air_time, distance,
         carrier, date, arr_delay, time_hour) %>%
  # Exclude missing data
  na.omit() %>%
  # For creating models, it is better to have qualitative columns
  # encoded as factors (instead of character strings)
  mutate_if(is.character, as.factor)

# about 16% of flights arrived late
flight_data %>%
  count(arr_delay) %>%
  mutate(prop = n/sum(n))

# variables that we don't want in our model, but could be useful later
flight_data %>%
  skim(dest, carrier)

# splitting up the data
set.seed(222)
## here we use the rsample package by first creating an object
## with out to split our data - by putting 3/4 of data in the training set
data_split <- initial_split(flight_data, prop = 3/4)

## Now, create data frames for the two sets:
train_data <- training(data_split)
test_data  <- testing(data_split)

# can confirm that the split worked
nrow(flight_data) * .75 # 244364.2
nrow(train_data)        # 244364

# now, let's create a recipe for the logistic regression model
flights_rec <-
  recipe(arr_delay ~ ., data = train_data)

# we can add roles to this recipe and update it. here, we define a custom role for
# the flight and time hour variables - ID
flights_rec <-
  flights_rec %>%
  update_role(flight, time_hour, new_role = "ID")

# check out the recipe
summary(flights_rec)

## This step of adding roles to a recipe is optional;
## the purpose of using it here is that those two variables
## can be retained in the data but not included in the model.
## This can be convenient when, after the model is fit, we want
## to investigate some poorly predicted value. These ID columns
## will be available and can be used to try to understand what went wrong.

# now onto some feature engineering. perhaps the data will impact the likelihood
# of a late arrival. more specifically, maybe the day of the week, month, and whether
# it was a hoiday might impact the likelihood. let's include that information in our recipe.
flights_rec <-
  recipe(arr_delay ~ ., data = train_data) %>%
  update_role(flight, time_hour, new_role = "ID") %>%
  step_date(date, features = c("dow", "month")) %>%     # create two new columns designating dow and month
  step_holiday(date,
               holidays = timeDate::listHolidays("US"), # binary variable of whether it's a holiday or not
               keep_original_cols = FALSE)              # remove original date variable

summary(flights_rec)

# note that the recipe formula does not automatically convert character class data into dummy vars
# you must declare it yourself
flights_rec <-
  recipe(arr_delay ~., data = train_data) %>%
  update_role(flight, time_hour, new_role = "ID") %>%
  step_date(date, features = c("dow", "month")) %>%
  step_holiday(date,
               holidays = timeDate::listHolidays("US"),
               keep_original_cols = FALSE) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) # remove zero variance predictors

test_data %>%
  distinct(dest) %>%
  anti_join(train_data) # dummy value with one vaue in test set, necessitating step_zv

# now, we can fit this model
lr_mod <-
  logistic_reg() %>%
  set_engine("glm")

# pair model and recipe together
flights_wflow <-
  workflow() %>%
  add_model(lr_mod) %>%
  add_recipe(flights_rec)

flights_wflow

# fit the model to the training set
lr_mod_fit <-
  flights_wflow %>%
  fit(data = train_data)

# predict arrival delays on the test data
predict(lr_mod_fit, test_data)
## this outputs the class. if we want to output probabilities instead, use the following command:

flights_aug <- augment(lr_mod_fit, test_data)
## can also specify type = "prob" in predict() function

flights_aug %>%
  select(arr_delay, time_hour, flight, .pred_class, .pred_on_time)

# use ROC curve for fit metric
flights_aug %>%
  roc_curve(truth = arr_delay, .pred_late) %>%
  autoplot()

## can also just get area under curve metric
flights_aug %>%
  roc_auc(truth = arr_delay, .pred_late)

