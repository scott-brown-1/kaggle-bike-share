#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)

source('./scripts/bike_share_analysis.R')

train <- vroom::vroom('./data/train.csv')
test <- vroom::vroom('./data/test.csv')

prepped_train <- prep_data(train)
prepped_test <- prep_data(test)

#########################
## Feature Engineering ##
#########################

prelim_ft_eng <- recipe(count~., data=prepped_train) %>% # Set model formula
  step_mutate(
    day_of_week=wday(datetime, label=T), # Add day of week
    hour=hour(datetime), # Add hour of day
    log_wind=log(windspeed), # Inferring that earlier jumps in windspeed more impactful
    daytime=(hour>6 & hour<22), # Add daytime (defined as between 6AM and 10PM)
    humidity=humidity/100 # Put humidity on percentage scale
  ) %>%
  step_mutate_at(
    log_wind, fn= ~ ifelse(is.infinite(.x), 0, .x)
  ) %>%
  step_poly(atemp, degree=2) %>% # Add squared temperature to increase contrast
  step_rm(temp) %>% #atempt is a better predictor than temp
  step_zv(all_predictors()) %>% # Remove zero-variance cols 
  step_normalize(all_numeric_predictors()) # Normalize features

# Set up preprocessing
prepped_recipe <- prep(prelim_ft_eng)

bake(prepped_recipe, new_data=prepped_train)
bake(prepped_recipe, new_data=prepped_test)

#########################
## Fit Regression Model #
#########################

# Define linear model
model <- linear_reg() %>%
  set_engine('lm')

model_workflow <- workflow(prepped_recipe) %>%
  add_model(model) %>%
  fit(data = prepped_train)

# Predict new rentals
y_pred <- exp(predict(model_workflow, new_data=prepped_test))

# Create output df in Kaggle format
output <- data.frame(
  datetime=as.character(format(prepped_test$datetime)),
  count=y_pred$.pred
)

vroom::vroom_write(output,'./outputs/lm_predictions.csv',delim=',')
