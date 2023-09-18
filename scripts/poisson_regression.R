#########################
### Imports and setup ###
#########################

library(poissonreg)
library(tidyverse)
library(tidymodels)

source('./scripts/bike_share_analysis.R')
source('./scripts/linear_regression.R')

# Whether or not to log transform the response variable
LOG_TRANSFORM <- T

train <- prep_train(vroom::vroom('./data/train.csv'), log_transform=LOG_TRANSFORM)
test <- vroom::vroom('./data/test.csv')

# The train predictions will be too accurate, but will this help?
# train <- cbind(train_raw,predict_with_lm(train_raw[1:1000,], train_raw, log_transform=T)$count)
# test <- cbind(test_raw,predict_with_lm(train_raw, test_raw, log_transform=T)$count)
# names(train)[length(names(train))] <- 'lm_pred'
# names(test)[length(names(test))] <- 'lm_pred'

#########################
## Feature Engineering ##
#########################

prelim_ft_eng <- recipe(count~., data=train) %>% # Set model formula
  step_mutate(
    day_of_week=wday(datetime, label=T), # Add day of week
    hour=hour(datetime), # Add hour of day
    log_wind=log(windspeed), # Inferring that earlier jumps in windspeed more impactful
    daytime=(hour>6 & hour<22), # Add daytime (defined as between 6AM and 10PM)
    humidity=humidity/100, # Put humidity on percentage scale
    weather=ifelse(weather==4, 3, weather), #Relabel weather 4 to 3
    # Fix dtypes
    season = factor(season, levels=1:4, labels=c('spring','summer','fall','winter')),
    holiday = factor(holiday),
    workingday = factor(workingday),
    weather = factor(weather, levels=1:4, c('clear','overcast','rainy','stormy')),
  ) %>%
  step_mutate_at(
    log_wind, fn= ~ ifelse(is.infinite(.x), 0, .x)
  ) %>%
  step_poly(atemp, degree=2) %>% # Add squared temperature to increase contrast
  step_rm(temp, windspeed) %>% #drop superfluous cols
  step_zv(all_predictors()) %>% # Remove zero-variance cols 
  step_normalize(all_numeric_predictors()) # Normalize features

# Set up preprocessing
prepped_recipe <- prep(prelim_ft_eng)

# Bake recipe
bake(prepped_recipe, new_data=train)
bake(prepped_recipe, new_data=test)

#########################
## Fit Regression Model #
#########################

# Define Poisson model
poisson_model <- poisson_reg() %>%
  set_engine('glm')

poisson_workflow <- workflow(prepped_recipe) %>%
  add_model(poisson_model) %>%
  fit(data = train)

# Predict new rentals
y_pred <- (if(LOG_TRANSFORM) exp(predict(poisson_workflow, new_data=test)) 
           else predict(poisson_workflow, new_data=test))

# Create output df in Kaggle format
output <- data.frame(
  datetime=as.character(format(test$datetime)),
  count=y_pred$.pred
)

vroom::vroom_write(output,'./outputs/poisson_predictions.csv',delim=',')
