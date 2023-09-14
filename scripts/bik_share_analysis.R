#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)

train <- vroom::vroom('./data/train.csv')
glimpse(train)

#########################
##### Data Cleaning #####
#########################

# Rename response var to avoid conflict with builtin "count" function
train <- rename(train, rentals = count)

# Clean data
train <- train %>%
  mutate(
    # Fix dtypes
    season = factor(season, levels=1:4, labels=c('spring','summer','fall','winter')),
    holiday = factor(holiday),
    workingday = factor(workingday),
    weather = factor(weather, levels=1:4, c('clear','overcast','rainy','stormy'))
  )

# Weather category 'stormy' only has one observation; reassign it to 'rainy'
train[train$weather == 'stormy','weather'] <- 'rainy'

#########################
## Feature Engineering ##
#########################

# Define model formula
formula <- rentals ~ 
  datetime + season + holiday +
  workingday + weather + temp +
  atemp + humidity + windspeed + 
  casual + registered

prelim_ft_eng <- recipe(formula, data=train) %>% # Set model formula
  step_mutate(
    day_of_week=wday(datetime, label=T), # Add day of week
    hour=hour(datetime), # Add hour of day
    log_wind=log(windspeed) # Inferring that earlier jumps in windspeed more impactful
  ) %>%
  # Drop features that we don't know yet
  # atempt is a better predictor than temp and the corr is too high to use both
  step_rm(c('casual','registered','temp')) %>%
  step_zv(all_predictors()) # Remove zero-variance cols

# Set up preprocessing
prepped_recipe <- prep(prelim_ft_eng)

# Apply preprocessing and final cleaning step to fix log(0)=Inf
train_clean <- bake(prepped_recipe, new_data=train)
train_clean[is.infinite(train_clean$log_wind),'log_wind'] <- 0

glimpse(train_clean)
View(train_clean)