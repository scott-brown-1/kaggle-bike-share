#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)

prep_train <- function(df, log_transform=T){
  #########################
  ##### Data Cleaning #####
  #########################
  
  # Rename response var to avoid conflict with builtin "count" function
  # NOTE: no longer doing this; Kaggle requires "count" name
  # df <- rename(df, rentals = count)
  
  # Drop features that we don't know yet or don't want
  if(('casual') %in% colnames(df) & 'registered' %in% colnames(df)){
    df <- df %>%
      select(-casual,-registered)
  }

  # Log transform response variable to avoid negative predictions
  if(('count') %in% colnames(df) && log_transform){
    df['count'] <- log(df$count)
  }
  
  return(df)
}

prep_train_split <- function(df, log_transform=T){
  #########################
  ##### Data Cleaning #####
  #########################
  
  # Drop features that we don't know yet or don't want
  df <- df %>%select(-count)
  
  # Log transform response variable to avoid negative predictions
  if(('casual') %in% colnames(df) && ('registered') %in% colnames(df) && log_transform){
    df['casual'] <- log(df$casual+1)
    df['registered'] <- log(df$registered+1)
  }
  
  return(df)
}

z_score <- function(v) return((v - mean(v)) / sd(v))

setup_train_recipe <- function(train, as_numeric=FALSE, form=count~.){
  prelim_ft_eng <- recipe(form, data=train) %>% # Set model formula
    step_mutate(
      day_of_week=factor(lubridate::wday(datetime, label=T)), # Add day of week
      hour=lubridate::hour(datetime), # Add hour of day,
      daytime=factor(hour>6 & hour<22), # Add daytime (defined as between 6AM and 10PM)
      hour=factor(hour),
      year=factor(lubridate::year(datetime)),
      #month=factor(lubridate::month(datetime)), # Add month of year #NOTE: this reduces performance
      log_wind=log(windspeed), # Inferring that earlier jumps in windspeed more impactful
      humidity=humidity/100, # Put humidity on percentage scale
      weather=ifelse(weather==4, 3, weather), #Relabel weather 4 to 3
      #hot=factor(atemp>35),
      #cold=factor(atemp<4.45),
      #temp_z=z_score(atemp),
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
    step_rm(temp, windspeed, datetime) %>% #drop superfluous cols
    step_zv(all_predictors()) %>% # Remove zero-variance cols 
    step_normalize(all_numeric_predictors()) # Normalize features
  
  if(as_numeric){
    prelim_ft_eng <- prelim_ft_eng %>%
      step_dummy(all_nominal_predictors())
  }
  
  # Set up preprocessing
  prepped_recipe <- prep(prelim_ft_eng)
  
  return(prepped_recipe)
}
