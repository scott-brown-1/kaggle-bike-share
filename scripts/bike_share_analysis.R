#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)

prep_data <- function(df){
  #########################
  ##### Data Cleaning #####
  #########################
  
  # Rename response var to avoid conflict with builtin "count" function
  # NOTE: no longer doing this; Kaggle requires "count" name
  # df <- rename(df, rentals = count)
  
  # Fix dtypes
  df <- df %>%
    mutate(
      season = factor(season, levels=1:4, labels=c('spring','summer','fall','winter')),
      holiday = factor(holiday),
      workingday = factor(workingday),
      weather = factor(weather, levels=1:4, c('clear','overcast','rainy','stormy'))
    )

  # Drop features that we don't know yet or don't want
  if(('casual') %in% colnames(df) & 'registered' %in% colnames(df)){
    df <- df %>%
      select(-casual,-registered)
  }
  
  # Log transform response variable to avoid negative predictions
  if(('count') %in% colnames(df)){
    df['count'] <- log(df$count)
  }
  
  # Weather category 'stormy' only has one observation; reassign it to 'rainy'
  df[df$weather == 'stormy','weather'] <- 'rainy'
  
  return(df)
}


#   #########################
#   ## Feature Engineering ##
#   #########################
#   
#   prelim_ft_eng <- NULL
#   if(!test_data){
#     prelim_ft_eng <- recipe(count~., data=df) %>% # Set model formula
#       step_mutate(
#         day_of_week=wday(datetime, label=T), # Add day of week
#         hour=hour(datetime), # Add hour of day
#         log_wind=log(windspeed), # Inferring that earlier jumps in windspeed more impactful
#         daytime=(hour>6 & hour<22), # Add daytime (defined as between 6AM and 10PM)
#         humidity=humidity/100 # Put humidity on percentage scale
#       ) %>%
#       step_mutate_at(
#         log_wind, fn= ~ ifelse(is.infinite(.x), 0, .x)
#       ) %>%
#       step_poly(atemp, degree=2) %>% # Add squared temperature to increase contrast
#       step_rm(temp) %>% #atempt is a better predictor than temp
#       step_zv(all_predictors()) %>% # Remove zero-variance cols 
#       step_normalize(all_numeric_predictors()) # Normalize features
#   }
#   
#   # Set up preprocessing
#   prepped_recipe <- prep(prelim_ft_eng)
#   bake(prepped_recipe, new_data=df)
#   
#   return(list(
#     'data'=df,
#     'recipe'=prepped_recipe
#   ))
# }
