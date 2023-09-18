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
