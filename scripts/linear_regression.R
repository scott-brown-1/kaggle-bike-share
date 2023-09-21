#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)

source('./scripts/bike_share_analysis.R')

# Whether or not to log transform the response variable
LOG_TRANSFORM <- T

train <- prep_train(vroom::vroom('./data/train.csv'), log_transform=LOG_TRANSFORM)
test <- vroom::vroom('./data/test.csv')

predict_with_lm <- function(train, test, log_transform=LOG_TRANSFORM){
  #########################
  ## Feature Engineering ##
  #########################
  
  prepped_recipe <- setup_train_recipe(train)
  
  # Bake recipe
  bake(prepped_recipe, new_data=train)
  bake(prepped_recipe, new_data=test)
  
  #########################
  ## Fit Regression Model #
  #########################
  
  # Define linear model
  linear_model <- linear_reg() %>%
    set_engine('lm')
  
  lm_workflow <- workflow(prepped_recipe) %>%
    add_model(linear_model) %>%
    fit(data = train)
  
  # Predict new rentals
  y_pred <- (if(log_transform) exp(predict(lm_workflow, new_data=test)) 
             else predict(lm_workflow, new_data=test))
  
  # Create output df in Kaggle format
  output <- data.frame(
    datetime=as.character(format(test$datetime)),
    count=y_pred$.pred
  )
  
  vroom::vroom_write(output,'./outputs/lm_predictions.csv',delim=',')
  
  return(output)
}

#predict_with_lm(train, test, log_transform=T)
