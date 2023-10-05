#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)

source('./scripts/bike_share_analysis.R')

set.seed(42)
doParallel::registerDoParallel(10)

# Whether or not to log transform the response variable
LOG_TRANSFORM <- T

train <- prep_train_split(vroom::vroom('./data/train.csv'), log_transform=LOG_TRANSFORM)
train_cas <- train %>% select(-registered)
train_reg <- train %>% select(-casual)
test <- vroom::vroom('./data/test.csv')

#########################
## Feature Engineering ##
#########################

# Set up preprocessing
prepped_cas_recipe <- setup_train_recipe(train_cas, as_numeric=TRUE, form=casual~.)
prepped_reg_recipe <- setup_train_recipe(train_reg, as_numeric=TRUE, form=registered~.)

train_predict <- function(train, test, prepped_recipe){
  # Bake recipe
  bake(prepped_recipe, new_data=train)
  bake(prepped_recipe, new_data=test)
  
  #########################
  ## Fit Regression Model #
  #########################
  
  set.seed(42)
  
  #Define BART model
  #Create the workflow
  bart_model <- 
    parsnip::bart(
      trees = 318,#,tune(), #250,
      prior_terminal_node_coef = 0.793,#tune(), #0.75,
      prior_terminal_node_expo = 1.86#tune(), #1.75,
    ) %>% 
    set_engine("dbarts") %>% 
    set_mode("regression")
  
  bart_workflow <-
    workflow(prepped_recipe) %>%
    add_model(bart_model)
  
  final_workflow <- bart_workflow %>%
    fit(data=train)
  
  # Predict new rentals
  y_pred <- predict(final_workflow, new_data=test) 
  if(LOG_TRANSFORM) y_pred <- exp(y_pred)-1
  
  return(y_pred)
}

casual_pred <- train_predict(train_cas, test, prepped_recipe=prepped_cas_recipe)
registered_pred <- train_predict(train_reg, test, prepped_recipe=prepped_reg_recipe)
y_pred <- casual_pred + registered_pred

# Create output df in Kaggle format
output <- data.frame(
  datetime=as.character(format(test$datetime)),
  count=y_pred$.pred
)

vroom::vroom_write(output,'./outputs/bart_split_predictions.csv',delim=',')
