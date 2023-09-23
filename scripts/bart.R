#########################
### Imports and setup ###
#########################

library(dbarts)
library(parsnip)
library(tidyverse)
library(tidymodels)

source('./scripts/bike_share_analysis.R')
source('./scripts/linear_regression.R')

# Whether or not to log transform the response variable
LOG_TRANSFORM <- T

train <- prep_train(vroom::vroom('./data/train.csv'), log_transform=LOG_TRANSFORM)
test <- vroom::vroom('./data/test.csv')

#########################
## Feature Engineering ##
#########################

# Set up preprocessing
prepped_recipe <- setup_train_recipe(train, as_numeric=TRUE)

# Bake recipe
bake(prepped_recipe, new_data=train)
bake(prepped_recipe, new_data=test)

#########################
## Fit Regression Model #
#########################

#Define BART model
#Create the workflow
bart_model <- 
  parsnip::bart(
    trees = 250,
    prior_terminal_node_coef = 0.75,
    prior_terminal_node_expo = 1.75,
    prior_outcome_range = 1.7
  ) %>% 
  set_engine("dbarts") %>% 
  set_mode("regression")

bart_workflow <-
  workflow(prepped_recipe) %>%
  add_model(bart_model) %>%
  fit(data=train)

# Predict new rentals
y_pred <- predict(bart_workflow, new_data=test) 
if(LOG_TRANSFORM) y_pred <- exp(y_pred)

# Create output df in Kaggle format
output <- data.frame(
  datetime=as.character(format(test$datetime)),
  count=y_pred$.pred
)

vroom::vroom_write(output,'./outputs/bart_predictions.csv',delim=',')

#Model tuning with grid search
# bart_model <- 
#   parsnip::bart(
#     trees = tune(),
#     prior_terminal_node_coef = tune(),
#     prior_terminal_node_expo = tune()
#   ) %>% 
#   set_engine("dbarts") %>% 
#   set_mode("regression")
# 
# #parameter object
# rf_param <- 
#   workflow() %>% 
#   add_model(bart_model) %>% 
#   add_recipe(prepped_recipe) %>% 
#   extract_parameter_set_dials() %>% 
#   finalize(train)
# 
# #space-filling design with integer grid argument
# df_folds <- vfold_cv(train)
# df_reg_tune <-
#   workflow() %>% 
#   add_recipe(prepped_recipe) %>% 
#   add_model(bart_model) %>% 
#   tune_grid(
#     df_folds,
#     grid = 5,
#     param_info = rf_param,
#     metrics = metric_set(rsq)
#   )
# 
# #Selecting the best parameters according to the r-square
# rf_param_best <- 
#   select_best(df_reg_tune, metric = "rsq") %>% 
#   select(-.config)
# 
# #Final estimation with the object of best parameters
# final_df_wflow <- 
#   workflow() %>% 
#   add_model(bart_model) %>% 
#   add_recipe(prepped_recipe) %>% 
#   finalize_workflow(rf_param_best)
# 
# set.seed(12345)
# final_df_fit <- 
#   final_df_wflow %>% 
#   last_fit(df_split)
# 
# #Computes final the accuracy metrics 
# collect_metrics(final_df_fit)