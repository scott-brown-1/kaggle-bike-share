#########################
### Imports and setup ###
#########################

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

set.seed(42)

#Define BART model
#Create the workflow
bart_model <- 
  parsnip::bart(
    trees = tune(), #250,
    prior_terminal_node_coef = tune(), #0.75,
    prior_terminal_node_expo = tune(), #1.75,
  ) %>% 
  set_engine("dbarts") %>% 
  set_mode("regression")

bart_workflow <-
  workflow(prepped_recipe) %>%
  add_model(bart_model)

# Define a grid of hyperparameters
#318, 0.793, 1.86
tuning_grid <- grid_regular(
  trees(),
  prior_terminal_node_coef(),
  prior_terminal_node_expo(),
  levels = 7#0 #10^2 tuning possibilities
)

# Specify the resampling strategy (e.g., 10-fold cross-validation)
cv <- vfold_cv(data=train, v = 7, repeats=1)

# parallel tune grid
doParallel::registerDoParallel(10)

# Perform parameter tuning
tune_results <- bart_workflow %>%
  tune_grid(
    resamples=cv,
    grid=2,#tuning_grid,
    metrics=metric_set(rmse))

# Graph tuning
# collect_metrics(tune_results) %>% # Gathers metrics into DF8
#   filter(.metric=="rmse") %>%
#   ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
#   geom_line()

# Get the best hyperparameters
best_params <- tune_results %>%
  select_best('rmse')

# Create and fit the best model
final_workflow <-
  bart_workflow %>%
  finalize_workflow(best_params) %>%
  fit(data=train)

final_workflow <- bart_workflow %>%
  fit(data=train)

saveRDS(final_workflow,'./models/bart_tuned.rds')

# Predict new rentals
y_pred <- predict(final_workflow, new_data=test) 
if(LOG_TRANSFORM) y_pred <- exp(y_pred)

# Create output df in Kaggle format
output <- data.frame(
  datetime=as.character(format(test$datetime)),
  count=y_pred$.pred
)

vroom::vroom_write(output,'./outputs/bart_predictions.csv',delim=',')
