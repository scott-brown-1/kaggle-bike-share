#########################
### Imports and setup ###
#########################

library(poissonreg)
library(tidyverse)
library(tidymodels)

source('./scripts/bike_share_analysis.R')
source('./scripts/linear_regression.R')

# Whether or not to log transform the response variable
LOG_TRANSFORM <-T

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

# Set up preprocessing
prepped_recipe <- setup_train_recipe(train, as_numeric=TRUE)

# Bake recipe
bake(prepped_recipe, new_data=train)
bake(prepped_recipe, new_data=test)

#########################
## Fit Regression Model #
#########################

# Define Poisson model
# TODO: give penalty and mixture parameters
poisson_model <- poisson_reg(penalty=tune(), mixture=tune()) %>%
  set_engine('glmnet')

# Define workflow
poisson_workflow <- workflow(prepped_recipe) %>%
  add_model(poisson_model)

# Define a grid of hyperparameters
tuning_grid <- grid_regular(
  penalty(),
  mixture(),
  levels = 10 #5^2 tuning possibilities
)

# Specify the resampling strategy (e.g., 10-fold cross-validation)
cv <- vfold_cv(data=train, v = 10, repeats=1)

# Perform parameter tuning
tune_results <- poisson_workflow %>%
  tune_grid(
    resamples=cv,
    grid=tuning_grid,
    metrics=metric_set(rmse,mae,rsq))

# Graph tuning
collect_metrics(tune_results) %>% # Gathers metrics into DF8
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()

# Get the best hyperparameters
best_params <- tune_results %>%
  select_best('rmse')

# Create and fit the best model
final_workflow <-
  poisson_workflow %>%
  finalize_workflow(best_params) %>%
  fit(data=train)

# Predict new rentals
y_pred <- predict(final_workflow, new_data=test)
if (LOG_TRANSFORM) y_pred <- exp(y_pred)

# Create output df in Kaggle format
output <- data.frame(
  datetime=as.character(format(test$datetime)),
  count=y_pred$.pred
)

vroom::vroom_write(output,'./outputs/poisson_predictions.csv',delim=',')
