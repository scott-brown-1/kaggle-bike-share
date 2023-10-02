#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)
library(poissonreg)
library(stacks)

source('./scripts/bike_share_analysis.R')

# Whether or not to log transform the response variable
LOG_TRANSFORM <- T
SEED <- 42
N_CORES <- 9 # Number of cores for parallelizing models/tuning

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
#### Prepare Stacking ###
#########################

## Set seed
set.seed(SEED)

## Parallel tune grid
doParallel::registerDoParallel(N_CORES)

## Specify the resampling strategy (e.g., 10-fold cross-validation)
cv <- vfold_cv(data=train, v=5, repeats=1)

## Create a control grid
untuned_model <- control_stack_grid() #If tuning over a grid
# tuned_model <- control_stack_resamples() #If not tuning a model

#########################
## PREPARE BASE MODELS ##
#########################

##### POISSON REGRESSION #####

# Define Poisson model
poisson_model <- poisson_reg(
  penalty=tune(), 
  mixture=tune()) %>%
  set_engine('glmnet')

# Define workflow
poisson_workflow <- workflow(prepped_recipe) %>%
  add_model(poisson_model)

# Define a grid of hyperparameters
poisson_tuning_grid <- grid_regular(
  penalty(),
  mixture(),
  levels = 10 #5^2 tuning possibilities
)

# Perform parameter tuning
tuned_poisson_models <- poisson_workflow %>%
  tune_grid(
    resamples=cv,
    grid=poisson_tuning_grid,
    metrics = metric_set(rmse), # or mae, rsq, etc.
    control = untuned_model)

##### BART #####

## Create model
bart_model <- 
  parsnip::bart(
    trees = tune(),
    prior_terminal_node_coef = tune(),
    prior_terminal_node_expo = tune(),
  ) %>% 
  set_engine("dbarts") %>% 
  set_mode("regression")

## Create workflow
bart_workflow <-
  workflow(prepped_recipe) %>%
  add_model(bart_model)

# Define a grid of hyperparameters
bart_tuning_grid <- grid_regular(
  trees(),
  prior_terminal_node_coef(),
  prior_terminal_node_expo(),
  levels = 3 #5^2 tuning possibilities
)

# Perform parameter tuning
tuned_bart_models <- bart_workflow %>%
  tune_grid(
    resamples=cv,
    grid = bart_tuning_grid,
    metrics = metric_set(rmse), # or mae, rsq, etc.
    control = untuned_model)

##### RANDOM FOREST #####

# Define forest model
forest_model <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 700) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# Define workflow
forest_wf <- workflow(prepped_recipe) %>%
  add_model(forest_model)

# Define a grid of hyperparameters
forest_tuning_grid <- grid_regular(
  mtry(range=c(4,ncol(train))),
  min_n(),
  levels = 5 #5^2 tuning possibilities
)

# Perform parameter tuning
tuned_forest_models <- forest_wf %>%
  tune_grid(
    resamples=cv,
    grid = forest_tuning_grid,
    metrics = metric_set(rmse), # or mae, rsq, etc.
    control = untuned_model)

#########################
# Stack models together #
#########################

# Create meta learner
bike_stack <- stacks() %>%
  add_candidates(tuned_poisson_models) %>%
  #add_candidates(tuned_bart_models) %>%
  add_candidates(tuned_forest_models)

# Fit meta learner
fitted_bike_stack <- bike_stack %>%
  blend_predictions() %>% # This is a Lasso (L1) penalized reg model
  fit_members()

# you can use as_tibble(bike_stack) to change meta learner to another model
# (but penalized reg w/ Lasso works well b/c of variable selection)

# See which trees were kept (view coefficients)
# collect_parameters(fitted_bike_stack, "tree_folds_fit")

# Predict with stacked models
y_pred <- predict(fitted_bike_stack, new_data=test)
if (LOG_TRANSFORM) y_pred <- exp(y_pred)

# Create output df in Kaggle format
output <- data.frame(
  datetime=as.character(format(test$datetime)),
  count=y_pred$.pred
)

vroom::vroom_write(output,'./outputs/stacked_model_preds.csv',delim=',')
