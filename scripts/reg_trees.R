#########################
### Imports and setup ###
#########################

library(rpart)
library(tidyverse)
library(tidymodels)

source('./scripts/bike_share_analysis.R')

# Whether or not to log transform the response variable
LOG_TRANSFORM <-T

train <- prep_train(vroom::vroom('./data/train.csv'), log_transform=LOG_TRANSFORM)
test <- vroom::vroom('./data/test.csv')

#########################
## Feature Engineering ##
#########################

# Set up preprocessing
prepped_recipe <- setup_train_recipe(train, as_numeric=FALSE)

# Bake recipe
bake(prepped_recipe, new_data=train)
bake(prepped_recipe, new_data=test)

#########################
## Fit Regression Model #
#########################

# Define Tree model
tree_model <- decision_tree(
    tree_depth = tune(),
    cost_complexity = tune(),
    min_n=tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")

# Define workflow
tree_wf <- workflow(prepped_recipe) %>%
  add_model(tree_model)

# Define a grid of hyperparameters
tuning_grid <- grid_regular(
  tree_depth(),
  cost_complexity(),
  min_n(),
  levels = 7 #10^2 tuning possibilities
)

# Specify the resampling strategy (e.g., 10-fold cross-validation)
cv <- vfold_cv(data=train, v=8, repeats=1)

# parallel tune grid
doParallel::registerDoParallel(10)

# Perform parameter tuning
tune_results <- tree_wf %>%
  tune_grid(
    resamples=cv,
    grid=tuning_grid,
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
  tree_wf %>%
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

vroom::vroom_write(output,'./outputs/tree_predictions.csv',delim=',')
