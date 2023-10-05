#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)
library(baguette)

source('./scripts/bike_share_analysis.R')

# Whether or not to log transform the response variable
LOG_TRANSFORM <-T

train <- prep_train(vroom::vroom('./data/train.csv'), log_transform=LOG_TRANSFORM)
test <- vroom::vroom('./data/test.csv')

#########################
## Feature Engineering ##
#########################

set.seed(42)

# parallel tune grid
doParallel::registerDoParallel(10)

# Set up preprocessing
prepped_recipe <- setup_train_recipe(train, as_numeric=FALSE)

# Bake recipe
bake(prepped_recipe, new_data=train)
bake(prepped_recipe, new_data=test)

#########################
## Fit Regression Model #
#########################

# Define Tree model
nlp_bag_model <- bag_mlp(
  hidden_units = 200,
  penalty = 0.01,
  epochs = 20,
) %>%
  set_engine('nnet') %>%
  set_mode("regression")

# Define workflow
nlp_bag_wf <- workflow(prepped_recipe) %>%
  add_model(nlp_bag_model)

final_wf <- nlp_bag_wf %>%
  fit(data=train)

# Predict new rentals
y_pred <- predict(final_wf, new_data=test)
if (LOG_TRANSFORM) y_pred <- exp(y_pred)

# Create output df in Kaggle format
output <- data.frame(
  datetime=as.character(format(test$datetime)),
  count=y_pred$.pred
)

vroom::vroom_write(output,'./outputs/neural_predictions.csv',delim=',')

### BREAK ###

# Define a grid of hyperparameters
tuning_grid <- grid_regular(
  mtry(range=c(4,ncol(train))),
  min_n(),
  levels = 5 #10^2 tuning possibilities
)

# Specify the resampling strategy (e.g., 5-fold cross-validation)
cv <- vfold_cv(data=train, v=5, repeats=1)

# parallel tune grid
doParallel::registerDoParallel(10)

# Perform parameter tuning
tune_results <- forest_wf %>%
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
  forest_wf %>%
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

vroom::vroom_write(output,'./outputs/forest_predictions.csv',delim=',')