#########################
### Imports and setup ###
#########################

library(tidyverse)
library(BART)

source('./scripts/bike_share_analysis.R')

train <- vroom::vroom('./data/train.csv')
test <- vroom::vroom('./data/test.csv')

#########################
## Feature Engineering ##
#########################

prep_for_bart <- function(df){
    df <- df %>%
      mutate(
        day_of_week=wday(datetime, label=T), # Add day of week
        hour=hour(datetime), # Add hour of day
        log_wind=log(windspeed), # Inferring that earlier jumps in windspeed more impactful
        daytime=(hour>6 & hour<22), # Add daytime (defined as between 6AM and 10PM)
        humidity=humidity/100, # Put humidity on percentage scale
        atemp_squared=atemp^2 # Add squared temperature to increase contrast
      ) %>%
      select(-temp) %>% #atempt is a better predictor than temp
      mutate_if(is.factor, as.numeric) %>%
      mutate_if(is.logical, as.numeric) %>%
      mutate_if(is.ordered, as.numeric) 
    
    df[is.infinite(df$log_wind),'log_wind'] <- 0
    
    return(df)
}

prepped_train <- prep_for_bart(prep_data(train))
prepped_test <- prep_for_bart(prep_data(test))

#########################
##### Fit BART Model ####
#########################

X_train <- prepped_train %>% select(-count, -datetime) %>% as.matrix()# %>% scale()
y_train <- prepped_train %>% select(count, -datetime) %>% as.matrix()# %>% scale()
X_test <- prepped_test %>% select(-datetime) %>% as.matrix()# %>% scale()

model <- wbart(
  x.train = X_train,
  y.train = y_train,
  rm.const=T,
  ntree=325,
  ndpost=750,
  nskip=150
)

# Predict new rentals
y_pred <- Matrix::colMeans(exp(predict(model, newdata=X_test)))

# Create output df in Kaggle format
output <- data.frame(
  datetime=as.character(format(prepped_test$datetime)),
  count=y_pred
)

vroom::vroom_write(output,'./outputs/bart_predictions.csv',delim=',')
