library(tidyverse)
library(vroom)

bike <- vroom('../data/train.csv')

print(head(bike))