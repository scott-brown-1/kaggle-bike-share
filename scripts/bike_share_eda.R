#########################
### Imports and setup ###
#########################

# Explore data descriptions on Kaggle at https://www.kaggle.com/competitions/bike-sharing-demand/data

# Import packages
library(tidyverse)
library(DataExplorer)

# Load data
bike <- vroom::vroom('./data/train.csv')

###########################
####### Examine Data ######
###########################

# Exmaine dataframe and check data types; check shape
glimpse(bike)
View(bike)

# Check categorical vs discrete vs continuous factors
intro_plot <- plot_intro(bike)
intro_plot

# Fix necessary dtypes
bike['season'] <- factor(bike$season)
bike['holiday'] <- factor(bike$holiday)
bike['workingday'] <- factor(bike$workingday)
bike['weather'] <- factor(bike$weather)

# Rename response var to avoid conflict with builtin "count" function
bike <- rename(bike, rentals = count)

###########################
### Check Missing Values ##
###########################

# Count total missing values
sum(sum(is.na(bike)))

# View missing values by feature
plot_missing(bike)

###########################
## Check For Consant Vals #
###########################

# View feature variance to find near-zero variance features
# Variance of categorical features is less useful

# Calculate the variance of relevant numeric features and store results in df
variances <- sapply(bike[,c('temp','atemp','humidity','windspeed')], var)
variance_df <- data.frame(Feature = names(variances), Variance = variances)

# Plot variances
variance_plot <- ggplot(variance_df, aes(x = Feature, y = Variance)) +
  geom_bar(stat = "identity", fill = "skyblue2") +
  labs(title = "Variance of Relevant Features",
       x = "Features",
       y = "Variance") +
  theme_classic()

variance_plot

###########################
## Visually examine data ##
###########################

# each feature; check for outliers
plot_histogram(bike)
plot_bar(bike)

###########################
# Check Var Relationships #
###########################

# Plot correlation heatmap
# NOTE: Although 'registered' and 'casual' have high correlations
# with response var, we don't know them unitl after we know count
corr_plot <- plot_correlation(bike)
corr_plot

###########################
### Examine response var ##
###########################
summary(bike$rentals)

response_hist <- ggplot(bike, aes(x=rentals)) +
  geom_histogram(binwidth=10, fill='skyblue') +
  labs(title="Histogram of Rental Count",x="Number of rentals", y = "Count of observations") +
  theme_classic()

response_hist

###########################
#### Create summary viz ###
###########################

# Create layout plot of relevant visualizations and save plot
combo_plot <- intro_plot + corr_plot + response_hist + variance_plot + 
  patchwork::plot_layout(widths = c(2, 1))

combo_plot

# Save plot
ggsave('combo_plot.jpg',combo_plot)

