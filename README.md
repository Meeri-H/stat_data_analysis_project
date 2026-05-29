# Statistical Data Analysis Project

Statistical analysis of an urban bike rental dataset, covering exploratory data analysis, descriptive statistics, and hypothesis testing. The dataset (bikes.data) contains rental records with variables including ticket type, trip cost, duration, distance, route, electrical assistance usage, and energy consumption.

## Analysis Overview

### Data Exploration
- Shapiro-Wilk normality tests on all quantitative variables (cost, duration, distance, energy used/collected)
- Descriptive statistics: mean, standard deviation, and median per variable
- Bar plots for categorical variables (ticket type, month, location, assistance)
- Density plots for quantitative variables
- Detection and removal of erroneous negative values

### Hypothesis Testing
- Six research questions are tested:
  1. Do single and season ticket users have different travel durations?
  2. Do single and savonia users differ in electrical assistance usage?
  3. Does distance travelled vary across months?
  4. Does distance correlate with energy used when assistance is enabled?
  5. Do season and savonia tickets differ in returning bikes to the start location?
  6. Does travel duration correlate with average speed?

## Files
- Statistical_Data_Analysis_Project.py: Main analysis script
- bikes.data: Bike rental dataset (CSV format)

## Dependencies
- pandas
- scipy
- matplotlib
- seaborn
- numpy

