import pandas as pd

# Load the data

df = pd.read_csv('bikes.data', sep=",", header=[0])

print(df)


"""
Variable properties: 
    ticket : categorical, nominal
    cost : quantitative, discrete, ordinal, ratio
    month : categorical, nominal
    location_from : categorical, nominal
    location_to : categorical, nominal
    duration : quantitative, discrete, ordinal, ratio
    distance : quantitative, discrete, ordinal, ratio
    assistance : categorical, only two types of variables -> indicator
    energy_used : quantitative, continuous, ordinal, ratio
    energy_collected : quantitative, continuous, ordinal, ratio
    
"""

# Basic statistics: 

from scipy import stats 

# Check the basic info of the data
df.info()

# Check if the quantitative variables are normally distributed using Shapiro-Wilk test

df_quantitative = df.loc[:,["cost","duration","distance","energy_used","energy_collected"]]

print(stats.shapiro(df.cost).pvalue)
print(stats.shapiro(df.duration).pvalue)
print(stats.shapiro(df.distance).pvalue)
print(stats.shapiro(df.energy_used).pvalue)
print(stats.shapiro(df.energy_collected).pvalue)

# None of the values are normally distributed 

# Calculate the mean, standard deviation and median for all the quantitative variables

print(pd.concat({'mean': df_quantitative.mean(), 'std': df_quantitative.std(), 
                 'median' : df_quantitative.median()},axis=1))

# Visualize data as plots

import matplotlib.pyplot as plt
import seaborn as sns

# Set color palette
sns.set_palette('colorblind')

# Let's do barplots for categorical variables 
sns.countplot(df["ticket"])
plt.title("Amount of different ticket types bought")
plt.show()

fig=sns.countplot(df["month"])
plt.title("Amount of bikes rented monthly")
fig.set(xticklabels=["April","May","June","July","August","September", "October"])
plt.show()

sns.countplot(df["location_from"])
plt.title("Amount of bikes rented from different locations")
plt.xticks(rotation=50)
plt.xlabel("location from")
plt.show()

sns.countplot(df["location_to"])
plt.title("Amount of bikes rented to different locations")
plt.xticks(rotation=50)
plt.xlabel("location to")
plt.show()

fig1=sns.countplot(df["assistance"])
plt.title("Amount of rented bikes using and not using electrical assistance")
fig1.set(xticklabels=["No","Yes"])
plt.show()

# For quantitative varibales we'll do density plots

df["cost"].plot.density()
plt.title("Cost of the rented bikes in euros")
plt.xlabel("cost in euros")
plt.show()

df["duration"].plot.density()
plt.title("Duration of the rental in seconds")
plt.xlabel("duration [s]")
plt.show()

df["distance"].plot.density()
plt.title("Distance of the trip in meters")
plt.xlabel("distance [m]")
plt.show()

df["energy_used"].plot.density()
plt.title("Energy used by the bike in watt-hours")
plt.xlabel("energy [Wh]")
plt.show()

df["energy_collected"].plot.density()
plt.title("Energy collected by the bike in watt-hours")
plt.xlabel("energy [Wh]")
plt.show()


# From the density plots it seems that there are negative values for all the quantitative variables
# and that cannot be correct. The data for the costs of the rentals is also difficult to
# interpret since the cost for season and savonia tickets is always zero so most of the cost data
# is just zero. 

# Next we'll check and remove the negative values.
import numpy as np

df.loc[(df['cost'] < 0),'cost']=np.nan

df.loc[(df['duration'] < 0),'duration']=np.nan
    
df.loc[(df['distance'] < 0), 'distance']=np.nan

df.loc[(df['energy_used'] < 0), 'energy_used']=np.nan

df.loc[(df['energy_collected'] < 0), 'energy_collected']=np.nan

df.info()

# Only the distance values have negative values. 

df.dropna

### DATA EXPLORATION ###

# Next we'll calculate the total distance, total time and the total amount of fees
# per each ticket type. After that we'll calculate the mean distance, the mean energy consumed 
# and the mean energy collected per the status of assistance.

print("Total amount of cost, distance and duration per ticket type:")
print(df.groupby("ticket").sum().iloc[:,[0,3,2]], end="\n\n")

print("Mean distance, mean energy consumed and mean energy collected per status of assistance:")
print(df.groupby("assistance").mean().iloc[:,[3,4,5]], end="\n\n")

# Let's find out the three most popular locations for each ticket type. First we'll create
# variables for each ticket type and then we can calculate the amount of each location in each
# of these ticket types and sum the start and end locations.

season = df.loc[df.ticket == "season"]
season_amount = season.location_from.value_counts()+season.location_to.value_counts()

single = df.loc[df.ticket == "single"]
single_amount = single.location_from.value_counts()+single.location_to.value_counts()

savonia = df.loc[df.ticket == "savonia"]
savonia_amount = savonia.location_from.value_counts()+savonia.location_to.value_counts()

result=pd.concat({'season': season_amount,'single': single_amount,'savonia': savonia_amount},axis=1)
print(result, end="\n\n")

# The three most popular location for season tickets are Tori, Väinölänniemi and Satama,
# for single tickets they are Tori, Satana and Väinölänniemi/Teatteri and for savonia tickets
# they are Microteknia, Tori and Snellmania.

# Next we'll visualise the monthly distance travelled per every ticket type. 

distance_sums = df.groupby(["month","ticket"]).sum().iloc[:,4]

season_dist = distance_sums[:,"season"]
single_dist = distance_sums[:,"single"]
savonia_dist = distance_sums[:,"savonia"]

f=pd.concat({'season': season_dist,'single': single_dist,'savonia': savonia_dist},axis=1)

fig_distance = f.plot.bar(title="Distance travelled monthly per every ticket type", ylabel="distance")
fig_distance.set(xticklabels=["April","May","June","July","August","September", "October"])
plt.show()

# The results are very intuitional: during the summer months the distance travelled than in the 
# spring and autumn months since the amount of bikes rented during summer is higher. During summer there's
# also more bikes rented with the single ticket type since people using season and savonai tickets are 
# probably using them to travel to work or school. The savonia tickets are probably used by students so
# it makes sense that the amount of distance travelled with the savonia tickets is highest during 
# autumn months.

# Let's visualise the net energy gain

energy_gain=df.energy_collected-df.energy_used

plt.plot(energy_gain,'o',markersize=1)
plt.xlabel("count")
plt.ylabel("energy gain [Wh]")
plt.title("Net energy gained during the rentals")
plt.show()

# The net energy gain is mostly negative which makes sense, since many of the renters used electrical assistance
# which consumes energy. 

# Lastly we'll visualise the pairwise relationships between the quantitative variables

# Quantitative variables: cost, duration, distance, energy_used, energy_collected

quantitatives = df.filter(['cost','duration','distance','energy_used','energy_collected'])

pd.plotting.scatter_matrix(quantitatives,figsize=(14,14))
#plt.suptitle("test")

# The pairwise relationships seem to be reasonable, since for example the cost of the trip correlates
# strongly with duration, distance and energy_used and that makes sense because the cost of the trip
# is higher when you use the bike longer. 

### HYPOTHESIS TESTING ###

# P-value is a number that is used to help decide whether the null hypothesis of a statistical
# test should be either accepted or rejected. The null hypothesis can be, for example, that there's
# no difference between the studied groups, and the smaller the obtained p-value is, the more likely
# it is that the null hypothesis would be very unlikely for said groups. So p-value can be used to
# make conclusions in such way that you have to know the null hypothesis of the statistical test 
# you want to use, and from the p-value you can determine whether the null hypothesis is either 
# likely or unlikely for the situation in question. However, p-values aren't always reliable if the
# data set is for example too small or too big, and you have to be careful to pick the right statistical
# test for your situation so that you won't get a wrong p-value.

# For the first question I've chosen to use Mann-Whitney U test, since in this question we analyze
# two numerical variables that are unpaired and not normally distributed, and the null hypothesis 
# is suitable for testing if there's difference between the two groups.

print("p-value for first question:", stats.mannwhitneyu(single["duration"], 
                                     season["duration"],use_continuity=True, alternative='two-sided')[1])


# The p-value is over 0.05, so there's no statistical evidence that the travel times are either longer 
# or shorter for the single ticket than the season ticket.

# For the second question we'll use Chi-squared t-test

assistance_counts = pd.concat({"single": single.assistance.value_counts(), 
                               "savonia": savonia.assistance.value_counts()}, axis=1)

print("p-value for the second question:", stats.chi2_contingency(assistance_counts)[1])

# The p-value is smaller than 0.05, so there seems to be statistical evidence that there's 
# a difference between the single and savonia tickets on how often the elctrical assistance is used

# For the third question we can use Kruskal-Wallis test

april = df.loc[df.month == 4]
may = df.loc[df.month == 5]
june = df.loc[df.month == 6]
july = df.loc[df.month == 7]
august = df.loc[df.month == 8]
september = df.loc[df.month == 9]
october = df.loc[df.month == 10]

print("p-value for the third question:", 
      stats.kruskal(april.distance,may.distance,june.distance, july.distance,
                    august.distance,september.distance,october.distance,nan_policy='omit').pvalue)

# The p-value is under 0.05, so there seems to be statistical evidence that the travel distances tend to be
# shorter or longer during one month than during the others. From the figure that was presented earlier it 
# can be noticed that during April the travel distances are significantly lower than during other months, so 
# this result seems to be correct.

# For the fourth question we need to study the correlation between distance and energy used, so we use Spearman
# correlation to do this since the variables are numerical and not normally distributed.

# distance, energy_used, assistance 1

assistances_on = df.loc[df["assistance"] == 1]

print("p-value for the fourth question:",stats.spearmanr(assistances_on["distance"],
                                                         assistances_on["energy_used"],nan_policy='omit')[1])
print("correlation coefficient for the fourth question:",stats.spearmanr(assistances_on["distance"],
                                                         assistances_on["energy_used"],nan_policy='omit')[0])
# The correlation coefficient is close to on and the p-value is less than 0.05 so there is statistical
# evidence that the distance and the energy used correlate with each other when the assistance is enabled.

# For the fifth question we'll use the Chi-squared t-test again, since we want to know if the amount of
# same start and end locations differ between two categroical variables.

same_locations=df.loc[df["location_from"] == df["location_to"]]
different_locations=df.loc[df["location_from"] != df["location_to"]]

savonia_same = same_locations.loc[same_locations.ticket == "savonia"]
savonia_different=different_locations.loc[different_locations.ticket == "savonia"]

season_same = same_locations.loc[same_locations.ticket == "season"]
season_different=different_locations.loc[different_locations.ticket == "season"]

print(season_same["ticket"].value_counts(), "\n",
      season_different["ticket"].value_counts())
print(savonia_same["ticket"].value_counts(),
      "\n",savonia_different["ticket"].value_counts())

same_location_counts = pd.concat({'season' : pd.Series([367,526]), 'savonia' : pd.Series([76,142])},axis=1)

#print(same_location_counts)

print("p-value for the fifth question:",stats.chi2_contingency(same_location_counts)[1])

# The p-value is over 0.05, so there seems to be no statistical evidence to claim that the season
# and savonia tickets differ with respect to how often the trip ends in the same place where
# it started.


# For the final question we need to calculate the speed first, and since it is distance divided by
# time we can use the duration and distance variables to calculate it

speed = df["distance"]/df["duration"]

# To find out if the travel time correlates with the average speed we need to use Spearman correlation 
# again, since neither of the variables can be assumed to be normally distributed

res=pd.concat({'duration': df["duration"], 'speed' : speed},axis=1)


print("p-value for the sixth question:",stats.spearmanr(df["duration"],
                                                         speed,nan_policy='omit')[1])

print("correlation coefficient for the sixth question:",stats.spearmanr(df["duration"],
                                                         speed,nan_policy='omit')[0])
      
# The p-value is very close to zero so it's less than 0.05, so there seems to be statistical evidence that
# the travel time correlates with the average speed at which the trip was made.