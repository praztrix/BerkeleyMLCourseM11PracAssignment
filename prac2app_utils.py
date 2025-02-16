
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns



def analyze_outliers(df, feature, outlier_threshold=1.5):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    # identify outliers
    outliers = df[(df[feature] < Q1 - outlier_threshold * IQR) | (df[feature] > Q3 + outlier_threshold * IQR)]
    print(" Outlier Row Count:", outliers.shape[0], " Outlier Row Percent:", 100*(outliers.shape[0]/df.shape[0]),"%")
    print(" Outlier Max:", outliers[feature].max(), 
          " Outlier Median:", outliers[feature].median(), 
          " Outlier Mean:", outliers[feature].mean(), 
          " Outlier Min:",outliers[feature].min())
    return outliers

def subplot_box_hist(df, target, box_title, hist_title):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
    axes[0].boxplot(df[target])
    axes[1].hist(df[target], bins=20)
    axes[0].set_title(box_title)
    axes[0].set_ylabel(target)
    axes[1].set_title(hist_title)
    axes[1].set_xlabel(target)
    axes[1].set_ylabel('count')
    plt.show()

def subplot_box_hist_scatter(df, feature, target, box_title, hist_title, scatter_title):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))
    axes[0].boxplot(df[feature])
    axes[1].hist(df[feature])
    axes[2].scatter(df[feature], df[target])
    axes[0].set_title(box_title)
    axes[0].set_ylabel(feature)
    axes[1].set_title(hist_title)
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel('count')
    axes[2].set_title(scatter_title)
    axes[2].set_xlabel(feature)
    axes[2].set_ylabel(target)
    plt.show()
    

#value count plot with count and percentage labels.
def pretty_value_count_plot(df, col, plt_title=""):
    ax = sns.countplot(x=df[col], order = df[col].value_counts(ascending=False).index)
    if(len(plt_title) > 0):
        ax.set_title(plt_title)
    df_y_count = df[col].value_counts(ascending=False)
    df_y_pcnt  = df[col].value_counts(ascending=False, normalize=True).values*100
    df_y_lbls = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(df_y_count, df_y_pcnt)]
    ax.bar_label(container=ax.containers[0], labels=df_y_lbls)
    plt.show()




def do_cat_cont_feature_independence_anova_test(df, feature, target):
    # Ref: https://www.youtube.com/watch?v=u3Hwt_jbbTc
    #Anova test
    grouped_values = []
    #retrieve unique values for the categorical feature.
    cond_groups = df[feature].unique()
    # for each category of the categorical feature, create a list of target variables and append it to grouped values
    # df[df[feature]==group][target] gives you a list of target variables for a specific group/category
    for group in cond_groups:
        grouped_values.append(df[df[feature]==group][target])
    s, pvalue = stats.f_oneway(*grouped_values)
    #NULL Hypothesis is feature is independent
    if(pvalue < 0.05):
        #reject the null hypothesis
        print(feature, " is likely dependent")
    else:
        print(feature, " is likely independent")
