# Educational Projects in Jupyter Notebooks
**This repository contains educational projects that were conducted based on datasets of CodeCademy and its partners, NLTK Toolkit Corpus, Kaggle, UCI’s Machine Learning Repository, Scikit-Learn example datasets. The purpose of each is to practice individual studying modules and consolidate the result as an active professional skill.**

## Brief projects' description (ordered by completion):

### 34. SQL (Windows Functions): ["Climate Change"](./sql/windows_functions/climate_change.ipynb)

The goal of this project is to practice SQL windows functions, ascertaining different climate change insights within "Global Land and Ocean Temperatures" dataset in the process.

----------------------

### 33. Unsupervised Learning (K-Means++): ["Handwriting Recognition"](./ml/unsupervised_learning/k_means/k_means_handwriting_rec.ipynb)

In this project, we will be using **K-Means++** clustering algorithm in scikit-learn inplementation on sklearn digits dataset to cluster images of handwritten digits.

----------------------

### 32. Supervised Machine Learning (Ensemble Technique - Random Forest): ["Predicting Income"](./ml/supervised_learning/random_forest/forest_income_project.ipynb)

In this project, we will be using an ensemble machine learning technique - **Random Forest**. We are going to compare its performance with **Dicision Tree** algorithm, on a dataset containing [census information from UCI’s Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Adult/). By using census data with a Random Forest and Dicision Tree, we will try to predict whether a person makes more than $50,000.

----------------------

### 31. Supervised Machine Learning (Decision Trees): ["Predict Continent and Language"](./ml/supervised_learning/trees/trees_find_flag.ipynb)

In this project, we’ll use **Decision Trees** to predict the continent a country is located on, and its language based on features of its flag and other country's properties. For instance, some colors are good indicators as well as the presence or absence of certain shapes could give one a hint. We’ll explore which features are the best to use and will create several Decision Trees to compare their results. The [**Flags Data Set**](https://archive.ics.uci.edu/ml/datasets/Flags) used in this project is provided by UCI’s Machine Learning Repository.

----------------------

### 30. Supervised Machine Learning (Support Vector Machines): ["Predict Baseball Strike Zones"](./ml/supervised_learning/svm_models/baseball_strike_zones.ipynb)

In this project, we will use a **Support Vector Machines**, trained using a [`pybaseball`](https://github.com/jldbc/pybaseball) dataset, to find strike zones decision boundary of players with different physical parameters.

----------------------

### 29. Supervised Machine Learning (Naive Bayes): ["Newsgroups Similarity"](./ml/supervised_learning/naive_bayes/newsgroups_simularity.ipynb)

In this project we will apply Scikit-learn’s **Multinomial Naive Bayes Classifier** to Scikit-learn’s example datasets to find which category combinations are harder for it to distinguish. We are going to achieve that by reporting the accuracy of several variations of the classifier that were fit on different categories of newsgroups.

----------------------

### 28. Supervised Machine Learning (Logistic Regression): ["Chances of Survival on Titanic"](./ml/supervised_learning/logistic_regression/logistic_regression_titanic.ipynb)

The goal of this project is to create a **Logistic Regression Model** that predicts which passengers survived the sinking of the Titanic, based on features like age, class and other relevant parameters.

-----------------------

### 27. Supervised Machine Learning (K-Nearest Neighbor classifier and others): ["Twitter Classification"](./ml/supervised_learning/k_nearest_neighbours/twitter_classification.ipynb)

The first goal of this project is to predict whether a tweet will go viral using a **K-Nearest Neighbour Classifier**. The second is to determine whether a tweet was sent from New York, London, or Paris using Logistic Regression and Naive Bayes classifiers with vectorization of different sparsity levels applied to their features. To even out gaps between number of tweets in different cities we'll apply text augmentation by BERT.

-----------------------

### 26. Supervised Machine Learning (Multiple Linear Regression): ["Tendencies of Housing Prices in NYC"](./ml/supervised_learning/linear_regression/multiple_linear_regression/prices_tendencies_for_housing.ipynb)

The goal of this project is to get an insight to the range of factors that influence NYC apartments' price formation, predict prices for several random apartments based on these insights with the help of **Multiple Linear Regression Model** and visualise some results using 2D and 3D graphs.

-----------------------

### 25. Supervised Machine Learning (Simple Linear Regression): ["Honey Production"](./ml/supervised_learning/linear_regression/simple_linear_regression/honey_production.ipynb)

The goal of this project is predicting honey production during upcoming years (till 2050) using **Simple Linear Regression Model** and some visualizations.

-----------------------

### 24. NLP, Feature Modeling (tf-idf): ["News Analysis"](./nlp/language_quantification/news_analysis/language_quantification.ipynb)

In this project we will use **"term frequency-inverse document frequency"** (tf-idf) to analyze each article’s content and uncover the terms that best describe each article, providing quick insight into each article’s topic.

-----------------------

### 23. NLP, Feature Modeling (Word Embeddings): ["U.S.A. Presidential Vocabulary"](./nlp/language_quantification/presidential_vocabulary/us_presidential_vocabulary.ipynb)

The goal of this project is to analyze the inaugural addresses of the presidents of the United States of America using **word embeddings**. By training sets of word embeddings on subsets of inaugural addresses, we can learn about the different ways in which the presidents use language to convey their agenda.

-----------------------

### 22. NLP, Topics Analysis (Chunking): ["Discover Insights into Classic Texts"](./nlp/language_parsing/language_parsing.ipynb)

The goal of this project is to discover the main themes and some other details from the two classic novels: Oscar Wilde’s **"The Picture of Dorian Gray"** and Homer’s **"The Iliad"**.  To achieve it, we are going to use `nltk` methods for preprocessing and creating Tree Data Structures, after which, we will apply filters to those Structures to get some desired insights.

-----------------------

### 21. NLP (Sentiment Analysis), Supervised Machine Learning (Ensemble Technique): ["Voting System"](./nlp/voting_system/nltk_scikitlearn_combined.ipynb)

The goal of this project is to create a voting system for *bivariant sentiment analysis* of any type of short reviews. To achieve this we are going to combine **Naive Bayes** algorithm from `nltk` and similar algorithms from `scikit-learn`. This combination should increase the accuracy and reliability of the confidence percentages.

-----------------------

### 20. NLP, Sentiment Analysis (Naive Bayes Classifier): ["Simple Naive Bayes Classifier"](./nlp/naive_bayes_classifier.ipynb)

The goal of this project is to build a simple **Naive Bayes Classifier** using `nltk toolkit`, and after that: train and test it on Movie Reviews corpora from `nltk.corpus`.

-----------------------

### 19. Analysis via SQL: ["Gaming Trends on Twitch"](./sql/twitch_data_extraction_and_visualization/twitch_data_extraction_and_visualisation.ipynb)

The goal of this project is to analyse gaming trends with SQL and visualise them with Matplotlib and Seaborn. 

-----------------------

 ### 18. Statistical Analysis and Visualisation: ["Airline Analysis"](./statistics_and_visualization/airline_analysis/airline_analysis.ipynb)

The goal of this project is to guide airline company clients' decisions by providing summary statistics that focus on one, two or several variables and visualise its results. 

-----------------------

### 17. Statistical Analysis and Visualisation: ["NBA Trends"](/statistics_and_visualization/associations_nba/nba_trends.ipynb)

In this project, we’ll analyze and visualise data from the NBA (National Basketball Association) and explore possible associations.

-----------------------

### 16. Statistical Analysis and Visualisation: ["Titanic"](/statistics_and_visualization/quant_and_categorical_var/titanic.ipynb)

The goal of this project is to investigate whether there are some correlations between the different aspects of physical and financial parameters and the survival rates of the Titanic passengers.

-----------------------

### 15. Data Transformation: ["Census Data Types Transformation"](./statistics_and_visualization/census_datatypes_transform/census_datatypes_transform.ipynb)

The goal of this project is to use pandas to clean, organise and prepare recently collected census data for further usage by machine learning algorithms.

-----------------------

### 14. Data Transformation: ["Company Transformation"](./statistics_and_visualization/data_transformation/company_transformation.ipynb)

The goal of this project is to apply data transformation techniques to better understand the company’s data and help to answer important questions from the management team.

-----------------------

### 13. Explorative Data Analysis (EDA): ["Diagnosing Diabetes"](./statistics_and_visualization/eda_diagnosing_diabetes/eda.ipynb)

The goal of this project is to explore data that looks at how certain diagnostic factors affect the diabetes outcome of women patients. We are going to use EDA skills to help inspect, clean, and validate the data.

-----------------------

### 12. Visualisaton using boxplot: ["Healthcare in Different States"](./statistics_and_visualization/healthcare_in_different_states/healthcare_in_different_states.ipynb)

In this project, we will use boxplots to investigate the way hospitals in different states across the United States charge their patients for medical procedures.

-----------------------

### 11. Visualisation using histogram: ["Traveling to Acadia"](./statistics_and_visualization/traveling_to_acadia/traveling_to_acadia.ipynb)

The goal of this project is to inform customers of a travel agency about the best time to visit Acadia to see flowers in bloom. We will use flower bloom data, and flight information to create visualisation and recommend the best time of year for someone to make a trip to Maine.

-----------------------

### 10. Analysis using statistical instruments: ["Life expectancy by country"](./statistics_and_visualization/life_expectancy_by_country/life_expectancy_by_country.ipynb)

The goal of this project is to investigate a dataset containing information about the average life expectancy in 158 different countries and, using statistical instruments, analyse an example country's location on a GDP scale based on the average life expectancy of this country.

-----------------------

### 9. Data wrangling, tidying and visualisation: ["Cleaning US Census Data"](./statistics_and_visualization/us_census_data/cleaning_us_census_data.ipynb)

The goal of this project is to make US Census data scalable and repeatable, converting data from multiple CSV files to pandas' DataFrames, as well as to collect some interesting insights from it and create informative visualisations as a result. 

-----------------------


### 8. WebScraping with Beautiful Soup: ["Chocolate bars"](./web_scraping/chocolate_scraping.ipynb)

The goal of this project is to analyze and manipulate data displayed in a table on a website: `https://content.codecademy.com/courses/beautifulsoup/cacao/index.html`, using BeautifulSoup and pandas that should help us transform this webpage into a DataFrame.

-----------------------

### 7. Visualisation using Seaborn: ["Visualizing World Cup Data"](./statistics_and_visualization/visualizing_world_cup_data/visualizing_fifa_data.ipynb)

The goal of this project is using Seaborn explore data from the Fifa World Cup from 1930-2014 to analyze trends and discover insights about football game.

-----------------------

### 6. Building Funnel via pandas: ["Page Visits"](./statistics_and_visualization/page_visits_funnel/page_visits_funnel.ipynb)

The goal of this project is to analyze data on visits to a website of Cool T-Shirts Inc by building a funnel via pandas.

-----------------------

### 5. Analysis via pandas: ["A/B Testing for ShoeFly.com"](./statistics_and_visualization/ab_testing_for_shoe_shop/ab_testing_for_shoe_shop.ipynb)

The goal of this project is to analyze the data, using Pandas aggregate measures, for an online shoe store ShoeFly.com that is performing an A/B testing.

-----------------------

### 4. Analysis via pandas: ["Petal Power Store Inventory"](./statistics_and_visualization/petal_power_inventory/petal_power_inventory.ipynb)

The goal of this project is to help to analyze a shop-inventory, using pandas, for a chain of gardening stores called Petal Power.

-----------------------

### 3. Visualisation using Matplotlib: ["Graphs and Charts for Teacher"](./statistics_and_visualization/graphs_for_teacher.ipynb)

The goal of this project is to create some graphs to help a high school math teacher, using Matplotlib and a bit of NumPy.

-----------------------
    
### 2. Python functions: ["Linear Regression"](./statistics_and_visualization/linear_regression.ipynb)

The goal is to create a function that will find a line of best fit on the Linear Regression graph when given a set of data. Modules to be used: loops, lists as well as basic arithmetic. 
    
-----------------------

### 1. Python Lists: ["Frida Calo"](./statistics_and_visualization/frida_calo.ipynb)
    
Project for the Lists module practicing. The task is to make simple lists of the artist's work for a potential digital guide. The project is __not a complete application__, but only demonstrates the skills of applying lists within Jupyter Notebook. 
    
    



