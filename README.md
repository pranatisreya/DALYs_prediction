
# PREDICTION OF DISABILITY-ADJUSTED LIFE YEARS DUE TO MENTAL ILLNESS – USING MACHINE LEARNING

# ABSTRACT

Machine learning techniques are applied for the analysis of the impact of mental illness
on the burden of disease. It is calculated using the disability-adjusted life year (DALY).
One DALY represents the loss of the equivalent of one year of full health. DALYs for
a disease or health condition are the sum of years of life lost due to premature mortality
(YLLs) and years of healthy life lost due to disability (YLDs) due to prevalent cases of
the disease or health condition in a population.

DALYs (Disability-Adjusted Life Years) is a measure that quantifies the burden of
disease on a population. It combines both the years of healthy life lost due to premature
death (YLLs) and the years lived with disability (YLDs) into a single summary metric.
The DALYs percentage represents the proportion or share of the overall disease burden
in a population that is attributed to a specific condition or cause. It is often used to
compare and prioritize different diseases or health conditions based on their impact on
population health. For example, if the DALYs percentage for mental illness in a
particular population is 10%, it means that mental illness accounts for 10% of the total
burden of disease in that population. By predicting DALYs associated with mental
illness using machine learning or data-driven approaches, researchers can gain insights
into the impact of mental health conditions on population health and inform public
health interventions, resource allocation, and policy decisions.

The critical analysis is done based on the Data sources, machine learning techniques
and feature extraction method. The reviewing is done based on major databases. The
extracted data is examined using statistical analysis and machine learning techniques
were applied. The prediction of the impact of mental illness on the population using
machine learning techniques is an alternative approach to the old traditional strategies
which are time consuming and may not be reliable.

The obtained prediction is a way of understanding the underlying impact of mental
illness on the health of the people and it enables us to get a healthy life expectancy. The
growing impact of mental illness and the challenges associated with detection and
treatment of mental disorders makes it necessary for us to understand the complete
effect of it on most of the population.



# 1. INTRODUCTION

## 1.1. INTRODUCTION

The changes in the social landscape have contributed to the increase in rate of
mental health problems and psychological disorders. The World Health Organization
(WHO) has defined ‘Mental Health’ as the condition of a person who is able to handle
his/her stress in life according to his/her ability but is still able to work normally and
productively as well as contribute to the society. Factors that affect mental health
probably originate from an individual’s way of life, such as work stress, bad financial
situation, family issues, relationship problems, and violence, along with
environmental factors. These situations can contribute to mental health disorders, such
as depression, anxiety, stress, and various psychological disorders that exert an impact
on the quality of life and holistic well-being of a person.

Approximately 450 million people worldwide are mentally ill, with the disease
accounting for 13% of the global disease burden. WHO estimated that one in four
individuals experiences mental disorders in any stage of their lives. In 2018, WHO
released a guideline on managing the physical conditions of adults with severe mental
health problems. Usually, people will die earlier than the general population if they
had severe mental disorders, such as depression, bipolar disorder (BD), psychotic
disorder, and schizophrenia. In addition, depression, which can lead to suicidal
ideation and suicide attempts, is estimated to affect 350 million people worldwide.

Mental health issues, early detection, accurate diagnosis, and effective
treatment can reduce the impact of it on the everyday life of people. Traditional
machine learning faces common training problems, such as overfitting, model
interpretation, and generalization. This system is designed to be used efficiently on
desktops. This project was started on the premise that there is enough openly available
data and information on the web that can be utilized to build a prediction model that
has access to making accurate prediction of the percentage impact of mental illness
on the Disability-Adjusted Life Years.


## 1.2. APPLICATION

The project on predicting disability-adjusted life years (DALYs) due to mental illness
can have several applications. The project can provide valuable information for public
health planners and policymakers to understand the burden of mental illness and
prioritize interventions and resource allocation accordingly. The project can assist in
allocating healthcare resources effectively. It can help determine where and how
resources should be directed to address the most significant mental health challenges
within a population. By identifying the factors contributing to high DALYs,
interventions can be designed to mitigate the impact and reduce the burden of mental
illness.

The project can enable comparisons between different mental health conditions and
their associated DALYs. This information can aid in understanding the relative impact
and prioritizing interventions for specific mental health disorders. The project's
predictions can serve as a baseline for evaluating the effectiveness of interventions and
mental health programs. By comparing the predicted DALYs with the actual outcomes
after implementing interventions, the success or impact of the interventions can be
assessed. The project's results can be used to raise public awareness about the burden
of mental illness. The data on DALYs can help advocate for increased support,
resources, and policy changes to address mental health issues in society.

Overall, the project's applications extend to informing public health planning, resource
allocation, intervention strategies, comparative analysis, evaluation of interventions,
and raising public awareness regarding mental illness and its impact on population
health.


## 1.3. PROBLEM STATEMENT

```
The burden of mental illness is a significant public health concern, and understanding
the prevalence and impact of specific mental health conditions is crucial for effective
intervention strategies. This project aims to analyze the association between the
prevalence percentages of schizophrenia, bipolar disorder, eating disorders, anxiety,
drug addiction, depression, and alcoholism, as well as the corresponding disability-
adjusted life years (DALYs) percentage. By investigating these relationships across
different countries and years, the objective is to identify patterns, trends, and potential
risk factors associated with mental illness. The findings will inform evidence-based
interventions, resource allocation, and policy recommendations to address the burden
of mental illness on a global scale.
.
```
1. 4. OBJECTIVE

The objective of this project is to develop a robust and accurate predictive model that
estimates the disability-adjusted life years (DALYs) attributed to mental illness.
Leveraging available data on the prevalence percentages of various mental health
conditions, including schizophrenia, bipolar disorder, eating disorders, anxiety, drug
addiction, depression, and alcoholism, the project aims to build a machine learning
model capable of forecasting the associated burden on population health. By utilizing
advanced statistical techniques and predictive algorithms, the model will analyze the
interplay between these mental health conditions and their impact on DALYs.

The goal is to provide policymakers, healthcare professionals, and stakeholders with
valuable insights into the prevalence, trends, and distribution of mental illness burden,
enabling evidence-based decision-making for resource allocation, intervention
planning, and policy formulation. The predictive model's accuracy and reliability will
empower stakeholders to make informed choices and prioritize efforts towards reducing
the overall burden of mental illness, improving mental healthcare accessibility, and
promoting the well-being of individuals and communities.


#2. PROPOSED SYSTEM:

Data Flow Diagram
The data flow diagram for the Disability-Adjusted Life Years Prediction using
machine learning is as follows.

1. Data Collection: Relevant datasets related to mental illness, disability adjusted
    life years (DALYs), and associated factors are collected from reliable sources such
    as healthcare databases, research studies, or public health organizations. These
    datasets may include demographic information, mental health indicators,
    socioeconomic factors, and DALY measurements.
2. Data Pre-processing: The collected data is preprocessed to ensure its quality and
    usability. This involves tasks such as handling missing values, dealing with
    outliers, normalizing or scaling numerical variables, and encoding categorical
    variables if necessary.
3. Feature Engineering: New features are created or derived from the existing
    dataset to enhance the prediction model's performance. These features may include
    aggregating or calculating statistics from multiple variables, incorporating
    temporal trends, or incorporating domain-specific knowledge to capture relevant
    information related to mental health and DALYs.
4. Feature Selection: The most relevant features that contribute significantly to
    predicting DALYs due to mental illness are selected. This step helps to reduce
    dimensionality and improve the model's efficiency. Feature selection techniques
    like statistical analysis, correlation analysis, or feature importance analysis from
    machine learning models are applied.
5. Model Selection: Choose an appropriate machine learning model suitable for
    predicting DALYs due to mental illness. Potential models for this task include
    regression-based models (e.g., linear regression, support vector regression),
    ensemble models (e.g., random forests, gradient boosting), or deep learning
models (e.g., neural networks) depending on the complexity of the problem and
the available data.
```
6. Model Training: Train the selected model using the preprocessed dataset.
    Techniques like cross-validation and hyperparameter tuning are applied to
    optimize the model's performance. The model is trained to predict DALYs based
    on the selected features and their relationships.
7. Model Evaluation: Evaluate the performance of the trained model using
    appropriate evaluation metrics such as mean absolute error (MAE), mean squared
    error (MSE), or R-squared value. This assessment helps determine the accuracy
    and reliability of the model in predicting DALYs due to mental illness.
8. Model Deployment: Deploy the trained model to make predictions on new or
    unseen data. This can be achieved by developing an application, web interface, or
    API that takes relevant input variables (such as demographic information, mental
    health indicators) and provides an output of predicted DALYs due to mental
    illness. The deployed model can be utilized by healthcare professionals,
    policymakers, or researchers to gain insights into the burden of mental illness and
    inform decision-making processes.

By following this data flow diagram, a machine learning-based system has been
developed to predict disability adjusted life years (DALYs) associated with mental
illness. This can contribute to understanding the impact of mental health conditions
on overall well-being and assist in the development of targeted interventions and
policies.



# 4. SYSTEM REQUIREMENTS

## 4 .1. HARDWARE REQUIREMENTS

The Hardware requirements for the DALYs prediction project:

1. Operating System: The project requires an operating system that supports data
    analysis and machine learning tools. Recommended options include Windows, Linux,
    or macOS.
2. Integrated Development Environment (IDE): It is advisable to use an IDE to
    facilitate coding, debugging, and execution. Recommended Python IDEs for the
    project include PyCharm, Spyder, or Jupyter Notebooks. Google Colab has been used
    for the Machine Learning Model and VS Code to implement the rest of the project.
3. Data Analysis and Visualization Tools: Tools for data manipulation,
    exploration, and visualization are essential. Consider using libraries like Pandas,
    Matplotlib, and Seaborn for efficient data analysis and visualization tasks.
4. Machine Learning Framework: Utilize a machine learning framework to build,
    train, and evaluate models for DALYs prediction. TensorFlow, PyTorch, or Scikit-
    learn are popular choices for implementing machine learning algorithms.
5. Database Management System (DBMS): Depending on the project's data
    storage and retrieval needs, a suitable DBMS such as MySQL, PostgreSQL, or
    MongoDB may be employed for efficient data management.

Additionally, considering the specific project requirements related to the prediction of
DALYs due to mental illness using machine learning, you may need to incorporate
technologies such as Flask for web development, HTML and CSS for user interface
design, and other relevant technologies.


## 4.2. SOFTWARE REQUIREMENTS

```
4.2.1. Python
Python is a popular and versatile programming language widely used in the field of
machine learning. It offers a simple and readable syntax, making it easy for developers
to write and maintain code. Python provides extensive libraries and frameworks
specifically designed for machine learning tasks, which greatly simplify the
development process.
```
```
Here are some key aspects of Python's use in machine learning:
```
1. Simplicity: Python's clean and intuitive syntax makes it beginner-friendly and
    enables rapid development. It emphasizes code readability, which is essential for
    machine learning projects that involve complex algorithms and data manipulation.
2. Abundant Libraries: Python offers numerous libraries tailored for machine
    learning tasks, such as NumPy, Pandas, and Matplotlib. These libraries provide
    efficient data structures, array operations, statistical analysis, and data
    visualization capabilities, enabling researchers and developers to work with data
    effectively.
3. Machine Learning Frameworks: Python is the primary language for many popular
    machine learning frameworks, including TensorFlow, PyTorch, and scikit-learn.
    These frameworks provide pre-built tools and functionalities for building, training,
    and evaluating machine learning models. They also offer a wide range of
    algorithms, allowing users to implement various techniques with ease.
4. Data Manipulation: Python's libraries, such as NumPy and Pandas, offer powerful
    tools for data manipulation and preprocessing. These libraries enable users to
    handle large datasets efficiently, perform data cleaning, handle missing values,
    and transform data into suitable formats for machine learning algorithms.
5. Visualization Capabilities: Python provides libraries like Matplotlib and Seaborn,
    which facilitate data visualization. Visualizing data is crucial for understanding
    patterns, trends, and relationships, helping machine learning practitioners gain
    insights and make informed decisions.
6. Easy Integration: Python seamlessly integrates with other programming
    languages, making it ideal for building end-to-end machine learning pipelines. For


```
example, you can use Python to preprocess data, train models, and then integrate
them into web applications or production systems using frameworks like Flask or
Django.
```
7. Supportive Community: Python has a vibrant and supportive community of
    developers and researchers who contribute to open-source projects. This
    community-driven ecosystem ensures continuous improvement and offers
    extensive resources, tutorials, and libraries for machine learning practitioners.

Overall, Python's simplicity, rich libraries, machine learning frameworks, and strong
community support make it an excellent choice for machine learning tasks. Its
versatility and ease of use have contributed to its widespread adoption in the field.

Any basic computer or laptop with python latest version installed in the system can
have this program implemented in.

```
Language/s Used: Python, html, css
```
```
Python version
(Recommended):
```
## 3.11.

```
Type: Desktop Application
```
The packages inbuilt in the python that are utilized in the code are.

```
NumPy is a Python library for scientific computing and data analysis. It offers
efficient tools for working with large arrays and matrices, including mathematical
functions, linear algebra operations, and statistical analysis. NumPy's optimized code
enables fast computation, making it ideal for scientific applications.
```
```
Pandas is an open-source library used for data manipulation and analysis. It provides
flexible data structures (DataFrames and Series) and features for cleaning,
```

```
transforming, and analyzing structured data. Pandas integrates well with other Python
libraries and is widely used in data science for its ease of use and powerful tools.
```
```
Seaborn is a data visualization library built on Matplotlib. It offers a high-level
interface for creating attractive statistical graphics. Seaborn provides various
visualization functions, making it easy to explore relationships between variables and
visualize distributions and summaries. It simplifies the creation of visually appealing
plots with minimal code.
```
```
Matplotlib.pyplot is a module within the Matplotlib library that enables the creation
of static data visualizations. It provides a simple interface for creating different types
of plots, such as line plots, scatter plots, and histograms. Matplotlib.pyplot is widely
used for its ease of use, flexibility, and customization options.
```
```
Scikit-learn (sklearn) is a popular Python library for machine learning tasks. It offers
a wide range of tools for data preprocessing, classification, regression, clustering, and
model selection. Scikit-learn provides an intuitive API, making it accessible to
beginners while offering advanced features like hyperparameter tuning and model
evaluation.
```
In summary, these libraries play vital roles in Python's scientific ecosystem, enabling
efficient array processing, data manipulation, visualization, and machine learning
tasks.

4.2.2. Flask
Flask is a lightweight web framework for Python that is commonly used for building
web applications, including those related to machine learning projects. It provides a
simple and flexible approach to developing web services and APIs.

Flask can be utilized in Python for machine learning projects in the following ways:

1. Serving Machine Learning Models: Flask can be used to deploy machine learning
    models as web services or APIs. By integrating Flask with a machine learning


```
framework like scikit-learn or TensorFlow, you can build a web application that
allows users to interact with the model through an API. Users can send requests to
the server, and the server can return predictions generated by the model.
```
2. Data Collection and Preprocessing: Flask can be used to create web-based
    interfaces for data collection or preprocessing tasks in machine learning projects.
    It enables users to input data through HTML forms or APIs, which can then be
    processed and used in the machine learning pipeline. Flask makes it easy to handle
    user input, validate data, and perform necessary preprocessing steps before feeding
    it into the model.
3. Building Dashboards and Visualizations: Flask can be used to create interactive
    dashboards and visualizations for machine learning projects. By integrating Flask
    with data visualization libraries like Matplotlib or Plotly, you can build web-based
    interfaces that display real-time insights, predictions, or performance metrics
    generated by machine learning models. Flask allows you to render dynamic web
    pages and update the content based on user interactions.
4. Experiment Tracking and Logging: Flask can be used to develop web-based
    interfaces for tracking and logging machine learning experiments. By integrating
    Flask with logging frameworks like TensorBoard or MLflow, you can create a
    web application that visualizes experiment results, tracks model performance over
    time, and provides insights into the training process. Flask enables you to display
    experiment metrics, visualize model architectures, and compare different runs
    conveniently.
5. Creating Annotation Tools: Flask can be used to build web-based annotation tools
    for machine learning projects that require labelled data. These tools allow users to
    annotate or label data directly through a web interface. Flask enables you to handle
    user interactions, store annotated data, and integrate it seamlessly into the machine
    learning workflow.

Flask's flexibility, simplicity, and rich ecosystem of extensions make it an excellent
choice for integrating machine learning models and building web-based components
in Python projects.


4.2.3. Google Colab:
Google Colab is a cloud-based integrated development environment (IDE) provided
by Google. It allows users to write and execute Python code in a web browser,
eliminating the need for local installations or configurations. Colab provides access to
powerful hardware resources, including GPUs and TPUs, making it suitable for
machine learning projects. With Colab, you can create and run Jupyter notebooks,
which are interactive documents that combine code, text, and visualizations. Colab
notebooks are often used in machine learning projects for data analysis, model
training, and experimentation.

4.2.4. HTML (Hypertext Markup Language):
HTML is the standard markup language used to create the structure and content of
web pages. It provides a set of tags and elements that define the layout, headings,
paragraphs, images, links, and other components of a web page. In the context of a
machine learning project on DALYs, HTML can be used to create a web-based user
interface for data input, displaying predictions, or visualizing results. For example,
HTML forms can be used to collect user input regarding relevant factors influencing
DALYs, and the collected data can be processed and analysed using machine learning
algorithms.

4.2.5. CSS (Cascading Style Sheets):
CSS is a style sheet language used to describe the presentation and appearance of
HTML documents. It allows you to define the layout, colours, fonts, and other visual
aspects of web pages. In the context of a machine learning project, CSS can be used
to style the HTML elements and create visually appealing and user-friendly interfaces.
CSS enables customization and improves the overall look and feel of the web-based
application or dashboard developed for the DALYs project. It helps in creating a
consistent design and enhances the user experience by making the application visually
engaging and intuitive.


4.2. 6. Machine Learning Classifiers
Machine learning classifiers have been trained and tested using training and test data
to select the best performing classifier. The sklearn library was used to import these
classifiers. The models were trained on X_train and y_train data and tested on X_test
data to predict rainy days.

Logistic Regression:
Logistic regression is a classification technique that explains the relationship between
a dependent binary variable and independent variables. It assumes a binary or
dichotomous dependent variable and requires the absence of outliers and strong
correlations among predictors.

Random Forest:
Random forest is a supervised machine learning algorithm that combines multiple
decision trees to achieve accurate classification. It generates many decision trees and
aggregates their results. Random forest can be used for both classification and
regression tasks. In random forest, decision trees are built using subsets of the training
data (bootstrap datasets), and the final model is created by considering the output of
multiple decision trees. This model is applied to the testing dataset to obtain accurate
predictions.

These classifiers are applied to machine learning projects for various tasks, such as
predicting rainy days based on the provided feature set. By training on labelled data
and utilizing the selected classifier, accurate predictions can be made for unseen data.

In the machine learning project on DALYs, Google Colab can be utilized as the
development environment to write and execute Python code for data preprocessing,
model training, and evaluation. HTML can be employed to develop web-based user
interfaces for data collection, input, and result visualization. CSS can be used to style
the HTML elements, ensuring a visually pleasing and user-friendly interface. These
technologies combined can help in creating an interactive and accessible application
for analysing DALYs and making predictions based on machine learning models.


## 4.3. FUNCTIONAL REQUIREMENTS

```
The functional requirements for the prediction of DALYs (Disability-Adjusted Life
Years) due to mental illness in an ML project could include:
```
1. Data Collection and Preprocessing: The system requires collection relevant data on
    mental health conditions, prevalence percentages, and associated DALYs from
    reliable sources. It preprocesses the data, including cleaning, normalization, and
    feature extraction, to prepare it for analysis and modelling.
2. Model Training and Evaluation: The system trains a machine learning model using
    the pre-processed data to predict DALYs associated with mental illness. It utilizes
    appropriate ML algorithms, such as regression or classification models, to achieve
    accurate predictions. The system evaluates the trained model using suitable metrics,
    such as mean squared error or accuracy, to assess its performance.
3. Feature Selection and Importance: The system determines the key features or
    variables that contribute significantly to the prediction of DALYs. It provides
    insights into the relative importance of different mental health conditions in relation
    to DALYs.
4. Prediction and Reporting: The system should accept input data on mental health
    conditions and provide predictions for the corresponding DALYs. It generates
    comprehensive reports or visualizations summarizing the predictions and
    associated uncertainties. The system presents the results in an easily interpretable
    format for users and stakeholders.

These functional requirements outline the necessary capabilities and features for the
prediction of DALYs due to mental illness using machine learning. They guide the
development of the ML project to ensure accurate predictions, insightful reporting, and
usability for users and stakeholders.


# 5. IMPLEMENTATION

Importing datasets

```
Fig. 5.1. importing dataset
```
```
Fig. 5.2. dataset
```
```
Fig. 5.3. merging datasets
```

```
Fig. 5.4. Dataset after feature extraction
```
Machine learning models can only work with numerical values. Feature Encoding is
process of transformation of the categorical values of the relevant features into
numerical ones.

```
Fig. 5.5. feature encoding and splitting of data.
```
The dataset is split to take 80 percent of the values to train the dataset and 20 percent
to test the dataset. After training the model on the training set we test the model on the
testing set. After the model performed well on the testing set, based on the confidence
it can be used for prediction.


```
Fig. 5.6. splitting data to training and testing the model
```
Linear Regression

```
Fig. 5.7. implementing linear regressor classifier
```
Random Forest Regressor

```
Fig. 5.8. implementing Random Forest Regressor Classifier
```

Taking input for the model prediction

```
Fig. 5.9. taking input for model prediction
```
Converting to pickle file

# Fig. 5.10. getting the ML model as pickle file


# FLASK

# HTML FILE


# 6. TESTING AND RESULTS

The results of the code implementation are as follows.

```
Fig. 6.1. user interface
```
```
Fig. 6.2. user input values
```

Fig. 6.3. input values being given to model by entering _‘_ predict _’_

```
Fig. 6.4. final output of predicted percentage
```

# 6. CONCLUSION AND FUTURE SCOPE

## 7.1. CONCLUSION

In conclusion, the project on predicting Disability Adjusted Life Years (DALYs) due
to mental illness using machine learning has shown promising results. By leveraging
relevant datasets and applying preprocessing techniques, feature engineering, and
model training, we have developed a model capable of predicting DALYs associated
with mental illness.

The selected machine learning model, along with appropriate feature selection, has
demonstrated good performance in predicting DALYs. Evaluation metrics such as
mean absolute error (MAE), mean squared error (MSE), and R-squared value have
indicated the accuracy and reliability of the model in capturing the impact of mental
illness on overall well-being.

The project's outcomes hold potential in several domains. Healthcare professionals
can utilize the model to gain insights into the burden of mental illness and allocate
resources effectively. Policymakers can leverage predictions to develop targeted
interventions and policies addressing mental health concerns. Researchers can further
explore the relationships between mental health indicators and DALYs to enhance
understanding and improve mental health outcomes.


## 7.2. FUTURE SCOPE

The prediction of DALYs due to mental illness using machine learning presents
opportunities for further research and improvements. Future exploration can focus on
dataset expansion, incorporating diverse and longitudinal data to enhance the model's
performance and understanding of factors influencing DALYs. Advanced machine
learning techniques, such as deep learning models and ensemble methods, may offer
better predictions and insights into the complex relationship between mental health
and DALYs.

Continuous refinement and expansion of feature engineering can capture the nuances
of mental illness's impact on DALYs. Developing real-time prediction systems and
decision support tools can aid healthcare professionals in resource allocation,
treatment planning, and early intervention strategies. By enhancing model
interpretability through feature importance analysis and model visualization,
stakeholders can better comprehend the factors influencing DALYs and trust the
predictions. Addressing these areas can lead to significant contributions to mental
health research, healthcare delivery, and policy development.


# 8. BIBLIOGRAPHY

- Disability-adjusted life years (DALYs) (who.int)
- Mental disorders (who.int)
- Mental health (who.int)
- VizHub - GBD Results (healthdata.org)
- What is Machine Learning? | IBM


