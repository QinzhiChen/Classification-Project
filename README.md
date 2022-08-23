Project Summary
 
Project Objectives
•	Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter Notebook report.
•	Create modules (acquire.py, prepare.py) that make your process repeateable.
•	Construct a model to predict customer churn using classification techniques.
•	Deliver a 5 minute presentation consisting of a high-level notebook walkthrough using your Jupyter Notebook from above; your presentation should be appropriate for your target audience.
•	Answer panel questions about your code, process, findings and key takeaways, and model.
Business Goals
•	Construct a ML classification model that accurately predicts Iris species.
•	Document your process well enough to be presented or read like a report.
Audience
•	Your target audience for your notebook walkthrough is your direct manager and their manager. This should guide your language and level of explanations in your walkthrough.
Project Deliverables
•	README.MD
•	A final report notebook
•	Acquire and Prepare Module
•	A predictions
•	A final report notebook presentation
•	All necessary modules to make my project reproducible
Project Context
•	The  dataset I'm using came from the Codeup database.
•	Find out more about Fisher's Iris Dataset here.
Data Dictionary
•	There are 29 columns, the main target column will be ‘churn_Yes’. It is an object datatype with two unique variable: Yes or No.
•	#   Column                                 Non-Null Count  Dtype  
•	---  ------                                 --------------  -----  
•	 0   customer_id                            4225 non-null   object 
•	 1   senior_citizen                         4225 non-null   int64  
•	 2   tenure                                 4225 non-null   int64  
•	 3   monthly_charges                        4225 non-null   float64
•	 4   total_charges                          4225 non-null   float64
•	 5   contract_type                          4225 non-null   object 
•	 6   signup_date                            4225 non-null   object 
•	 7   partner_Yes                            4225 non-null   uint8  
•	 8   dependents_Yes                         4225 non-null   uint8  
•	 9   phone_service_Yes                      4225 non-null   uint8  
•	 10  multiple_lines_No phone service        4225 non-null   uint8  
•	 11  multiple_lines_Yes                     4225 non-null   uint8  
•	 12  online_security_No internet service    4225 non-null   uint8  
•	 13  online_security_Yes                    4225 non-null   uint8  
•	 14  online_backup_No internet service      4225 non-null   uint8  
•	 15  online_backup_Yes                      4225 non-null   uint8  
•	 16  device_protection_No internet service  4225 non-null   uint8  
•	 17  device_protection_Yes                  4225 non-null   uint8  
•	 18  tech_support_No internet service       4225 non-null   uint8  
•	 19  tech_support_Yes                       4225 non-null   uint8  
•	 20  streaming_tv_No internet service       4225 non-null   uint8  
•	 21  streaming_tv_Yes                       4225 non-null   uint8  
•	 22  streaming_movies_No internet service   4225 non-null   uint8  
•	 23  streaming_movies_Yes                   4225 non-null   uint8  
•	 24  paperless_billing_Yes                  4225 non-null   uint8  
•	 25  churn_Yes                              4225 non-null   uint8  
•	 26  internet_service_type_Fiber optic      4225 non-null   uint8  
•	 27  internet_service_type_None             4225 non-null   uint8  
•	 28  gender_Male                            4225 non-null   uint8  

Initial Hypotheses
•	Hypothesis 1 -
•	alpha = .05
•	H0= The monthly charges has no impact on whether to churn
•	Ha= The monthly charges has significant impact on whether to churn
•	Outcome: we rejected the Null Hypothesis.
•	Hypothesis 2 -
•	alpha = .05
•	H0=The signup month has no significant difference on the churn rate
•	Ha= The signup month has significant difference on the churn rate
•	Outcome:  we rejected the Null Hypothesis.
 
Executive Summary - Conclusions & Next Steps
 
    - All of the classification models [LogisticRegression, DecisionTree, RandomForest, and KNeighbors predicted] has similar accuracy (around 76%-82%)
    - The random forest model was used due to it has the best outcome with 81% accuracy in train dataset.
    - The exploration revealed that people are likely to sign up the service in December with month to month contract, and after one month, 61% of them will churn.
    - The exploration also revealed that the next steps we should implement is to gather more data on what is the motive those people decided to sign up in December and how many features they have included in their service.
    - The exploration revealed that customers are like to churn in December when they have less than 4 features. 
Pipeline Stages Breakdown
 
Plan
    - Create .ignore file and setting up the environment to ensure no sensitive information is publish to the    public.
    - Create two .py files and create some functions that contain acquire, prepare, and split the data.
    - Create initial notebook to test out and explore data. Create some functions in that notebook to import the work to the final notebook.
    - Create a README.md with all information required.
    - In final notebook, we will import those best models.
    - Create a final csv file for the test prediction.
    - Document the conclusion, takeaway, and next steps in the Final notebook. 
Plan -> Acquire
    -  Using MySql to join tables and ensure all data are collected.
    -  Create a function in acquire.py, and change it to pandas DataFrame.
    - Import function to final notebook.
    - Initial data summary 
 
Plan -> Acquire -> Prepare
    - Create a prepare.py file and create functions in that file.
    - Import the prepare function from the prepare.py module and use it to prepare the data in the Final Report Notebook.
 
Plan -> Acquire -> Prepare -> Explore
    -  Ask some questions that might be a keyway to find out why customers churn.
    -  Have two statistical tests in the data exploration.
    -  Create visualizations and discovers some relationships.
    -  Notes some key takeaways while exploring
    -  Summarize the conclusion and provide some recommendations.
 
Plan -> Acquire -> Prepare -> Explore -> Model
    - define the baseline and calculate its accuracy.
    - Train the models
    - compare those models and find out the best model to use
    - Evaluate those models with validate datasets and compare the accuracy.
    - Test the best model
 
Plan -> Acquire -> Prepare -> Explore -> Model -> Deliver
    - Summarize findings at the beginning of the final notebook
    - Answer all questions ask, and provide a summary about why customers might choose to churn
 
Files Needed to Reproduce This Project
 
•	 Read this README.md
•	 Download the aquire.py, prepare.py, report_before_the_report.py, exp_mod.py and final_report.ipynb files into your working directory
•	 Have your own env file to your directory. (user, password, host)
•	 Run the final_report.ipynb notebook
