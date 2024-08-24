from pycaret.classification import *
import pandas as pd

# loading data
data = pd.read_csv("./data.csv")

# setting up the environement for classification
clf = setup(
    data=data,
    target='GradeClass', 
    ignore_features=['StudentID'],  
    categorical_features=['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering'],  # List of categorical features
    numeric_features=['Age', 'StudyTimeWeekly', 'Absences', 'GPA'],  
    session_id=123  
)

# comparing between all models
best_model = compare_models()

# saving the best model
save_model(best_model,"student_performance_model")
print("the model was saved succesfully")
