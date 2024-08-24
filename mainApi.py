from fastapi import FastAPI, Query
from pycaret.classification import load_model, predict_model
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins={"*"},  # Allows specific origins
    allow_credentials=True,  # Allows cookies to be included in requests
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

model = load_model("./student_performance_model")
@app.get("/predict/")
async def predict(
    Age: int = Query(..., description="The age of the student"),
    Gender: int = Query(..., description="The gender of the student (0: Female, 1: Male)"),
    Ethnicity: int = Query(..., description="Ethnicity of the student (e.g., 0, 1, 2, etc.)"),
    ParentalEducation: int = Query(..., description="Parental education level (e.g., 0: none, 1: high school, etc.)"),
    StudyTimeWeekly: float = Query(..., description="Average study time per week in hours"),
    Absences: int = Query(..., description="Number of absences"),
    Tutoring: int = Query(..., description="Whether the student receives tutoring (0: No, 1: Yes)"),
    ParentalSupport: int = Query(..., description="Whether the student has parental support (0: No, 1: Yes)"),
    Extracurricular: int = Query(..., description="Participation in extracurricular activities (0: No, 1: Yes)"),
    Sports: int = Query(..., description="Participation in sports (0: No, 1: Yes)"),
    Music: int = Query(..., description="Involvement in music activities (0: No, 1: Yes)"),
    Volunteering: int = Query(..., description="Participation in volunteering activities (0: No, 1: Yes)"),
    GPA: float = Query(..., description="Grade Point Average of the student")
):
    data = {
        "Age": Age,
        "Gender": Gender,
        "Ethnicity": Ethnicity,
        "ParentalEducation": ParentalEducation,
        "StudyTimeWeekly": StudyTimeWeekly,
        "Absences": Absences,
        "Tutoring": Tutoring,
        "ParentalSupport": ParentalSupport,
        "Extracurricular": Extracurricular,
        "Sports": Sports,
        "Music": Music,
        "Volunteering": Volunteering,
        "GPA": GPA,
    }
    df = pd.DataFrame([data])
    predictions = predict_model(model, data = df)
    predicted_grade = predictions["prediction_label"].iloc[0]
    grade_map = {
        0: "excelent",
        1: "very good",
        2: "good",
        3: "acceptabel",
        4: "loser"
    }
    grade = grade_map.get(predicted_grade, "inknowen")
    return {"grade" : grade}