import pandas as pd
import lotus
from lotus.models import LM
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["AZURE_API_KEY"] = os.getenv("AZURE_API_KEY")
os.environ["AZURE_ENDPOINT_URL"] = os.getenv("AZURE_API_BASE")
os.environ["AZURE_API_VERSION"] = os.getenv("AZURE_API_VERSION", "2024-05-01-preview")
model_deployment = os.getenv("LLM_DEPLOYMENT", "gpt-4o-mini")

# configure the LM, and remember to export your API key
lm = LM(model=model_deployment)
lotus.settings.configure(lm=lm)

# create dataframes with course names and skills
courses_data = {
    "Course Name": [
        "History of the Atlantic World",
        "Riemannian Geometry",
        "Operating Systems",
        "Food Science",
        "Compilers",
        "Intro to computer science",
    ]
}
skills_data = {"Skill": ["Math", "Computer Science"]}
courses_df = pd.DataFrame(courses_data)
skills_df = pd.DataFrame(skills_data)

# lotus sem join 
print("performing join now")
res = courses_df.sem_join(skills_df, "Taking {Course Name} will help me learn {Skill}")
print(res)

# Print total LM usage
lm.print_total_usage()