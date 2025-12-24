import pandas as pd
import numpy as np

np.random.seed(42)

num_student=100

study_hours=np.random.randint(1,9,num_student)
attendance=np.random.randint(50,101,num_student)
assignment=np.random.randint(30,51,num_student)
internal_marks=np.random.randint(20,52,num_student)

final_score=(study_hours*5+attendance*0.8+assignment*0.3+internal_marks*0.5+np.random.randint(-10,10,num_student))

final_score=np.clip(final_score,40,99)

data=pd.DataFrame({"study_hours":study_hours,"attendance":attendance,"assignment":assignment,"internal_marks":internal_marks,"final_score":final_score})
data.to_csv("STUDENT_DATASET.csv",index=False)
print("Dataset created successfully.")