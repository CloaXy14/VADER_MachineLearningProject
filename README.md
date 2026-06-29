 VADER: Video Addiction Diagnostic & Evaluation Routine

**A Machine Learning Decision Support System (DSS) for Digital Wellbeing.**

VADER is an end-to-end, classical machine learning application designed to classify problematic short-video usage patterns from proxy behavioral telemetry. Instead of merely tracking screen time, this system analyzes nuanced habits—such as midnight watch spikes, daily session frequency, and content breadth—to output actionable, probability-based diagnostics.

### Tech Stack
* **Machine Learning:** `scikit-learn`, `pandas`, `numpy`
* **Deployment:** `streamlit`
* **Database:** `sqlite3`
* **Language:** `Python 3.12`


### How to Initialize this project:

1.) Open terminal in VSCode or Powershell if you are in Windows

2.) Create a new virtual enviroment, run [python -m venv .venv] in the terminal

3.) Activate the new virtual environment, run [.venv\Scripts\Activate] in the terminal

4.) Install required dependencies, run [pip install -r requirements.txt] in the terminal

5.) Run streamlit to view the website, run [streamlit run app.py] in the terminal

## How to use the application

1.) Fill in the columns with your current digital use information

2.) Once it is filled, click the button that says "Generate Diagnostic Reports" to view your digital wellbeing.

After doing those steps, you can see your diagnostic results, which is how addicted you are to your gadgets, you can also see the Confidence Probability Matrix determining your status, your Clinical Notes too that gives a comment based on your diagnostic results, Last but not least, you can see the feedback section where you can give feedback towards the results of your digital wellbeing diagnosis.
