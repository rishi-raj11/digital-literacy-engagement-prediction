Digital Literacy Engagement Prediction System

A Machine Learning project that predicts learner engagement levels (**Low**, **Medium**, or **High**) using digital literacy and learning behavior indicators.

Features

- Random Forest Classifier
- Interactive Streamlit Web App
- Real-time Engagement Prediction
- Confidence Score Analysis
- Feature Importance Evaluation

Tech Stack

- Python
- Pandas
- Scikit-Learn
- Streamlit

Input Parameters

- Computer Knowledge
- Internet Usage
- Mobile Literacy
- Quiz Performance
- Session Frequency
- Adaptability
- Skill Application
- Learning Feedback

Output

The model predicts:

- Low Engagement
- Medium Engagement
- High Engagement

along with prediction confidence scores.

Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Project Structure

```text
├── app.py
├── train_model.py
├── digital_literacy_dataset.csv
├── requirements.txt
└── README.md
```

Future Improvements

- Hyperparameter Tuning
- Better Model Accuracy
- IoT-based Data Integration
- Cloud Deployment


Developed by Rishiraj Purohit
