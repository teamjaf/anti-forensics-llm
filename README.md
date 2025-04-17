# 🧠 Unmasking Digital Forensics: From Simulated Anti-Forensic Footprints to Large Language Model-Augmented Detection Frameworks

This public repository serves as the open-source foundation for the final project of the **Advanced Digital Forensics** course, supervised by **Prof. Gustavo Chaparro Baquero** at Florida International University. 

👤 **Project Lead**: Md Jafrin Hossain  
🤝 **Team Members**: Umme Nusrat Jahan, Kavennesh Balachandar Valarmathi, Raja Shekar Reddy Seelam

---

## 📚 Project Summary
This project presents a complete investigative framework for simulating, generating, and detecting anti-forensic techniques using synthetic data, traditional machine learning models, and large language models (LLMs).

While not industry-ready, this research-driven project is designed to explore innovative approaches in forensic analysis using AI. **Community contributions are welcome**.

---

## 🔬 Phases of the Project
1. 🛠️ Simulate malicious activities
2. 🧪 Apply anti-forensics techniques (e.g., timestomping, log deletion)
3. 🗃️ Develop a structured forensic trace database
4. 🧬 Generate synthetic forensic data
5. 📊 Train & evaluate ML models for detection of anti-forensics
6. 🤖 Build an LLM-powered assistant for log analysis

---

## 🛠 Tools Used
### Forensics / Anti-Forensics
- FTK Imager
- Autopsy
- OSForensics
- HxD Hex Editor
- Steghide

### Machine Learning Libraries
- scikit-learn
- XGBoost
- pandas, NumPy
- seaborn, matplotlib

### LLM and NLP
- HuggingFace Transformers
- FLAN-T5 (`google/flan-t5-base`)
- Gradio (for interactive interface)

---

## 🤖 Machine Learning Model Results (on 9,721 Records)

📊 Dataset Split:
- Training set: 6,804 (69.99%)
- Validation set: 1,944 (20.00%)
- Test set: 973 (10.01%)

📈 Model Performance:

**Random Forest**  
Precision: 0.7273  
Recall: 0.9412  
F1-Score: 0.8205  
False Positives: 246  

**Logistic Regression**  
Precision: 0.7156  
Recall: 1.0000  
F1-Score: 0.8342  
False Positives: 277  

**XGBoost**  
Precision: 0.7267  
Recall: 0.9383  
F1-Score: 0.8190  
False Positives: 246  

🏆 **Best Model**: Logistic Regression (F1: 0.8342)

---

## 🔍 LLM Forensics Assistant (FLAN-T5)
Using `google/flan-t5-base` to analyze forensic logs through natural language prompts.

### ✅ Quick Setup
```python
# Step 1: Install Required Libraries
!pip install -q transformers accelerate gradio

# Step 2: Import Libraries
from transformers import pipeline
import pandas as pd

# Step 3: Load Dataset
df = pd.read_csv("adf_fp.csv")

# Step 4: Prompt Generator
def generate_prompt(row):
    return (
        f"Analyze this forensic log:\n"
        f"Timestamp: {row['Timestamp']}\n"
        f"User: {row['User']}\n"
        f"Action: {row['Action']}\n"
        f"File: {row['File_Involved']}\n"
        f"Network IP: {row['Network_IP']}\n"
        f"Process Name: {row['Process_Name']}\n"
        f"Registry Modified: {row['Registry_Modified']}\n"
        f"Persistence Technique: {row['Persistence_Technique']}\n"
        f"Log Cleared: {row['Log_Cleared']}\n"
        f"Timestomped: {row['Timestomped']}\n"
        f"Encoded Payload: {row['Encoded_Payload']}\n"
        f"Encrypted Channel: {row['Encrypted_Channel']}\n"
        f"Uses Anti-Forensics: {row['Uses_Anti_Forensics']}\n"
        f"Explain whether this is an anti-forensic attempt and why."
    )

# Step 5: Load LLM Model
pipe = pipeline("text2text-generation", model="google/flan-t5-base")

# Step 6: Run Example
prompt = generate_prompt(df.iloc[0])
result = pipe(prompt, max_new_tokens=200)[0]['generated_text']
print(result)
```

### ✅ Gradio App (Optional UI)
```python
import gradio as gr

def analyze_row(index):
    prompt = generate_prompt(df.iloc[int(index)])
    output = pipe(prompt, max_new_tokens=200)[0]['generated_text']
    return output

gr.Interface(
    fn=lambda text: pipe(text, max_new_tokens=200)[0]['generated_text'],
    inputs=gr.Textbox(label="🔍 Enter your forensic log for analysis", lines=10, placeholder="Paste your forensic log here..."),
    outputs="text",
    title="🧠 Forensic LLM Assistant (Freeform Log Analyzer)"
).launch()
```

---

## 📎 Citation
This work was completed as part of the **EEL6803: Advanced Digital Forensics** course. All contributions are for educational and research purposes.

Feel free to fork, use, and contribute to this repo for learning and academic exploration.

---

> "The goal of forensics is not just to discover what happened — but to uncover what was meant to be hidden."

---
