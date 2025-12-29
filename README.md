# ML Spam Detection (SSOSM)

```bash
  _   _             _        _____                      
 | \ | |           | |      / ____|                     
 |  \| | ___   ___ | |_ ___| (___   ___ ___  _ __   ___ 
 | . ` |/ _ \ / _ \| __|_  /\___ \ / __/ _ \| '_ \ / _ \
 | |\  | (_) | (_) | |_ / / ____) | (_| (_) | |_) |  __/
 |_| \_|\___/ \___/ \__/___|_____/ \___\___/| .__/ \___|
                                            | |         
                                            |_|         

```

---

## 📌 Overview

**NOOTZSCOPE** is a Machine Learning–based email spam classifier developed for the **SSOSM laboratory**.  
It classifies emails as **clean (`cln`)** or **infected/spam (`inf`)** using generic ML techniques (TF-IDF + linear models), fully compliant with course rules.

No prebuilt anti-spam libraries are used.

---

## 🧠 Approach

- **Text representation**
  - Word n-grams (1–2)
  - Character n-grams (3–5)
- **Model**
  - Linear classifier (Logistic Regression / Linear SVM)
- **Input**
  - Email subject = first line of file
  - Email body = rest of file
  - Supports text or HTML, unknown encodings
- **Output**
  - Deterministic, strict format required by the assignment

---

## 🖥️ CLI Usage

### Project info

```bash
python main.py -info output.json
```

## 🐍 Virtual Environment Setup (Python 3.12)

It is recommended to use a virtual environment to ensure reproducible results.

### Create venv

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

deactivate
```
