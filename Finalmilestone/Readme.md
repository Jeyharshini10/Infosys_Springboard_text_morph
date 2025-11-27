# 🧠 TextMorph
### AI-Powered Content Simplification, Summarization & Paraphrasing Suite

Transforming complex text into clear, concise, and accessible communication.

---

## 🔗 Quick Links
| Category | Link |
|----------|------|
| Demo Video | Coming Soon |
| Source Code | This Repository |
| Docker Support | Yes |
| AI Models | Pegasus · BART · FLAN-T5 |

---

## 📖 About the Project
TextMorph is an AI-powered platform that performs *summarization, **paraphrasing, and **readability analysis* using transformer-based NLP models.  
It helps students, researchers, and professionals understand complex documents with ease and enhances clarity through rewriting and simplification.

Built as part of the *Infosys Springboard Internship Final Project*.

---

## 🎯 Problem Statement
Users struggle to understand long and complex textual content, especially in academic, legal, and medical domains.  
Manual summarization is time-consuming, rewriting is difficult, and existing tools only solve partial needs without security or history tracking.

### Input:
- Raw text (50–3000 words), document files

### Output:
- Summaries (Short / Medium / Long)
- Paraphrased content (Simple / Neutral / Advanced)
- Readability metrics + visual gauges

---

## 🚀 Key Features

### 👤 User Features
- 🔐 Secure JWT Authentication & Registration
- 📊 Readability Analyzer (Flesch, Gunning Fog, SMOG, Coleman-Liau)
- ✂️ Summarization Engine (Pegasus/BART/T5)
- 🔁 Paraphrasing Engine (Sentence or Paragraph level)
- 👥 Side-by-Side Comparison View
- ⭐ Rating + Comments
- 🕘 History with Reuse
- 🧑 Profile Management

### 🛠 Admin Features
- Manage Users: Add / Delete / Promote (max 2 admins)
- Visual Analytics & Usage Trends
- Global Search: feedback + history + actions
- System-wide audit logging

---

## 🧩 System Architecture

### Workflow

![Image](https://github.com/user-attachments/assets/a7997fe9-3226-4f4b-80e0-7516029f2299)


## 🛠 Tech Stack

| Layer | Technologies |
|--------|-------------|
| Frontend | Streamlit |
| Backend | FastAPI, Python |
| NLP Models | Pegasus, BART, FLAN-T5 |
| Database | SQLite |
| Deployment | Docker |
| Security | JWT + bcrypt |

---

## 🤖 Models Used

| Model | Purpose |
|--------|----------|
| Pegasus | High-quality abstractive summarization |
| BART | Balanced summarization & rewriting |
| FLAN-T5 | Paraphrasing & complexity control |
| NLTK | Readability metrics engine |



<h3>Project Structure</h3>

<pre>
TextMorph/
│
├── app.py                  # Streamlit UI
├── backend/
│   ├── auth.py             # JWT security & login logic
│   ├── models.py           # DB schema
│   ├── ml_engine.py        # Summarization & paraphrasing core
│   ├── readability.py      # Readability scoring
│   ├── history.py          # History logging
│   ├── feedback.py         # Ratings & comments
│   └── admin.py            # Admin permissions
│
├── requirements.txt
├── Dockerfile
├── .env.example
└── docs/
    ├── architecture.png
    ├── db_schema.png
    └── screenshots/
</pre>



## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- Git
- (Optional) Docker

### Clone Repository
bash
git clone <repository-link>
cd TextMorph
pip install -r requirements.txt

## 🛠 Setup Environment

Create a `.env` file in the project root and add the following:

env
JWT_SECRET_KEY=your_secret_key_here
SMTP_EMAIL=your_email_here
SMTP_PASS=your_app_password_here


### Run Project
streamlit run app.py

### How to Use
- Register / Login
- Upload text or document
- Select summarization or paraphrasing
- Adjust output style
- View results with comparison
- Submit feedback and save history
- Admin can review system analytics

### Datasets & Evaluation

#### Datasets
| Dataset | Purpose |
|---------|---------|
| WikiAuto | Text simplification |
| Newsela | Grade-level rewriting |
| ASSET | Paraphrasing benchmark |

#### Evaluation Metrics
- ROUGE-L
- BLEU
- Readability Delta
- Perplexity


### Roadmap
- Fine-tuned custom models
- GPU cloud deployment
- Support for multiple languages
- Mobile application
- Advanced visualization dashboards

### Team
| Name | Role | Responsibility |
|-------|-------|----------------|
| Team Members | ML Engineer | Model Integration & Evaluation |
| … | Backend Developer | JWT + DB |
| … | Frontend Developer | Streamlit UI |
| … | Documentation | PPT, Report, README |

### License
MIT License — Free to use, modify, and distribute with credits

### Support
If you like this project, please ⭐ star the repository!
