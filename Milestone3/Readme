AI Text Intelligence Suite — Streamlit NLP Application
Overview

The AI Text Intelligence Suite is an advanced NLP-powered Streamlit web application designed to analyze, summarize, and enhance written content using artificial intelligence.

This project integrates multiple Natural Language Processing (NLP) capabilities such as readability analysis, summarization, and paraphrasing, all built within a secure user authentication system. It combines the practical utility of AI models with an interactive and educational interface — ideal for students, content creators, educators, and researchers who aim to understand or refine text quality.

Project Modules & Milestones
🔹 Module 1 — Foundation & Readability Analysis

Features Implemented:

GitHub Setup & Project Structure:
Organized repository with modular components for authentication, NLP tools, and UI. Includes demo video for deployment walkthrough.
Secure Authentication (JWT):
User registration and login system with JWT tokens and bcrypt encryption for secure credential handling.
User Roles & Profile Management:
Introduced Admin and User roles. Users can set reading preferences and preferred content types for a personalized experience.
Minimal & Responsive Streamlit UI:
Built an intuitive web interface using Streamlit for seamless navigation and real-time interaction.
File/Text Upload Module:
Supports uploading .txt, .pdf, or direct text input for processing and analysis.
Readability Index Integration (via NLTK & TextStat):
Computes linguistic complexity using metrics such as:
Flesch–Kincaid Grade Level
SMOG Index
Gunning Fog Index
Coleman–Liau Index
Dynamic Score Display:
Automatically calculates and displays readability metrics upon file upload or text input.
Visual Complexity Indicators:
Uses color-coded highlights and graphical charts to visually represent the ease or difficulty of text comprehension.
Purpose & Uses

Helps writers, students, and educators assess how readable their text is.

Enables content creators to adapt tone and vocabulary for target audiences.

Serves as an educational tool to learn about linguistic complexity metrics.

🔹 Module 2 — Advanced NLP (Summarization & Paraphrasing)

Features Implemented:

Integration of Pretrained Transformer Models:

Utilizes powerful Hugging Face models for text generation:
PegasusForConditionalGeneration
FLAN-T5
BART
Adaptive Summarization Controls:

Adds toggles for Short, Medium, and Long summaries depending on user preference.
Evaluation Metrics (ROUGE):

Implements ROUGE scoring to evaluate summary quality and accuracy.
Sentence & Paragraph-Level Paraphrasing:

Uses BART or T5 to rewrite text while maintaining context, allowing users to simplify or rephrase writing.
Complexity Selector:

Dropdown menu lets users choose paraphrase difficulty —
Simple (Easy-to-read)
Standard (Balanced)
Advanced (Rich vocabulary)
Side-by-Side Comparison Interface:

Displays original vs paraphrased content for visual clarity and learning impact.
Purpose & Uses

Ideal for content rewriters, educators, and researchers to improve clarity and avoid redundancy.

Helps students simplify academic material or generate concise study notes.

Provides AI-assisted writing support, making it easier to summarize lengthy documents.

Encourages ethical and intelligent text reuse using NLP models instead of manual rewriting.

What Makes This Project Special
End-to-End AI Text System — Combines multiple NLP tools (readability, summarization, paraphrasing, chat) under one streamlined platform.
Secure & Role-Based Access — Built-in admin control ensures professional-grade system management.
Educational Focus — Demonstrates real-world NLP concepts like readability formulas, transformer models, and ROUGE metrics interactively.
Visual Intelligence — Dynamic charts and color codes make data interpretation intuitive.
Customizable Model Integration — Ready to plug in other models like GPT or custom Hugging Face pipelines.
Deployable Anywhere — Lightweight and fully compatible with Ngrok, Streamlit Cloud, or local servers.
Scalable Design — Modular structure allows easy extension for future milestones (e.g., sentiment analysis, translation, plagiarism detection).
Tech Stack
Category	Technology
Frontend	Streamlit
Backend	Python
NLP Libraries	NLTK, Transformers (Pegasus, BART, T5), Textstat
Auth & Security	bcrypt, JWT
Database	SQLite
Deployment	Pyngrok
Evaluation Metrics	ROUGE Score
Visualization	Matplotlib, Streamlit Charts
Installation & Setup
Install Dependencies

pip install -r requirements.txt
Run the Application

streamlit run app1.py --server.port 8502
Expose via Ngrok

from pyngrok import ngrok
ngrok.kill()
public_url = ngrok.connect(8502)
print(f"Public URL: {public_url}")
User Roles
Regular User
Analyze text readability
Summarize and paraphrase text
Compare results in real time
Admin
Manage user accounts
Add or delete users
Monitor system analytics
Interface Overview
Feature — Description
Login/Register Page — Secure entry with role selection
Dashboard — Displays all NLP tools
Readability Module — Visualizes linguistic metrics
Summarizer — Generates concise summaries
Paraphraser — Rewrites text with preserved meaning
Comparison View — Shows original vs modified content
Admin Panel — Manage users and data access
🎥 Demo Video
Click here to watch the demo video

Future Enhancements
Sentiment and Emotion Detection
Plagiarism Detection & Citation Assistant
Multilingual Readability Scoring
Voice Input & Output Support
Integration with OpenAI GPT Models
Download Reports in PDF Format
Contributors
Jey Harshini , Manasa mateti , Vijaya sri Vyshnavi devi , Vijendra Cherupally
Passionate about AI, NLP, and user-centric application design. Focused on creating intelligent systems that blend education, usability, and innovation.

Summary
The AI Text Intelligence Suite bridges linguistic theory and practical AI by turning NLP models into an accessible, educational, and productivity-oriented platform.

It’s not just an application — it’s an intelligent writing assistant that can read, evaluate, summarize, and rewrite text just like a human editor — but with the power of machine learning and natural language understanding.
