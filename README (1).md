# 🚀 CreditWise Deployment Guide
## Step-by-Step: Deploy to Streamlit Cloud (Free & Easy)

---

## 📁 Your Project Files

```
creditwise_app/
├── app.py                    ← Main Streamlit application
├── loan_approval_data.csv    ← Dataset
├── requirements.txt          ← Python dependencies
└── README.md                 ← This file
```

---

## STEP 1 — Test Locally First

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
# OR: venv\Scripts\activate    # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```
App will open at → http://localhost:8501

---

## STEP 2 — Push to GitHub

1. Create a free account at https://github.com
2. Create a **New Repository** named `creditwise-loan-system`
3. Run these commands in your project folder:

```bash
git init
git add .
git commit -m "Initial CreditWise ML app"
git remote add origin https://github.com/YOUR_USERNAME/creditwise-loan-system.git
git push -u origin main
```

Make sure `loan_approval_data.csv` is included in the repo!

---

## STEP 3 — Deploy on Streamlit Community Cloud (FREE)

1. Go to → https://share.streamlit.io
2. Sign in with your **GitHub account**
3. Click **"New app"**
4. Fill in:
   - **Repository:** `YOUR_USERNAME/creditwise-loan-system`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy!"**

✅ Your app will be live at:
`https://YOUR_USERNAME-creditwise-loan-system-app-XXXXX.streamlit.app`

---

## STEP 4 — Share Your App

Copy the live URL and share it with anyone — no installation needed!

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Check `requirements.txt` has all packages |
| `FileNotFoundError: loan_approval_data.csv` | Make sure CSV is committed to GitHub |
| App crashes on predict | Ensure all feature names match training |
| Slow loading | Normal on first visit — model trains fresh each time |

---

## 🌟 Optional Enhancements

- **Save trained model:** Add `joblib.dump(model, 'model.pkl')` and load on startup for faster inference
- **Add authentication:** Use `streamlit-authenticator` for login
- **Custom domain:** Available in Streamlit Cloud paid plans

---

## Quick Commands Reference

```bash
streamlit run app.py              # Run locally
streamlit run app.py --server.port 8080   # Custom port
pip freeze > requirements.txt     # Update requirements
```
