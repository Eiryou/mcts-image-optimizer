# Render Deployment

This repo includes `render.yaml`.

Steps:
1) Push to GitHub
2) Render → New → Web Service → Connect repo
3) Deploy (Render reads render.yaml)

Start command:
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false
