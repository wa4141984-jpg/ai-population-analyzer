# 🌍 AI-Powered World Population Analyzer

A beginner-friendly **Streamlit** dashboard that analyzes real world population data with **Pandas** and **NumPy**, draws interactive **Plotly** charts and maps, and uses **Google Gemini** (via Google AI Studio) to explain the numbers in plain language.

## Application features

| # | Feature | Where |
|---|---------|-------|
| 1 | Total world population (with change vs. previous year) | Dashboard |
| 2 | Search and select any country | Country Analysis |
| 3 | Country-wise population table (rank, share of total) | Country Analysis |
| 4 | Top 10 most populated countries | Dashboard |
| 5 | Compare several countries (bar chart, history, growth) | Population Comparison |
| 6 | Continent-wise population (bar, pie, over time) | Continent Analysis |
| 7 | Historical growth, yearly growth rate, average yearly growth, fastest/slowest growing countries | Historical Trends |
| 8 | Interactive charts and tables | All pages |
| 9 | Interactive world population map | World Map |
| 10 | Upload your own CSV dataset (column matching + cleaning report) | Dataset Upload |
| 11 | Download results as CSV | Download buttons on every analysis page |
| 12 | AI Population Insights (Gemini) | AI Insights |

The sidebar has page navigation, a **Year** selector, and a **Dataset** switch (appears after you upload a CSV).

## Data source (no fake numbers)

Population figures come from the official **World Bank API** – *World Development Indicators, indicator `SP.POP.TOTL` (total population)*, yearly data from 1960 onward. The app downloads them when it starts and caches them for 24 hours. The data source and the selected year are shown on every page.

* Regional aggregates such as "World" or "Euro area" are removed, so only countries and territories remain.
* The World Bank API has no continents, so the app uses a small built-in lookup table (country code → continent). It contains **no population numbers**. Codes missing from the table appear as `Unknown`.
* After a successful download, a backup copy is saved to `data/world_population_cache.csv`. If the API is unreachable later, the app uses that copy. If neither is available, a clear error message is shown and you can still upload your own CSV.

## Technology stack

* **Python** 3.10+ (3.11 or 3.12 recommended)
* **Pandas** and **NumPy** – cleaning, filtering, sorting, grouping, growth calculations
* **Streamlit** – user interface
* **Plotly** – interactive charts and map
* **Google Gen AI SDK** (`google-genai`) with **Google AI Studio / Gemini API** – AI insights
* **Requests** – downloads World Bank data
* **GitHub** (code storage) and **Streamlit Community Cloud** (deployment)

## Project folder structure

```
world-population-analyzer/
├── app.py                 # the whole application
├── requirements.txt       # Python dependencies
├── README.md              # this file
├── .gitignore             # (you create it, see below) keeps secrets out of GitHub
├── .streamlit/
│   └── secrets.toml       # (you create it, local only) holds GEMINI_API_KEY
└── data/
    └── world_population_cache.csv   # created automatically after the first download
```

## Installation and running locally

1. Install Python 3.10 or newer from <https://www.python.org/downloads/>.
2. Put `app.py`, `requirements.txt` and `README.md` in a folder, then open a terminal in that folder.
3. (Recommended) create a virtual environment:

   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS / Linux:
   source venv/bin/activate
   ```

4. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Add your Gemini API key (next section), then start the app:

   ```bash
   streamlit run app.py
   ```

   Your browser opens at <http://localhost:8501>. The first start takes a few seconds while the World Bank data downloads.

## Google AI Studio (Gemini) API key setup

1. Open <https://aistudio.google.com/apikey> and sign in with your Google account.
2. Click **Create API key** and copy it. Never share it or post it online.
3. Give the key to the app in **one** of these ways. The app looks for `GEMINI_API_KEY` first in Streamlit secrets, then in environment variables.

   **Option A – Streamlit secrets (recommended).** Create the file `.streamlit/secrets.toml`:

   ```toml
   GEMINI_API_KEY = "paste-your-key-here"
   ```

   **Option B – environment variable.**

   ```bash
   # macOS / Linux
   export GEMINI_API_KEY="paste-your-key-here"
   # Windows PowerShell
   $env:GEMINI_API_KEY="paste-your-key-here"
   ```

   **Option C – paste it into the AI Insights page.** It is kept only for your current browser session and is never saved.

4. The AI Insights page has a **Gemini model** box (default `gemini-2.5-flash`). If Google retires that model and you see a "model not found" message, enter a current model name from <https://ai.google.dev/gemini-api/docs/models>. You can also set a different default with the `GEMINI_MODEL` environment variable.

What Gemini receives: only a short text summary of the dataset (top countries, continent totals, yearly totals, growth highlights). You can read it in the *"See the data summary that is sent to Gemini"* box. The API key is never included, and the full dataset is never sent.

Invalid keys, missing keys, quota limits, unknown model names and network problems show friendly error messages instead of crashing the app.

## Uploading your own CSV

Open **Dataset Upload**, upload a CSV and confirm the column matching. Common column names are detected automatically.

| Column | Required? | Example names that are detected |
|--------|-----------|---------------------------------|
| Country | Yes | `Country`, `Country Name`, `Entity` |
| Population | Yes | `Population`, `Pop`, `Value` |
| Year | No (you can enter one year) | `Year`, `Date` |
| Continent | No | `Continent`, `Region` |
| ISO3 code | No (needed for best map results) | `Code`, `ISO3`, `Country Code` |

The app converts numbers like `1,234,567`, removes rows with missing or invalid populations, and removes duplicate country/year rows, then shows a cleaning report. Remove summary rows such as "World" before uploading. After loading, every page uses your data; switch back with the **Dataset** selector in the sidebar.

## Running in Google Colab

Colab cannot open `localhost` directly, so a temporary tunnel is used.

1. Open <https://colab.research.google.com> and create a notebook.
2. Click the 🔑 **Secrets** icon in the left bar, add a secret named `GEMINI_API_KEY` with your key, and enable *Notebook access*.
3. Upload `app.py` and `requirements.txt` (Files panel → Upload), then run each block in its own cell:

   ```python
   !pip install -q -r requirements.txt
   ```

   ```python
   import os
   from google.colab import userdata
   os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
   ```

   ```python
   !streamlit run app.py --server.headless true > streamlit_log.txt 2>&1 &
   !curl https://loca.lt/mytunnelpassword
   !npx localtunnel --port 8501
   ```

4. Open the `https://....loca.lt` link that is printed. If asked for a tunnel password, paste the IP address printed by the `curl` command.

## Uploading the project to GitHub

1. Create a free account at <https://github.com> and click **New repository** (for example `world-population-analyzer`). Do not add a README there, because you already have one.
2. In your project folder create a file named `.gitignore` with this content, so your key is never uploaded:

   ```
   .streamlit/secrets.toml
   .env
   venv/
   __pycache__/
   ```

3. Upload with Git:

   ```bash
   git init
   git add app.py requirements.txt README.md .gitignore
   git commit -m "World Population Analyzer"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/world-population-analyzer.git
   git push -u origin main
   ```

   (Or, in the browser: **Add file → Upload files**, then drag `app.py`, `requirements.txt`, `README.md` and `.gitignore`. Never upload `secrets.toml`.)

## Deploying on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Click **Create app** (or **New app**) and choose your repository, branch `main`, and main file path `app.py`.
3. Open **Advanced settings**, pick Python 3.11 or 3.12, and paste your secret in the **Secrets** box:

   ```toml
   GEMINI_API_KEY = "paste-your-key-here"
   ```

4. Click **Deploy**. The first build installs `requirements.txt` and takes a few minutes. You then get a public link to share.

To change the key later, open the app's **Settings → Secrets** on Streamlit Cloud.

## How to run the application (quick summary)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Troubleshooting

* **"World Bank dataset could not be loaded"** – check your internet connection and reload, or use the Dataset Upload page.
* **"API key is not valid"** – create a new key in Google AI Studio and update your secret.
* **"Model not found"** – change the model name on the AI Insights page.
* **Quota / rate limit message** – wait a minute; the free tier has request limits.
* **Map shows few countries for an uploaded file** – add an ISO3 code column (e.g. `IND`, `USA`) or keep the "match with the World Bank list" option ticked.# ai-population-analyzer
