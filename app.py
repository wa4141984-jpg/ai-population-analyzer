"""
AI-Powered World Population Analyzer
====================================
A beginner-friendly Streamlit dashboard that analyzes world population data
with Pandas / NumPy, draws interactive Plotly charts and uses Google Gemini
(Google Gen AI SDK) to explain the numbers.

Run locally with:   streamlit run app.py
"""

import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# The Google Gen AI SDK is optional at import time so the dashboard still works
# (without the AI page) if the package is missing.
try:
    from google import genai
    from google.genai import types as genai_types

    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    genai_types = None
    GENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. PAGE SETUP AND CONSTANTS
# ---------------------------------------------------------------------------
# set_page_config must be the first Streamlit command in the script.
st.set_page_config(
    page_title="World Population Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

WORLD_BANK_COUNTRY_URL = "https://api.worldbank.org/v2/country"
WORLD_BANK_POPULATION_URL = (
    "https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL"
)
DATA_SOURCE_NAME = (
    "World Bank - World Development Indicators "
    "(indicator SP.POP.TOTL, total population)"
)
CACHE_FILE = os.path.join("data", "world_population_cache.csv")

DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

WORLD_LABEL = "World Bank dataset"
CUSTOM_LABEL = "My uploaded dataset"
NONE_OPTION = "(none)"

PAGES = [
    "🏠 Dashboard",
    "🌍 Country Analysis",
    "⚖️ Population Comparison",
    "🗺️ World Map",
    "🌎 Continent Analysis",
    "📈 Historical Trends",
    "🤖 AI Insights",
    "📤 Dataset Upload",
]

# The World Bank API does not provide continents, so this small lookup table
# (country code -> continent) is used to group countries. It contains NO
# population numbers. Codes that are not listed here are shown as "Unknown".
CONTINENT_GROUPS = {
    "Africa": (
        "DZA AGO BEN BWA BFA BDI CPV CMR CAF TCD COM COD COG CIV DJI EGY GNQ ERI "
        "SWZ ETH GAB GMB GHA GIN GNB KEN LSO LBR LBY MDG MWI MLI MRT MUS MAR MOZ "
        "NAM NER NGA RWA STP SEN SYC SLE SOM ZAF SSD SDN TZA TGO TUN UGA ZMB ZWE"
    ),
    "Asia": (
        "AFG ARM AZE BHR BGD BTN BRN KHM CHN GEO HKG IND IDN IRN IRQ ISR JPN JOR "
        "KAZ KWT KGZ LAO LBN MAC MYS MDV MNG MMR NPL PRK OMN PAK PSE PHL QAT SAU "
        "SGP KOR LKA SYR TJK THA TLS TUR TKM ARE UZB VNM YEM"
    ),
    "Europe": (
        "ALB AND AUT BLR BEL BIH BGR HRV CYP CZE DNK EST FRO FIN FRA DEU GIB GRC "
        "HUN ISL IRL IMN ITA XKX LVA LIE LTU LUX MLT MDA MCO MNE NLD MKD NOR POL "
        "PRT ROU RUS SMR SRB SVK SVN ESP SWE CHE UKR GBR CHI"
    ),
    "North America": (
        "ATG ABW BHS BRB BLZ BMU VGB CAN CYM CRI CUB CUW DMA DOM SLV GRL GRD GTM "
        "HTI HND JAM MEX NIC PAN PRI KNA LCA MAF VCT SXM TTO TCA USA VIR"
    ),
    "South America": "ARG BOL BRA CHL COL ECU GUY PRY PER SUR URY VEN",
    "Oceania": (
        "ASM AUS FJI PYF GUM KIR MHL FSM NRU NCL NZL MNP PLW PNG WSM SLB TON TUV VUT"
    ),
}
CONTINENT_BY_CODE = {
    code: continent
    for continent, codes in CONTINENT_GROUPS.items()
    for code in codes.split()
}

# Column names that are recognised automatically when a CSV is uploaded.
COLUMN_ALIASES = {
    "Country": ["country", "country name", "country_name", "entity", "name", "nation"],
    "Population": ["population", "pop", "total population", "value", "population_total"],
    "Year": ["year", "date", "yr"],
    "Continent": ["continent", "region"],
    "Code": ["code", "iso3", "iso_3", "country code", "country_code", "iso code", "iso_code"],
}


# ---------------------------------------------------------------------------
# 2. STYLING AND SMALL UI HELPERS
# ---------------------------------------------------------------------------
def apply_custom_style():
    """Adds a little CSS to make the dashboard look modern."""
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            background-color: rgba(28, 131, 225, 0.08);
            border: 1px solid rgba(28, 131, 225, 0.25);
            padding: 12px 16px;
            border-radius: 10px;
        }
        .page-banner {
            background: linear-gradient(90deg, #0f4c81 0%, #1c83e1 100%);
            padding: 20px 26px;
            border-radius: 12px;
            margin-bottom: 18px;
        }
        .page-banner h1 { color: white; margin: 0; font-size: 2rem; }
        .page-banner p { color: #e6f0fa; margin: 4px 0 0 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title, subtitle):
    """Shows a coloured banner at the top of every page."""
    st.markdown(
        f'<div class="page-banner"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def show_chart(fig):
    """Displays a Plotly figure using the full available width."""
    st.plotly_chart(fig, width="stretch")


def format_population(value):
    """Turns a big number into easy-to-read text (e.g. 1.43 billion)."""
    if value is None or pd.isna(value):
        return "N/A"
    value = float(value)
    if abs(value) >= 1e9:
        return f"{value / 1e9:.2f} billion"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.2f} million"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.1f} thousand"
    return f"{value:,.0f}"


def prepare_table_for_display(table, population_columns=None):
    """Returns a copy of a table with thousands separators for nicer display."""
    shown = table.copy()
    for column in population_columns or []:
        if column in shown.columns:
            shown[column] = shown[column].map(
                lambda v: "" if pd.isna(v) else f"{v:,.0f}"
            )
    for column in shown.columns:
        if "(%)" in str(column) or column in ("Share of World (%)",):
            shown[column] = shown[column].round(2)
    return shown


def show_table(table, population_columns=None):
    """Displays a Pandas table in the app."""
    st.dataframe(
        prepare_table_for_display(table, population_columns),
        hide_index=True,
    )


def download_button(table, label, file_name, key):
    """Adds a button that downloads a table as a CSV file."""
    csv_bytes = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=file_name,
        mime="text/csv",
        key=key,
    )


# ---------------------------------------------------------------------------
# 3. DATA LOADING AND CLEANING (PANDAS)
# ---------------------------------------------------------------------------
def clean_population_data(df):
    """
    Cleans a population table that has the columns
    Country, Population, Year (required) and Continent, Code (optional).

    Cleaning steps: fix data types, remove missing values, remove invalid
    populations, remove duplicate (Country, Year) records.
    Returns the cleaned table and a small report (dictionary).
    """
    df = df.copy()
    rows_start = len(df)

    # Make sure every expected column exists.
    for column in ["Country", "Population", "Year", "Continent", "Code"]:
        if column not in df.columns:
            df[column] = np.nan

    # Remove rows with a missing country first, then tidy the text.
    df = df.dropna(subset=["Country"])
    df["Country"] = df["Country"].astype(str).str.strip()
    df = df[df["Country"] != ""].copy()

    # Population: remove thousands separators such as "1,234,567" then convert.
    population_text = df["Population"].astype(str).str.replace(",", "", regex=False)
    df["Population"] = pd.to_numeric(population_text, errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    # Remove rows with missing or invalid values.
    df = df.dropna(subset=["Population", "Year"])
    rows_after_missing = len(df)
    df = df[df["Population"] > 0].copy()
    df["Year"] = df["Year"].astype(int)

    # Remove duplicates (the same country listed twice for the same year).
    rows_before_duplicates = len(df)
    df = df.drop_duplicates(subset=["Country", "Year"], keep="first")
    duplicates_removed = rows_before_duplicates - len(df)

    # Tidy optional columns.
    df["Continent"] = df["Continent"].fillna("Unknown").astype(str).str.strip()
    df["Continent"] = df["Continent"].replace({"": "Unknown", "nan": "Unknown"})
    code_text = df["Code"].astype(str).str.strip().str.upper()
    df["Code"] = code_text.replace({"": np.nan, "NAN": np.nan, "NONE": np.nan})

    df = df[["Country", "Code", "Continent", "Year", "Population"]]
    df = df.sort_values(["Country", "Year"]).reset_index(drop=True)

    report = {
        "rows_start": rows_start,
        "rows_final": len(df),
        "missing_or_invalid_removed": rows_start - rows_after_missing,
        "duplicates_removed": duplicates_removed,
    }
    return df, report


def fetch_all_pages(url, params):
    """Downloads every page of a World Bank API response."""
    records = []
    page = 1
    while True:
        page_params = dict(params)
        page_params["page"] = page
        response = requests.get(url, params=page_params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            raise ValueError("The World Bank API returned an unexpected response.")
        meta, page_records = payload[0], payload[1]
        records.extend(page_records)
        if page >= int(meta.get("pages", 1)):
            break
        page += 1
    return records


@st.cache_data(ttl=60 * 60 * 24, show_spinner="Downloading population data from the World Bank...")
def download_world_bank_data():
    """
    Downloads real population data from the official World Bank API.
    If anything goes wrong this function raises an error, so a failed
    download is never stored in the cache.
    """
    # Step 1: list of real countries (the API also lists regions such as
    # "World" or "Europe & Central Asia", which are marked as "Aggregates").
    country_records = fetch_all_pages(
        WORLD_BANK_COUNTRY_URL, {"format": "json", "per_page": 400}
    )
    valid_countries = {}
    for item in country_records:
        region_name = (item.get("region") or {}).get("value", "")
        if region_name != "Aggregates":
            valid_countries[item["id"]] = item["name"]

    # Step 2: population for every country and year.
    end_year = datetime.now().year
    population_records = fetch_all_pages(
        WORLD_BANK_POPULATION_URL,
        {"format": "json", "per_page": 20000, "date": f"1960:{end_year}"},
    )
    rows = []
    for record in population_records:
        code = record.get("countryiso3code")
        value = record.get("value")
        if code in valid_countries and value is not None:
            rows.append(
                {
                    "Country": valid_countries[code],
                    "Code": code,
                    "Year": record["date"],
                    "Population": value,
                }
            )
    if not rows:
        raise ValueError("The World Bank API returned no population rows.")

    df = pd.DataFrame(rows)
    df["Continent"] = df["Code"].map(CONTINENT_BY_CODE).fillna("Unknown")
    df, _ = clean_population_data(df)
    return df


def get_world_bank_dataset():
    """
    Returns (dataframe, note, error_message).
    Tries the live World Bank download first and falls back to a saved copy.
    """
    try:
        df = download_world_bank_data()
        try:
            os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
            df.to_csv(CACHE_FILE, index=False)
        except OSError:
            pass  # Saving a backup copy is optional.
        return df, "live download from the World Bank API", None
    except Exception as error:  # network problems, API changes, etc.
        download_error = str(error)

    if os.path.exists(CACHE_FILE):
        try:
            saved_df, _ = clean_population_data(pd.read_csv(CACHE_FILE))
            if not saved_df.empty:
                return saved_df, "saved local copy (World Bank API was unavailable)", None
        except Exception:
            pass

    message = (
        "The World Bank population dataset could not be loaded "
        f"({download_error[:200]}). Please check your internet connection and "
        "reload the page, or upload your own CSV file on the 'Dataset Upload' page."
    )
    return None, "", message


def guess_column(columns, aliases):
    """Finds the first column whose name matches one of the aliases."""
    for column in columns:
        if str(column).strip().lower() in aliases:
            return column
    return None


def read_uploaded_csv(uploaded_file):
    """Reads an uploaded CSV file. Returns (dataframe, error_message)."""
    raw_bytes = uploaded_file.getvalue()
    if not raw_bytes:
        return None, "The uploaded file is empty."
    df = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            return None, "The CSV file has no data."
        except pd.errors.ParserError as error:
            return None, f"The CSV file could not be parsed: {str(error)[:200]}"
        except Exception as error:
            return None, f"The CSV file could not be read: {str(error)[:200]}"
    if df is None:
        return None, "The CSV file could not be decoded. Please save it as UTF-8."
    if df.empty:
        return None, "The CSV file has headers but no rows."
    df.columns = [str(column).strip() for column in df.columns]
    return df, None


def enrich_with_reference(df, reference):
    """Fills missing country codes and continents by matching country names."""
    df = df.copy()
    unique_reference = reference.drop_duplicates(subset="Country")
    code_map = dict(zip(unique_reference["Country"], unique_reference["Code"]))
    continent_map = dict(zip(unique_reference["Country"], unique_reference["Continent"]))

    df["Code"] = df["Code"].fillna(df["Country"].map(code_map))
    unknown = df["Continent"] == "Unknown"
    df.loc[unknown, "Continent"] = df.loc[unknown, "Country"].map(continent_map).fillna("Unknown")
    return df


# ---------------------------------------------------------------------------
# 4. ANALYSIS HELPERS (PANDAS + NUMPY)
# ---------------------------------------------------------------------------
def get_latest_year(df):
    """Latest year that has data for most countries (ignores incomplete years)."""
    counts = df.groupby("Year")["Country"].nunique()
    complete_years = counts[counts >= counts.max() * 0.8]
    return int(complete_years.index.max())


def get_snapshot(df, year):
    """All countries for one year."""
    return df[df["Year"] == year].copy()


def add_rank_and_share(snapshot):
    """Sorts countries by population and adds Rank and Share of World (%)."""
    snapshot = snapshot.sort_values("Population", ascending=False).reset_index(drop=True)
    snapshot["Rank"] = np.arange(1, len(snapshot) + 1)
    snapshot["Share of World (%)"] = snapshot["Population"] / snapshot["Population"].sum() * 100
    return snapshot


def calculate_cagr(start_value, end_value, years):
    """Compound annual growth rate in percent (NaN if it cannot be computed)."""
    if years <= 0 or start_value <= 0 or end_value <= 0:
        return np.nan
    return (np.power(end_value / start_value, 1.0 / years) - 1.0) * 100.0


def add_growth_columns(series):
    """Adds yearly change and growth rate (%) to a Year/Population table."""
    series = series.sort_values("Year").reset_index(drop=True)
    series["Change"] = series["Population"].diff()
    series["Growth Rate (%)"] = (series["Population"] / series["Population"].shift(1) - 1) * 100
    return series


def growth_between_years(df, start_year, end_year, min_population=0):
    """Compares every country's population in two years."""
    pivot = df.pivot_table(index="Country", columns="Year", values="Population", aggfunc="sum")
    if start_year not in pivot.columns or end_year not in pivot.columns or start_year >= end_year:
        return pd.DataFrame()
    result = pivot[[start_year, end_year]].dropna()
    result.columns = ["Start Population", "End Population"]
    result = result[result["Start Population"] >= max(min_population, 1)].copy()
    ratio = result["End Population"] / result["Start Population"]
    result["Growth (%)"] = (ratio - 1) * 100
    result["Average Yearly Growth (%)"] = (np.power(ratio, 1.0 / (end_year - start_year)) - 1) * 100
    result = result.reset_index().sort_values("Growth (%)", ascending=False)
    return result.reset_index(drop=True)


def summarize_by_continent(snapshot):
    """Groups countries by continent."""
    summary = snapshot.groupby("Continent").agg(
        Countries=("Country", "nunique"),
        Total=("Population", "sum"),
        Average=("Population", "mean"),
    )
    summary = summary.reset_index()
    summary = summary.rename(columns={"Total": "Total Population", "Average": "Average Population"})
    summary["Share of World (%)"] = summary["Total Population"] / summary["Total Population"].sum() * 100
    return summary.sort_values("Total Population", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. GEMINI AI HELPERS
# ---------------------------------------------------------------------------
def get_gemini_api_key():
    """Reads the Gemini API key from Streamlit secrets or environment variables."""
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None  # No secrets file exists - that is fine.
    if not key:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return str(key).strip() if key else None


def ask_gemini(api_key, model_name, prompt):
    """
    Sends a prompt to Gemini. Returns (answer_text, error_message);
    exactly one of the two is None.
    """
    if not GENAI_AVAILABLE:
        return None, "The 'google-genai' package is not installed. Run: pip install google-genai"
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=(
                    "You are a careful demographics analyst. Use ONLY the data "
                    "given in the prompt. Never invent statistics. If the data "
                    "does not contain the answer, say so clearly."
                ),
                temperature=0.3,
            ),
        )
        text = response.text
        if not text:
            return None, (
                "Gemini returned an empty answer (it may have been blocked by a "
                "safety filter). Please try rephrasing your request."
            )
        return text, None
    except Exception as error:
        message = str(error)
        lowered = message.lower()
        if "api key not valid" in lowered or "api_key_invalid" in lowered or "invalid api key" in lowered:
            return None, "Your Gemini API key is not valid. Please check the key and try again."
        if "permission_denied" in lowered or "403" in lowered:
            return None, "Access was denied. Check that your API key is allowed to use the Gemini API."
        if "not found" in lowered or "404" in lowered:
            return None, f"The model '{model_name}' was not found. Try another model name, for example 'gemini-2.5-flash'."
        if "429" in lowered or "quota" in lowered or "resource_exhausted" in lowered:
            return None, "The Gemini quota or rate limit was reached. Please wait a moment and try again."
        if "connect" in lowered or "timeout" in lowered or "timed out" in lowered:
            return None, "Could not reach the Gemini service. Please check your internet connection."
        return None, f"Gemini returned an error: {message[:300]}"


def build_data_summary(df, year, source_label, countries=None):
    """
    Builds a compact text summary of the dataset. Gemini only receives this
    summary (never the API key and never the whole file).
    """
    snapshot = add_rank_and_share(get_snapshot(df, year))
    lines = [
        f"Data source: {source_label}",
        f"Selected year: {year}",
        f"Years available in the dataset: {int(df['Year'].min())} to {int(df['Year'].max())}",
        f"Countries/territories in selected year: {len(snapshot)}",
        f"Total population in selected year: {snapshot['Population'].sum():,.0f}",
        f"Average population per country: {snapshot['Population'].mean():,.0f}",
        f"Median population per country: {np.median(snapshot['Population']):,.0f}",
        "",
        "Top 10 most populated countries in the selected year:",
    ]
    for _, row in snapshot.head(10).iterrows():
        lines.append(f"  {int(row['Rank'])}. {row['Country']}: {row['Population']:,.0f} ({row['Share of World (%)']:.2f}% of total)")

    lines.append("")
    lines.append("Population by continent in the selected year:")
    for _, row in summarize_by_continent(snapshot).iterrows():
        lines.append(f"  {row['Continent']}: {row['Total Population']:,.0f} ({row['Countries']} countries)")

    # World total for up to 8 evenly spaced years (NumPy picks the positions).
    yearly = df.groupby("Year")["Population"].sum()
    if len(yearly) > 1:
        positions = np.unique(np.linspace(0, len(yearly) - 1, min(8, len(yearly))).astype(int))
        lines.append("")
        lines.append("Total population of all countries over time:")
        for position in positions:
            lines.append(f"  {int(yearly.index[position])}: {yearly.iloc[position]:,.0f}")

        years = sorted(df["Year"].unique().tolist())
        growth = growth_between_years(df, years[0], years[-1], min_population=1_000_000)
        if not growth.empty:
            lines.append("")
            lines.append(f"Fastest growing countries ({years[0]} to {years[-1]}, start population of 1 million or more):")
            for _, row in growth.head(5).iterrows():
                lines.append(f"  {row['Country']}: {row['Growth (%)']:.1f}% total growth")
            lines.append("Slowest growing or declining countries (same period):")
            for _, row in growth.tail(5).iterrows():
                lines.append(f"  {row['Country']}: {row['Growth (%)']:.1f}% total growth")

    for country in countries or []:
        country_df = df[df["Country"] == country].sort_values("Year")
        if country_df.empty:
            continue
        positions = np.unique(np.linspace(0, len(country_df) - 1, min(8, len(country_df))).astype(int))
        lines.append("")
        lines.append(f"Population history of {country}:")
        for position in positions:
            row = country_df.iloc[position]
            lines.append(f"  {int(row['Year'])}: {row['Population']:,.0f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. PAGES
# ---------------------------------------------------------------------------
def page_dashboard(df, year, source_label):
    page_header("World Population Dashboard", f"Overview of population data for {year}")
    snapshot = get_snapshot(df, year)
    if snapshot.empty:
        st.warning("There is no data for the selected year.")
        return

    total = snapshot["Population"].sum()
    earlier_years = [y for y in sorted(df["Year"].unique().tolist()) if y < year]
    growth_text = None
    if earlier_years:
        previous_year = earlier_years[-1]
        previous = get_snapshot(df, previous_year)[["Country", "Population"]]
        common = snapshot.merge(previous, on="Country", suffixes=("", "_previous"))
        if not common.empty and common["Population_previous"].sum() > 0:
            growth = (common["Population"].sum() / common["Population_previous"].sum() - 1) * 100
            growth_text = f"{growth:+.2f}% vs {previous_year}"

    top_country = snapshot.sort_values("Population", ascending=False).iloc[0]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total population", format_population(total), delta=growth_text)
    col2.metric("Countries / territories", f"{len(snapshot)}")
    col3.metric("Average population", format_population(np.mean(snapshot["Population"])))
    col4.metric("Median population", format_population(np.median(snapshot["Population"])))
    col5.metric("Most populated", top_country["Country"], delta=format_population(top_country["Population"]), delta_color="off")

    left, right = st.columns(2)
    with left:
        st.subheader("Top 10 most populated countries")
        top10 = snapshot.sort_values("Population", ascending=False).head(10)
        fig = px.bar(
            top10.sort_values("Population"),
            x="Population",
            y="Country",
            orientation="h",
            color="Population",
            color_continuous_scale="Blues",
            text_auto=".3s",
        )
        fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
        show_chart(fig)
    with right:
        st.subheader("Population share by continent")
        continents = summarize_by_continent(snapshot)
        fig = px.pie(continents, names="Continent", values="Total Population", hole=0.45)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        show_chart(fig)

    yearly = df.groupby("Year", as_index=False)["Population"].sum()
    if len(yearly) > 1:
        st.subheader("Total population of all countries over time")
        fig = px.area(yearly, x="Year", y="Population")
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        show_chart(fig)

    st.subheader("Top 10 table")
    top10_table = add_rank_and_share(snapshot).head(10)[["Rank", "Country", "Continent", "Population", "Share of World (%)"]]
    show_table(top10_table, ["Population"])
    download_button(top10_table, "⬇️ Download top 10 as CSV", f"top10_population_{year}.csv", "dl_top10")
    st.caption(f"Data source: {source_label} | Year: {year}")


def page_country(df, year, source_label):
    page_header("Country Analysis", "Search a country and explore its population")
    countries = sorted(df["Country"].unique().tolist())
    search = st.text_input("🔍 Search for a country", placeholder="Type part of a country name, e.g. Pak")
    matches = [c for c in countries if search.strip().lower() in c.lower()] if search.strip() else countries
    if not matches:
        st.warning("No country matches your search. Try a different name.")
        return
    default_index = matches.index("Pakistan") if "Pakistan" in matches else 0
    country = st.selectbox("Select a country", matches, index=default_index)

    country_df = df[df["Country"] == country].sort_values("Year")
    available = country_df[country_df["Year"] <= year]
    if available.empty:
        st.warning(f"No data for {country} in or before {year}. Showing its earliest year instead.")
        row = country_df.iloc[0]
    else:
        row = available.iloc[-1]
    row_year = int(row["Year"])
    if row_year != year:
        st.info(f"{country} has no record for {year}; showing {row_year} instead.")

    year_snapshot = get_snapshot(df, row_year)
    population = float(row["Population"])
    rank = int((year_snapshot["Population"] > population).sum()) + 1
    share = population / year_snapshot["Population"].sum() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Population ({row_year})", format_population(population))
    col2.metric("World rank", f"#{rank} of {len(year_snapshot)}")
    col3.metric("Share of total", f"{share:.2f}%")
    col4.metric("Continent", str(row["Continent"]))

    if len(country_df) > 1:
        first_row, last_row = country_df.iloc[0], country_df.iloc[-1]
        total_change = (last_row["Population"] / first_row["Population"] - 1) * 100
        cagr = calculate_cagr(first_row["Population"], last_row["Population"], int(last_row["Year"] - first_row["Year"]))
        col5, col6, col7 = st.columns(3)
        col5.metric(f"Population in {int(first_row['Year'])}", format_population(first_row["Population"]))
        col6.metric(f"Total change to {int(last_row['Year'])}", f"{total_change:+.1f}%")
        col7.metric("Average yearly growth", "N/A" if pd.isna(cagr) else f"{cagr:.2f}%")

        history = add_growth_columns(country_df[["Year", "Population"]])
        left, right = st.columns(2)
        with left:
            st.subheader(f"Population of {country}")
            fig = px.line(history, x="Year", y="Population", markers=True)
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            show_chart(fig)
        with right:
            st.subheader("Growth compared with the previous record (%)")
            fig = px.bar(history.dropna(subset=["Growth Rate (%)"]), x="Year", y="Growth Rate (%)")
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            show_chart(fig)
        with st.expander("Show yearly data table"):
            show_table(history, ["Population", "Change"])
            download_button(history, f"⬇️ Download {country} data as CSV", f"{country}_population.csv", "dl_country")
    else:
        st.info("This dataset has only one year, so no history can be shown.")

    st.divider()
    st.subheader(f"All countries in {row_year}")
    table = add_rank_and_share(year_snapshot)
    if search.strip():
        table = table[table["Country"].str.lower().str.contains(search.strip().lower(), regex=False)]
    table = table[["Rank", "Country", "Continent", "Population", "Share of World (%)"]]
    show_table(table, ["Population"])
    download_button(table, "⬇️ Download this table as CSV", f"countries_population_{row_year}.csv", "dl_all_countries")
    st.caption(f"Data source: {source_label}")


def page_compare(df, year, source_label):
    page_header("Population Comparison", "Compare the population of several countries")
    snapshot = get_snapshot(df, year)
    if snapshot.empty:
        st.warning("There is no data for the selected year.")
        return
    countries = sorted(df["Country"].unique().tolist())
    default = snapshot.sort_values("Population", ascending=False).head(3)["Country"].tolist()
    selected = st.multiselect("Select countries to compare (2 or more)", countries, default=default)
    if len(selected) < 2:
        st.info("Please select at least two countries.")
        return

    chosen = add_rank_and_share(snapshot)
    chosen = chosen[chosen["Country"].isin(selected)].copy()
    if chosen.empty:
        st.warning("None of the selected countries has data for this year.")
        return
    largest = chosen["Population"].max()
    chosen["Compared with Largest (%)"] = chosen["Population"] / largest * 100

    st.subheader(f"Population in {year}")
    fig = px.bar(chosen, x="Country", y="Population", color="Country", text_auto=".3s")
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    show_chart(fig)

    history = df[df["Country"].isin(selected)].sort_values(["Country", "Year"])
    if history["Year"].nunique() > 1:
        st.subheader("Population over time")
        fig = px.line(history, x="Year", y="Population", color="Country")
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        show_chart(fig)

    st.subheader("Comparison table")
    growth_rows = []
    for country in selected:
        country_df = history[history["Country"] == country]
        if len(country_df) > 1:
            first_row, last_row = country_df.iloc[0], country_df.iloc[-1]
            cagr = calculate_cagr(first_row["Population"], last_row["Population"], int(last_row["Year"] - first_row["Year"]))
        else:
            cagr = np.nan
        growth_rows.append({"Country": country, "Average Yearly Growth (%)": cagr})
    table = chosen[["Rank", "Country", "Continent", "Population", "Share of World (%)", "Compared with Largest (%)"]]
    table = table.merge(pd.DataFrame(growth_rows), on="Country", how="left")
    show_table(table, ["Population"])
    download_button(table, "⬇️ Download comparison as CSV", f"comparison_{year}.csv", "dl_compare")

    missing = [c for c in selected if c not in chosen["Country"].tolist()]
    if missing:
        st.warning("No data in the selected year for: " + ", ".join(missing))
    st.caption(f"Data source: {source_label} | Year: {year}")


def page_map(df, year, source_label):
    page_header("World Population Map", f"Interactive map of population in {year}")
    snapshot = get_snapshot(df, year)
    if snapshot.empty:
        st.warning("There is no data for the selected year.")
        return

    has_codes = snapshot["Code"].notna().any()
    if has_codes:
        map_data = snapshot.dropna(subset=["Code"])
    else:
        map_data = snapshot
        st.info("No country codes found, so the map matches countries by name. Some countries may be missing.")

    # Cap the colour scale at the 95th percentile so smaller countries stay visible.
    upper = float(np.percentile(map_data["Population"], 95))
    try:
        if has_codes:
            fig = px.choropleth(
                map_data, locations="Code", color="Population", hover_name="Country",
                hover_data={"Code": False, "Population": ":,.0f", "Continent": True},
                color_continuous_scale="YlOrRd", range_color=(0, upper),
            )
        else:
            fig = px.choropleth(
                map_data, locations="Country", locationmode="country names", color="Population",
                hover_name="Country", hover_data={"Population": ":,.0f", "Continent": True},
                color_continuous_scale="YlOrRd", range_color=(0, upper),
            )
        fig.update_geos(projection_type="natural earth", showcountries=True)
        fig.update_layout(height=600, margin=dict(l=0, r=0, t=10, b=0))
        show_chart(fig)
        st.caption("Colours are capped at the 95th percentile so that smaller countries stay visible; hover over a country for its exact population.")
    except Exception as error:
        st.error(f"The map could not be drawn: {str(error)[:200]}")
    st.caption(f"Data source: {source_label} | Year: {year}")


def page_continent(df, year, source_label):
    page_header("Continent Analysis", "Population grouped by continent")
    snapshot = get_snapshot(df, year)
    if snapshot.empty:
        st.warning("There is no data for the selected year.")
        return
    summary = summarize_by_continent(snapshot)
    if (summary["Continent"] == "Unknown").all():
        st.warning("This dataset has no continent information, so continents cannot be analysed.")
        return
    if "Unknown" in summary["Continent"].tolist():
        st.info("Some countries have no continent information and are grouped as 'Unknown'.")

    left, right = st.columns(2)
    with left:
        st.subheader("Total population by continent")
        fig = px.bar(summary, x="Continent", y="Total Population", color="Continent", text_auto=".3s")
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
        show_chart(fig)
    with right:
        st.subheader("Share of world population")
        fig = px.pie(summary, names="Continent", values="Total Population", hole=0.45)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        show_chart(fig)

    st.subheader("Continent summary table")
    show_table(summary, ["Total Population", "Average Population"])
    download_button(summary, "⬇️ Download continent summary as CSV", f"continents_{year}.csv", "dl_continent")

    by_year = df.groupby(["Year", "Continent"], as_index=False)["Population"].sum()
    if by_year["Year"].nunique() > 1:
        st.subheader("Continent population over time")
        fig = px.area(by_year, x="Year", y="Population", color="Continent")
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        show_chart(fig)

    st.subheader("Explore one continent")
    continent = st.selectbox("Choose a continent", summary["Continent"].tolist())
    inside = add_rank_and_share(snapshot[snapshot["Continent"] == continent])
    fig = px.bar(inside.head(10).sort_values("Population"), x="Population", y="Country", orientation="h", text_auto=".3s")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    show_chart(fig)
    inside_table = inside[["Rank", "Country", "Population", "Share of World (%)"]]
    show_table(inside_table, ["Population"])
    st.caption(f"Data source: {source_label} | Year: {year}")


def page_trends(df, year, source_label):
    page_header("Historical Trends", "How population has changed over time")
    all_years = sorted(df["Year"].unique().tolist())
    if len(all_years) < 2:
        st.info("Historical analysis needs a dataset with at least two different years.")
        return

    scope = st.radio("Analyse", ["World total", "Continent", "Country"], horizontal=True)
    if scope == "World total":
        series = df.groupby("Year", as_index=False)["Population"].sum()
        label = "All countries"
    elif scope == "Continent":
        continents = sorted(df["Continent"].unique().tolist())
        label = st.selectbox("Choose a continent", continents)
        series = df[df["Continent"] == label].groupby("Year", as_index=False)["Population"].sum()
    else:
        countries = sorted(df["Country"].unique().tolist())
        label = st.selectbox("Choose a country", countries, index=countries.index("Pakistan") if "Pakistan" in countries else 0)
        series = df[df["Country"] == label].groupby("Year", as_index=False)["Population"].sum()

    start_year, end_year = st.slider("Year range", min_value=int(all_years[0]), max_value=int(all_years[-1]), value=(int(all_years[0]), int(all_years[-1])))
    series = series[(series["Year"] >= start_year) & (series["Year"] <= end_year)]
    if len(series) < 2:
        st.warning("Please select a range that contains at least two years of data.")
        return
    series = add_growth_columns(series)

    first_row, last_row = series.iloc[0], series.iloc[-1]
    total_change = (last_row["Population"] / first_row["Population"] - 1) * 100
    cagr = calculate_cagr(first_row["Population"], last_row["Population"], int(last_row["Year"] - first_row["Year"]))
    peak = series.loc[series["Growth Rate (%)"].idxmax()] if series["Growth Rate (%)"].notna().any() else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Population in {int(first_row['Year'])}", format_population(first_row["Population"]))
    col2.metric(f"Population in {int(last_row['Year'])}", format_population(last_row["Population"]))
    col3.metric("Total change", f"{total_change:+.1f}%")
    col4.metric("Average yearly growth", "N/A" if pd.isna(cagr) else f"{cagr:.2f}%")
    if peak is not None:
        st.caption(f"Fastest growth: {peak['Growth Rate (%)']:.2f}% in {int(peak['Year'])}.")

    left, right = st.columns(2)
    with left:
        st.subheader(f"Population: {label}")
        fig = px.line(series, x="Year", y="Population", markers=True)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        show_chart(fig)
    with right:
        st.subheader("Growth compared with the previous record (%)")
        fig = px.bar(series.dropna(subset=["Growth Rate (%)"]), x="Year", y="Growth Rate (%)")
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        show_chart(fig)
    with st.expander("Show data table"):
        show_table(series, ["Population", "Change"])
    download_button(series, "⬇️ Download trend data as CSV", "population_trend.csv", "dl_trend")

    st.divider()
    st.subheader("Fastest and slowest growing countries")
    years_in_range = [y for y in all_years if start_year <= y <= end_year]
    min_population = st.number_input("Only include countries with at least this many people in the first year", min_value=0, value=1_000_000, step=100_000)
    growth = growth_between_years(df, years_in_range[0], years_in_range[-1], int(min_population))
    if growth.empty:
        st.info("Not enough data to compare countries for this range.")
    else:
        st.caption(f"Comparison of {years_in_range[0]} and {years_in_range[-1]}")
        left, right = st.columns(2)
        with left:
            st.markdown("**Top 10 fastest growing**")
            fig = px.bar(growth.head(10).sort_values("Growth (%)"), x="Growth (%)", y="Country", orientation="h")
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            show_chart(fig)
        with right:
            st.markdown("**Top 10 slowest growing or declining**")
            fig = px.bar(growth.tail(10).sort_values("Growth (%)", ascending=False), x="Growth (%)", y="Country", orientation="h")
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            show_chart(fig)
        show_table(growth, ["Start Population", "End Population"])
        download_button(growth, "⬇️ Download growth table as CSV", "country_growth.csv", "dl_growth")
    st.caption(f"Data source: {source_label}")


def page_ai(df, year, source_label):
    page_header("AI Population Insights", "Powered by Google Gemini")
    st.write(
        "Gemini receives a short statistical summary of the **currently selected dataset** "
        "(not your API key and not the full file) and explains it in plain language."
    )
    if not GENAI_AVAILABLE:
        st.error("The 'google-genai' package is not installed. Run `pip install google-genai` and restart the app.")
        return

    api_key = get_gemini_api_key()
    if api_key:
        st.success("Gemini API key found (Streamlit secrets or environment variable).")
    else:
        st.warning(
            "No Gemini API key found. Add `GEMINI_API_KEY` to your Streamlit secrets or "
            "environment variables (see README.md), or paste a key below for this session only."
        )
        typed_key = st.text_input("Gemini API key (kept only in this browser session)", type="password")
        api_key = typed_key.strip() or None

    model_name = st.text_input("Gemini model", value=DEFAULT_GEMINI_MODEL, help="Change this if Google retires the default model.").strip()

    task = st.selectbox(
        "What would you like Gemini to do?",
        [
            "Explain population trends",
            "Compare countries",
            "Summarize population statistics",
            "Explain population growth patterns",
            "Ask a question about the dataset",
        ],
    )

    selected_countries = []
    question = ""
    if task == "Compare countries":
        countries = sorted(df["Country"].unique().tolist())
        default = get_snapshot(df, year).sort_values("Population", ascending=False).head(2)["Country"].tolist()
        selected_countries = st.multiselect("Countries to compare (2 or more)", countries, default=default)
    elif task == "Ask a question about the dataset":
        question = st.text_area("Your question", placeholder="Which continent grew fastest? How does the top country compare with the second?")

    if st.button("✨ Generate AI insight", type="primary"):
        if not api_key:
            st.error("Please provide a Gemini API key first.")
            return
        if not model_name:
            st.error("Please enter a Gemini model name.")
            return
        if task == "Compare countries" and len(selected_countries) < 2:
            st.error("Please select at least two countries to compare.")
            return
        if task == "Ask a question about the dataset" and not question.strip():
            st.error("Please type a question.")
            return

        instructions = {
            "Explain population trends": "Explain the main population trends visible in the data summary. Mention concrete numbers from the data.",
            "Compare countries": "Compare the selected countries using the numbers provided: size, share of the total, and how each changed over time.",
            "Summarize population statistics": "Write a clear, well-organised summary of the key population statistics in the data summary.",
            "Explain population growth patterns": "Explain the population growth patterns in the data (fast growth, slow growth, decline) and offer possible demographic reasons, labelling any reasoning that goes beyond the data as general background knowledge.",
            "Ask a question about the dataset": f"Answer this question using only the data summary: {question.strip()}",
        }
        summary = build_data_summary(df, year, source_label, selected_countries)
        prompt = f"DATA SUMMARY:\n{summary}\n\nTASK:\n{instructions[task]}\n\nUse short paragraphs or bullet points. Keep the answer under 400 words."

        with st.spinner("Gemini is analysing the data..."):
            answer, error = ask_gemini(api_key, model_name, prompt)
        if error:
            st.error(error)
        else:
            st.subheader("AI insight")
            st.markdown(answer)
            st.caption("AI answers can contain mistakes. Please verify important numbers against the tables and charts.")

    with st.expander("See the data summary that is sent to Gemini"):
        st.text(build_data_summary(df, year, source_label, selected_countries))


def page_upload(world_df):
    page_header("Dataset Upload", "Analyse your own CSV population dataset")
    message = st.session_state.pop("upload_message", None)
    if message:
        st.success(message)

    st.markdown(
        "Upload a CSV file with at least a **country** column and a **population** column. "
        "A **year** column enables historical analysis; **continent** and **ISO3 code** columns are optional. "
        "Please remove summary rows such as 'World' before uploading."
    )
    template = pd.DataFrame(columns=["Country", "Code", "Continent", "Year", "Population"])
    download_button(template, "⬇️ Download an empty CSV template", "population_template.csv", "dl_template")

    if st.session_state.get("custom_df") is not None:
        st.info(f"Currently loaded: **{st.session_state.get('custom_name', 'uploaded file')}** ({len(st.session_state['custom_df'])} rows).")
        if st.button("🗑️ Remove uploaded dataset"):
            st.session_state.pop("custom_df", None)
            st.session_state.pop("custom_name", None)
            st.session_state["next_dataset"] = WORLD_LABEL
            st.rerun()

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
        return
    raw_df, error = read_uploaded_csv(uploaded)
    if error:
        st.error(error)
        return

    st.subheader("Preview")
    st.dataframe(raw_df.head(20), hide_index=True)

    columns = list(raw_df.columns)
    optional_columns = [NONE_OPTION] + columns

    def default_index(options, name):
        guess = guess_column(columns, COLUMN_ALIASES[name])
        if guess is None:
            return 0
        return options.index(guess)

    st.subheader("Match your columns")
    col1, col2, col3 = st.columns(3)
    country_col = col1.selectbox("Country column", columns, index=default_index(columns, "Country"))
    population_col = col2.selectbox("Population column", columns, index=default_index(columns, "Population"))
    year_col = col3.selectbox("Year column", optional_columns, index=default_index(optional_columns, "Year"))
    col4, col5, col6 = st.columns(3)
    continent_col = col4.selectbox("Continent column", optional_columns, index=default_index(optional_columns, "Continent"))
    code_col = col5.selectbox("ISO3 code column", optional_columns, index=default_index(optional_columns, "Code"))
    fallback_year = col6.number_input("Year of the data (used only if there is no year column)", min_value=1000, max_value=2200, value=datetime.now().year - 1, step=1)

    use_reference = False
    if world_df is not None:
        use_reference = st.checkbox("Fill in missing continents and map codes by matching country names with the World Bank list", value=True)

    if st.button("✅ Clean and use this dataset", type="primary"):
        if country_col == population_col:
            st.error("The country and population columns must be different.")
            return
        standard = pd.DataFrame({"Country": raw_df[country_col], "Population": raw_df[population_col]})
        standard["Year"] = raw_df[year_col] if year_col != NONE_OPTION else int(fallback_year)
        standard["Continent"] = raw_df[continent_col] if continent_col != NONE_OPTION else np.nan
        standard["Code"] = raw_df[code_col] if code_col != NONE_OPTION else np.nan

        cleaned, report = clean_population_data(standard)
        if cleaned.empty:
            st.error("No valid rows remain after cleaning. Please check that the population column contains numbers.")
            return
        if use_reference and world_df is not None:
            cleaned = enrich_with_reference(cleaned, world_df)

        st.session_state["custom_df"] = cleaned
        st.session_state["custom_name"] = uploaded.name
        st.session_state["next_dataset"] = CUSTOM_LABEL
        st.session_state["upload_message"] = (
            f"Dataset loaded: {report['rows_final']} clean rows kept out of {report['rows_start']} "
            f"({report['missing_or_invalid_removed']} rows with missing/invalid values and "
            f"{report['duplicates_removed']} duplicates removed). The whole app now uses your dataset."
        )
        st.rerun()


# ---------------------------------------------------------------------------
# 7. MAIN APPLICATION
# ---------------------------------------------------------------------------
def main():
    apply_custom_style()

    world_df, world_note, world_error = get_world_bank_dataset()
    custom_df = st.session_state.get("custom_df")

    # Allow the upload page to switch the dataset (must happen before the radio is drawn).
    next_dataset = st.session_state.pop("next_dataset", None)
    if next_dataset:
        st.session_state["dataset_choice"] = next_dataset

    st.sidebar.title("🌍 World Population Analyzer")
    page = st.sidebar.radio("Navigation", PAGES)
    st.sidebar.divider()

    # Decide which dataset is active.
    dataset_options = []
    if world_df is not None:
        dataset_options.append(WORLD_LABEL)
    if custom_df is not None:
        dataset_options.append(CUSTOM_LABEL)

    active_df = None
    source_label = ""
    if len(dataset_options) > 1:
        choice = st.sidebar.radio("Dataset", dataset_options, key="dataset_choice")
    elif len(dataset_options) == 1:
        choice = dataset_options[0]
    else:
        choice = None

    if choice == WORLD_LABEL:
        active_df = world_df
        source_label = f"{DATA_SOURCE_NAME} - {world_note}"
    elif choice == CUSTOM_LABEL:
        active_df = custom_df
        source_label = f"Uploaded file: {st.session_state.get('custom_name', 'custom CSV')}"

    if world_error and world_df is None:
        st.sidebar.error("World Bank data unavailable.")

    # The upload page works even when no dataset is loaded yet.
    if page == PAGES[7]:
        page_upload(world_df)
        return

    if active_df is None or active_df.empty:
        st.error(world_error or "No dataset is available. Please upload a CSV file on the 'Dataset Upload' page.")
        return

    years = sorted(active_df["Year"].unique().tolist(), reverse=True)
    latest_year = get_latest_year(active_df)
    year = st.sidebar.selectbox("Year", years, index=years.index(latest_year))
    st.sidebar.caption(f"**Source:** {source_label}")
    st.sidebar.caption(f"Years in data: {years[-1]} to {years[0]} | Latest complete year: {latest_year}")

    page_functions = {
        PAGES[0]: page_dashboard,
        PAGES[1]: page_country,
        PAGES[2]: page_compare,
        PAGES[3]: page_map,
        PAGES[4]: page_continent,
        PAGES[5]: page_trends,
        PAGES[6]: page_ai,
    }
    page_functions[page](active_df, year, source_label)


if __name__ == "__main__":
    main()
