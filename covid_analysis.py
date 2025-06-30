import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# Set visualization output directory
VIS_DIR = 'visualization'
os.makedirs(VIS_DIR, exist_ok=True)

# -------------------------------
# 1. DATA ACQUISITION & LOADING
# -------------------------------
# TODO: Download or load datasets: Johns Hopkins, OWID, Oxford, Population
# Example URLs (update as needed):
# Johns Hopkins: https://github.com/CSSEGISandData/COVID-19
# OWID: https://covid.ourworldindata.org/data/owid-covid-data.csv
# Oxford: https://github.com/OxCGRT/covid-policy-tracker

def download_data():
    """
    Download required COVID-19 datasets if not already present.
    Downloads:
    - Johns Hopkins CSSE time series (confirmed, deaths, recovered)
    - Our World in Data (OWID) COVID-19 dataset
    - Oxford COVID-19 Government Response Tracker (OxCGRT)
    """
    datasets = [
        {
            'name': 'Johns Hopkins Confirmed',
            'url': 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
            'path': 'time_series_covid19_confirmed_global.csv'
        },
        {
            'name': 'Johns Hopkins Deaths',
            'url': 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',
            'path': 'time_series_covid19_deaths_global.csv'
        },
        {
            'name': 'Johns Hopkins Recovered',
            'url': 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv',
            'path': 'time_series_covid19_recovered_global.csv'
        },
        {
            'name': 'OWID',
            'url': 'https://covid.ourworldindata.org/data/owid-covid-data.csv',
            'path': 'owid-covid-data.csv'
        },
        {
            'name': 'Oxford Policy',
            'url': 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv',
            'path': 'OxCGRT_latest.csv'
        }
    ]
    for ds in datasets:
        if not os.path.exists(ds['path']):
            print(f"Downloading {ds['name']} dataset...")
            try:
                r = requests.get(ds['url'], timeout=60)
                r.raise_for_status()
                with open(ds['path'], 'wb') as f:
                    f.write(r.content)
                print(f"Downloaded {ds['path']}.")
            except Exception as e:
                print(f"Failed to download {ds['name']}: {e}")
        else:
            print(f"{ds['path']} already exists. Skipping download.")

# -------------------------------
# 2. DATA CLEANING & MERGING
# -------------------------------

def load_and_merge_data():
    """
    Load, clean, and merge Johns Hopkins and OWID datasets into a single DataFrame.
    - Aligns on country and date
    - Normalizes cases/deaths/vaccinations per 100k population
    - Handles missing data
    - Saves merged dataset as 'merged_covid_dataset.csv'
    Returns: merged DataFrame
    """
    # Load Johns Hopkins datasets (wide format)
    df_conf = pd.read_csv('time_series_covid19_confirmed_global.csv')
    df_deaths = pd.read_csv('time_series_covid19_deaths_global.csv')
    df_recov = pd.read_csv('time_series_covid19_recovered_global.csv')

    # Melt wide format to long format
    def melt_jhu(df, value_name):
        df_long = df.melt(
            id_vars=['Country/Region', 'Province/State', 'Lat', 'Long'],
            var_name='date', value_name=value_name
        )
        # Aggregate by country and date
        df_long = df_long.groupby(['Country/Region', 'date'], as_index=False)[value_name].sum()
        return df_long

    df_conf_long = melt_jhu(df_conf, 'confirmed')
    df_deaths_long = melt_jhu(df_deaths, 'deaths')
    df_recov_long = melt_jhu(df_recov, 'recovered')

    # Merge JHU datasets
    df_jhu = df_conf_long.merge(df_deaths_long, on=['Country/Region', 'date'], how='outer')
    df_jhu = df_jhu.merge(df_recov_long, on=['Country/Region', 'date'], how='outer')
    df_jhu['date'] = pd.to_datetime(df_jhu['date'], errors='coerce')
    df_jhu = df_jhu.rename(columns={'Country/Region': 'location'})

    # Load OWID dataset
    df_owid = pd.read_csv('owid-covid-data.csv', parse_dates=['date'])

    # Merge JHU with OWID on country and date
    merged = pd.merge(
        df_jhu,
        df_owid,
        left_on=['location', 'date'],
        right_on=['location', 'date'],
        how='left',
        suffixes=('', '_owid')
    )

    # Fill population from OWID
    merged['population'] = merged['population'].fillna(method='ffill').fillna(method='bfill')
    # Normalize per 100k population
    for col in ['confirmed', 'deaths', 'recovered', 'new_cases', 'new_deaths', 'new_vaccinations']:
        if col in merged.columns:
            merged[f'{col}_per100k'] = merged[col] / merged['population'] * 1e5

    # Handle missing values
    merged = merged.sort_values(['location', 'date'])
    merged = merged.groupby('location').apply(lambda group: group.fillna(method='ffill')).reset_index(drop=True)
    merged = merged.fillna(0)

    # Save merged dataset
    merged.to_csv('merged_covid_dataset.csv', index=False)
    print('Merged dataset saved as merged_covid_dataset.csv')
    return merged

# -------------------------------
# 3. ANALYSIS & VISUALIZATION
# -------------------------------

def plot_cases_by_country(df):
    """
    Enhanced: Plot country-level time series of confirmed, recovered, deaths (top 6 countries by total cases).
    - Uses 7-day rolling averages for smoother lines.
    - Larger figure, bigger fonts, clear colors, gridlines, separated legend.
    - Saves plot as visualization/cases_by_country_over_time.png
    """
    import matplotlib.dates as mdates
    top_countries = df.groupby('location')['confirmed'].max().sort_values(ascending=False).head(6).index
    plt.figure(figsize=(16, 9))
    for country in top_countries:
        country_df = df[df['location'] == country].copy()
        country_df['cases_7d'] = country_df['confirmed_per100k'].rolling(7, min_periods=1).mean()
        country_df['deaths_7d'] = country_df['deaths_per100k'].rolling(7, min_periods=1).mean()
        plt.plot(country_df['date'], country_df['cases_7d'], label=f'{country} (Cases)', linewidth=2)
        plt.plot(country_df['date'], country_df['deaths_7d'], '--', label=f'{country} (Deaths)', linewidth=2)
    plt.title('COVID-19 Confirmed Cases and Deaths Over Time (Top 6 Countries, per 100k)', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cases/Deaths per 100k', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=12, ncol=2, frameon=True)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(VIS_DIR, 'cases_by_country_over_time.png'), bbox_inches='tight', dpi=180)
    plt.close()


def plot_deaths_vs_vaccination(df):
    """
    Enhanced: Compare deaths vs vaccination rates per capita (latest available date, top 20 countries).
    - Horizontal bar plot with clear color distinction.
    - Larger figure, bigger fonts, separated legends, gridlines.
    - Saves plot as visualization/deaths_vs_vaccination.png
    """
    latest = df.groupby('location')['date'].max().reset_index()
    merged_latest = df.merge(latest, on=['location', 'date'], how='inner')
    merged_latest = merged_latest[merged_latest['population'] > 1e6]
    merged_latest = merged_latest.sort_values('deaths_per100k', ascending=False).head(20)
    plt.figure(figsize=(14, 10))
    bar1 = plt.barh(merged_latest['location'], merged_latest['deaths_per100k'], color='tomato', alpha=0.8, label='Deaths per 100k')
    bar2 = plt.barh(merged_latest['location'], merged_latest['people_fully_vaccinated_per_hundred'], color='mediumseagreen', alpha=0.6, label='Fully Vaccinated (%)')
    plt.title('Deaths vs Vaccination Rates per Capita (Top 20 Impacted Countries)', fontsize=18)
    plt.xlabel('Value', fontsize=14)
    plt.ylabel('Country', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='lower right', fontsize=12, frameon=True)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(VIS_DIR, 'deaths_vs_vaccination.png'), bbox_inches='tight', dpi=180)
    plt.close()


def plot_policy_vs_outcomes(df):
    """
    Enhanced: Compare policy stringency index vs new cases/deaths for top 4 countries.
    - Uses 7-day rolling averages for new cases/deaths.
    - Bar plots for cases/deaths, line for stringency.
    - Larger fonts, separated legends, gridlines, clearer colors.
    - Handles missing data for US and others.
    - Saves plot as visualization/policy_vs_outcomes.png
    """
    import matplotlib.dates as mdates
    top_countries = df.groupby('location')['confirmed'].max().sort_values(ascending=False).head(4).index
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
    for ax, country in zip(axes.flat, top_countries):
        country_df = df[df['location'] == country].copy()
        # Fill missing values
        for col in ['stringency_index', 'new_cases_per100k', 'new_deaths_per100k']:
            if col not in country_df or country_df[col].isnull().all():
                country_df[col] = 0
            country_df[col] = country_df[col].fillna(0)
        # Rolling averages
        country_df['cases_7d'] = country_df['new_cases_per100k'].rolling(7, min_periods=1).mean()
        country_df['deaths_7d'] = country_df['new_deaths_per100k'].rolling(7, min_periods=1).mean()
        ax2 = ax.twinx()
        # Bar plots for cases and deaths
        ax2.bar(country_df['date'], country_df['cases_7d'], width=4, color='orange', alpha=0.4, label='New Cases (7d avg)')
        ax2.bar(country_df['date'], country_df['deaths_7d'], width=4, color='red', alpha=0.3, label='New Deaths (7d avg)')
        # Line plot for stringency
        ax.plot(country_df['date'], country_df['stringency_index'], color='blue', lw=2, label='Stringency Index')
        # Titles and labels
        ax.set_title(f'{country}: Policy Stringency vs Outcomes', fontsize=15)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Stringency Index', color='blue', fontsize=12)
        ax2.set_ylabel('Cases/Deaths per 100k', color='black', fontsize=12)
        # Formatting
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='black')
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        # Legends
        ax.legend(loc='upper left', fontsize=10, frameon=True)
        ax2.legend(loc='upper right', fontsize=10, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.suptitle('Policy Stringency vs COVID-19 Outcomes (Top 4 Countries)', fontsize=18, y=1.02)
    plt.savefig(os.path.join(VIS_DIR, 'policy_vs_outcomes.png'), bbox_inches='tight', dpi=180)
    plt.close()


def plot_heatmap_correlation(df):
    """
    Enhanced: Show heatmap of correlations between key variables (cases, deaths, vaccinations, stringency, population).
    - Larger figure, bigger fonts, clear colorbar, annotated values.
    - Saves plot as visualization/heatmap_correlation.png
    """
    corr_cols = [
        'confirmed_per100k', 'deaths_per100k', 'recovered_per100k',
        'new_cases_per100k', 'new_deaths_per100k',
        'people_fully_vaccinated_per_hundred', 'stringency_index', 'population'
    ]
    df_corr = df[corr_cols].dropna().corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'shrink': 0.8}, annot_kws={"size": 14})
    plt.title('Correlation Heatmap: Cases, Deaths, Vaccination, Policy, Population', fontsize=18)
    plt.xticks(fontsize=14, rotation=45, ha='right')
    plt.yticks(fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(VIS_DIR, 'heatmap_correlation.png'), bbox_inches='tight', dpi=180)
    plt.close()

# -------------------------------
# 4. MAIN EXECUTION
# -------------------------------

def main():
    download_data()
    merged_df = load_and_merge_data()
    plot_cases_by_country(merged_df)
    plot_deaths_vs_vaccination(merged_df)
    plot_policy_vs_outcomes(merged_df)
    plot_heatmap_correlation(merged_df)
    print('All visualizations saved in the visualization/ folder.')

if __name__ == '__main__':
    main()
