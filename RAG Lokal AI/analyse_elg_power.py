import pandas as pd

# Indlæs begge parquet filer
elgiganten = pd.read_parquet(
    r"C:\Users\mehdi\Documents\Dataanalyse 2 semester\Valgfag NLP\Uge 21\Torsdag\elgiganten.parquet",
    engine='fastparquet'
)

power = pd.read_parquet(
    r"C:\Users\mehdi\Documents\Dataanalyse 2 semester\Valgfag NLP\Uge 21\Torsdag\power.parquet", 
                        engine='fastparquet')

# Tjek begge datasæt
print("=== ELGIGANTEN ===")
print(f"Størrelse: {elgiganten.shape}")
print(f"Kolonner: {elgiganten.columns.tolist()}")
print(elgiganten.head(3))

print("\n=== POWER ===")
print(f"Størrelse: {power.shape}")
print(f"Kolonner: {power.columns.tolist()}")
print(power.head(3))


# tjekke rating skala

print("Elgiganten rating:")
print(elgiganten['rating'].value_counts().sort_index())

print("\nPower rating:")
print(power['rating'].value_counts().sort_index())



# Meta data på begge

# Saml dem i en fil

import pandas as pd
import re
import requests
import spacy
import gender_guesser.detector as gender
from sentida import Sentida

# Sample 1000 fra hver så det går hurtigt
elg = elgiganten.sample(1000, random_state=42).copy()
pow = power.sample(1000, random_state=42).copy()

# Tilføj brand-kolonne så vi kan skelne dem
elg['brand'] = 'Elgiganten'
pow['brand'] = 'Power'

# Saml dem i ét datasæt
df = pd.concat([elg, pow], ignore_index=True)
print(f"Samlet datasæt: {len(df)} anmeldelser")
print(df['brand'].value_counts())


# Metadata (setniment, køn og dato)

# ================================
# SENTIMENT
# ================================
sv = Sentida()

def safe_sentida(tekst):
    try:
        if len(str(tekst).strip()) < 3:  # for kort tekst
            return 0.0
        return sv.sentida(str(tekst), output="mean", normal=False)
    except:
        return 0.0  # returner 0 hvis fejl

df['sscore'] = df['contentclean'].apply(safe_sentida)
print(df['sscore'].describe())

# ================================
# KØN
# ================================
d = gender.Detector(case_sensitive=False)

def guess_gender_dk(name):
    firstname = str(name).split()[0].strip()
    result = d.get_gender(firstname, 'denmark')
    if result == 'unknown' and '-' in firstname:
        result = d.get_gender(firstname.split('-')[0], 'denmark')
    if result in ('female', 'mostly_female'): return 'F'
    if result in ('male',   'mostly_male'):   return 'M'
    return 'Ukendt'

df['gender'] = df['name'].apply(guess_gender_dk)
print(df['gender'].value_counts())

# ================================
# DATO
# ================================
df['ts']      = pd.to_datetime(df['published'])
df['year']    = df['ts'].dt.year
df['month']   = df['ts'].dt.month
df['weekday'] = df['ts'].dt.weekday

print(df[['brand', 'name', 'gender', 'sscore', 'rating']].head(6))


# sammenlignings anlayser


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# Farver
BLAA  = '#2D6BE4'   # Elgiganten
ROED  = '#E45C2D'   # Power
GRAA  = '#AAAAAA'

plt.rcParams.update({
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
    'figure.dpi':        130
})

# ================================
# ANALYSE 1: HVEM HAR GLADEST KUNDER?
# Sentiment per brand
# ================================
brand_sentiment = df.groupby('brand')['sscore'].mean().reset_index()

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(brand_sentiment['brand'], brand_sentiment['sscore'],
              color=[BLAA, ROED], width=0.4, edgecolor='white')

for bar, val in zip(bars, brand_sentiment['sscore']):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{val:.2f}', ha='center', fontsize=12, fontweight='bold')

ax.set_title("Hvem har gladest kunder?", fontsize=14, fontweight='bold')
ax.set_ylabel("Gennemsnitlig sentiment score")
plt.tight_layout()
plt.show()

# ================================
# ANALYSE 2: RATING FORDELING — SIDE OM SIDE
# ================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (brand, grp), color in zip(
    axes,
    df.groupby('brand'),
    [BLAA, ROED]
):
    counts = grp['rating'].value_counts().sort_index()
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=color, edgecolor='white')
    total = len(grp)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 5,
                f'{val/total*100:.0f}%',
                ha='center', fontsize=9)
    ax.set_title(f"{brand} — rating fordeling",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("Rating")
    ax.set_ylabel("Antal anmeldelser")

plt.tight_layout()
plt.show()

# ================================
# ANALYSE 3: SENTIMENT PER KØN PER BRAND
# ================================
koen_brand = (df[df['gender'] != 'Ukendt']
              .groupby(['brand', 'gender'])['sscore']
              .mean()
              .reset_index())

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=koen_brand, x='brand', y='sscore',
            hue='gender', palette={'M': BLAA, 'F': ROED})

ax.set_title("Sentiment per køn per brand",
             fontsize=14, fontweight='bold')
ax.set_ylabel("Gennemsnitlig sentiment")

# Fix legend — hent handles og sæt korrekte labels
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['Kvinder (F)', 'Mænd (M)'], title='Køn')

plt.tight_layout()
plt.show()
# ================================
# ANALYSE 4: SENTIMENT OVER TID — BEGGE BRANDS
# ================================
tid = (df.groupby(['brand', df['ts'].dt.to_period('Q')])['sscore']
       .mean()
       .reset_index())
tid['ts'] = tid['ts'].astype(str)

fig, ax = plt.subplots(figsize=(12, 5))
for brand, color in [('Elgiganten', BLAA), ('Power', ROED)]:
    data = tid[tid['brand'] == brand]
    ax.plot(range(len(data)), data['sscore'],
            color=color, linewidth=2.5,
            marker='o', markersize=5, label=brand)

step = 2
ax.set_xticks(range(0, len(tid[tid['brand']=='Elgiganten']), step))
ax.set_xticklabels(
    tid[tid['brand']=='Elgiganten']['ts'].iloc[::step],
    rotation=45
)
ax.legend()
ax.set_title("Sentiment over tid — Elgiganten vs Power",
             fontsize=14, fontweight='bold')
ax.set_ylabel("Gennemsnitlig sentiment")
plt.tight_layout()
plt.show()


# Gem det vi allerede har
df.to_csv(
    r"C:\Users\mehdi\Documents\Dataanalyse 2 semester\Valgfag NLP\Uge 21\Torsdag\elg_power_metadata.csv",
    index=False,
    encoding="utf-8-sig"
)
print(df.columns.tolist())
