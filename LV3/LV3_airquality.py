import urllib.request
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

url = (
    'http://iszz.azo.hr/iskzl/rs/podatak/export/xml?'
    'postaja=160&polutant=5&tipPodatka=0'
    '&vrijemeOd=01.01.2017&vrijemeDo=31.12.2017'
)

try:
    print("Dohvaćanje podataka s REST API-ja...")
    xml_bytes = urllib.request.urlopen(url, timeout=30).read()
    root = ET.fromstring(xml_bytes)

    rows = []

    for child in root:
        elems = list(child)

        try:
            rows.append({
                'mjerenje': float(elems[0].text),
                'vrijeme': elems[2].text
            })
        except (IndexError, ValueError, TypeError):
            continue

    df = pd.DataFrame(rows)

    df['vrijeme'] = pd.to_datetime(df['vrijeme'], utc=True)

    df['month'] = df['vrijeme'].dt.month
    df['dayOfWeek'] = df['vrijeme'].dt.dayofweek

    print(f"\nDataFrame: {len(df)} redova")
    print(df.head(3).to_string(index=False))

    print("\n" + "─" * 55)
    print("3 datuma s najvećom dnevnom koncentracijom PM10 (μg/m³):")

    top3 = df.sort_values('mjerenje', ascending=False).head(3)

    for _, row in top3.iterrows():
        datum = row['vrijeme'].strftime('%d.%m.%Y')
        print(f"   {datum} → {row['mjerenje']:.1f} μg/m³")

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(
        df['vrijeme'],
        df['mjerenje'],
        color='steelblue',
        linewidth=0.9,
        alpha=0.85,
        label='PM10'
    )

    for _, row in top3.iterrows():
        ax.annotate(
            f"{row['mjerenje']:.0f}",
            xy=(row['vrijeme'], row['mjerenje']),
            xytext=(10, 8),
            textcoords='offset points',
            fontsize=8,
            color='red',
            arrowprops=dict(
                arrowstyle='->',
                color='red',
                lw=0.8
            )
        )

    ax.axhline(
        50,
        color='orange',
        linestyle='--',
        linewidth=1,
        label='Granična vrijednost (50 μg/m³)'
    )

    ax.set_title('Dnevna koncentracija PM10 – Osijek, 2017.')
    ax.set_xlabel('Datum')
    ax.set_ylabel('Koncentracija PM10 (μg/m³)')
    ax.legend()
    ax.grid(linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

except urllib.error.URLError as e:
    print(f"Greška pri dohvatu podataka: {e}")

except ET.ParseError as e:
    print(f"Greška pri parsiranju XML-a: {e}")

except Exception as e:
    print(f"Neočekivana greška: {e}")