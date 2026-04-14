import fastf1
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# --- 0. ÖNBELLEK AYARI (VERİLERİ BİLGİSAYARA KAYDEDER) ---
if not os.path.exists('f1_cache'):
    os.makedirs('f1_cache')
fastf1.Cache.enable_cache('f1_cache')

all_race_data = []

# --- 1. DATA COLLECTION (FASTF1 - BEGINNER STYLE) ---
for year in range(2014, 2026):
    print(f"Veri çekiliyor: {year}...")
    
    # Sezonun takvimini alıyoruz
    schedule = fastf1.get_event_schedule(year)
    # Sadece geçerli yarışları (Round > 0) alıyoruz
    races = schedule[schedule['RoundNumber'] > 0]
    
    for index, event in races.iterrows():
        round_num = event['RoundNumber']
        # Pist ismini zenginleştirme (enrichment) için hazırlıyoruz
        circuit_ref = event['Location'].lower().replace(' ', '_')
        
        try:
            # Yarış oturumunu yüklüyoruz
            session = fastf1.get_session(year, round_num, 'R')
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            results = session.results
            
            # Her bir pilotun sonucunu listeye ekliyoruz
            for i, result in results.iterrows():
                row = {
                    'Year': year,
                    'Round': round_num, 
                    'Circuit': circuit_ref,
                    'Driver': result['Abbreviation'],
                    'Constructor': result['TeamName'],
                    'Status': result['Status']
                }
                all_race_data.append(row)
        except:
            continue

df = pd.DataFrame(all_race_data)
print(f"Data collection completed. Total records: {len(df)}")

# --- 2. DATA ENRICHMENT (ZENGİNLEŞTİRME) ---
circuit_info = [
    {'Circuit': 'melbourne', 'Altitude': 5, 'Type': 'Street'},
    {'Circuit': 'kuala_lumpur', 'Altitude': 18, 'Type': 'Permanent'},
    {'Circuit': 'shanghai', 'Altitude': 5, 'Type': 'Permanent'},
    {'Circuit': 'sakhir', 'Altitude': 2, 'Type': 'Permanent'},
    {'Circuit': 'barcelona', 'Altitude': 155, 'Type': 'Permanent'},
    {'Circuit': 'monte_carlo', 'Altitude': 69, 'Type': 'Street'},
    {'Circuit': 'montreal', 'Altitude': 13, 'Type': 'Street'},
    {'Circuit': 'spielberg', 'Altitude': 677, 'Type': 'Permanent'},
    {'Circuit': 'silverstone', 'Altitude': 153, 'Type': 'Permanent'},
    {'Circuit': 'hockenheim', 'Altitude': 103, 'Type': 'Permanent'},
    {'Circuit': 'budapest', 'Altitude': 236, 'Type': 'Permanent'},
    {'Circuit': 'spa', 'Altitude': 418, 'Type': 'Permanent'},
    {'Circuit': 'monza', 'Altitude': 162, 'Type': 'Permanent'},
    {'Circuit': 'marina_bay', 'Altitude': 15, 'Type': 'Street'},
    {'Circuit': 'suzuka', 'Altitude': 45, 'Type': 'Permanent'},
    {'Circuit': 'sochi', 'Altitude': 2, 'Type': 'Street'},
    {'Circuit': 'austin', 'Altitude': 161, 'Type': 'Permanent'},
    {'Circuit': 'são_paulo', 'Altitude': 785, 'Type': 'Permanent'},
    {'Circuit': 'abu_dhabi', 'Altitude': 10, 'Type': 'Permanent'},
    {'Circuit': 'mexico_city', 'Altitude': 2285, 'Type': 'Permanent'},
    {'Circuit': 'baku', 'Altitude': -28, 'Type': 'Street'},
    {'Circuit': 'le_castellet', 'Altitude': 432, 'Type': 'Permanent'},
    {'Circuit': 'zandvoort', 'Altitude': 15, 'Type': 'Permanent'},
    {'Circuit': 'imola', 'Altitude': 37, 'Type': 'Permanent'},
    {'Circuit': 'jeddah', 'Altitude': 12, 'Type': 'Street'},
    {'Circuit': 'miami', 'Altitude': 2, 'Type': 'Street'},
    {'Circuit': 'lusail', 'Altitude': 15, 'Type': 'Permanent'},
    {'Circuit': 'las_vegas', 'Altitude': 620, 'Type': 'Street'},
    {'Circuit': 'istanbul', 'Altitude': 130, 'Type': 'Permanent'},
    {'Circuit': 'mugello', 'Altitude': 242, 'Type': 'Permanent'},
    {'Circuit': 'nürburg', 'Altitude': 500, 'Type': 'Permanent'},
    {'Circuit': 'portimão', 'Altitude': 60, 'Type': 'Permanent'}
]

enrichment_df = pd.DataFrame(circuit_info)
df = pd.merge(df, enrichment_df, on='Circuit', how='left')
df['Type'] = df['Type'].fillna('Permanent')
df['Altitude'] = df['Altitude'].fillna(150)

# --- 3. MECHANICAL DNF CLASSIFICATION ---
def check_mechanical_dnf(status):
    mechanical_words = ['Engine', 'Gearbox', 'Electrical', 'Hydraulics', 'Overheating', 'Power Unit', 'Mechanical']
    for word in mechanical_words:
        if word in str(status):
            return 1
    return 0

df['is_mechanical_dnf'] = df['Status'].apply(check_mechanical_dnf)
df = df[~df['Status'].str.contains('Accident|Collision', na=False)]

# --- 4. HYPOTHESIS TESTING (ANALİZLER) ---

# Hipotez 1: Rakım
mechanical_dnf_altitudes = df[df['is_mechanical_dnf'] == 1]['Altitude']
finished_race_altitudes = df[df['is_mechanical_dnf'] == 0]['Altitude']

t_stat, p_val_altitude = stats.ttest_ind(mechanical_dnf_altitudes, finished_race_altitudes, equal_var=False)

print("\n--- Hypothesis 1: Altitude Effect (Track Altitude vs. Failure) ---")
print("P-value:", p_val_altitude)

if p_val_altitude < 0.05:
    print("Result: Track altitude significantly affects DNFs.")
    print("Analysis: Statistical evidence confirms that the altitude of a track has a meaningful impact on mechanical reliability.")
else:
    print("Result: Track altitude does not significantly affect DNFs.")
    print("Analysis: There is no statistical evidence to suggest that track altitude significantly influences the rate of mechanical failures.")

# Hipotez 2: Pist Tipi
contingency_table = pd.crosstab(df['Type'], df['is_mechanical_dnf'])
chi2_stat, p_val_type, dof, expected_values = stats.chi2_contingency(contingency_table)

print("\n--- Hypothesis 2: Circuit Type Effect (Street vs. Permanent) ---")
print("P-value:", p_val_type)

if p_val_type < 0.05:
    print("Result: Circuit type changes the DNF rate.")
    print("Analysis: There is a statistically significant difference in mechanical failure rates between street circuits and permanent tracks.")
else:
    print("Result: Circuit type does not change the DNF rate.")
    print("Analysis: The data suggests that whether a track is a street circuit or a permanent one does not significantly impact mechanical reliability.")

# Hipotez 3: Motor Yorgunluğu (Round)
mechanical_dnf_rounds = df[df['is_mechanical_dnf'] == 1]['Round']
finished_race_rounds = df[df['is_mechanical_dnf'] == 0]['Round']

t_stat, p_val_round = stats.ttest_ind(mechanical_dnf_rounds, finished_race_rounds, equal_var=False)

print("\n--- Hypothesis 3: Linear Seasonal Fatigue (Round Effect) ---")
print("P-value:", p_val_round)

if p_val_round < 0.05:
    print("Result: Component fatigue matters! DNFs happen later in the season.")
    print("Analysis: Statistical evidence shows that mechanical failures occur more frequently in the later stages of the championship.")
else:
    print("Result: DNFs are spread equally throughout the season.")
    print("Analysis: There is no significant statistical evidence that the race number (Round) affects the likelihood of failure.")


# Hipotez 4: Engine Lifecycle (Mod-5)
def get_engine_cycle_position(race_number):
    position = race_number % 7
    if position == 0:
        return 7
    return position

df['Engine_Cycle_Pos'] = df['Round'].apply(get_engine_cycle_position)

end_cycle_df = df[df['Engine_Cycle_Pos'] >= 6]
end_cycle_rates = end_cycle_df['is_mechanical_dnf']

start_cycle_df = df[df['Engine_Cycle_Pos'] <= 2]
start_cycle_rates = start_cycle_df['is_mechanical_dnf']

t_stat, p_val_cycle = stats.ttest_ind(end_cycle_rates, start_cycle_rates, equal_var=False)

print("\n--- Hypothesis 4: Engine Lifecycle Cycle (Mod-7 Effect) ---")
print("P-value:", p_val_cycle)

if p_val_cycle < 0.05:
    print("Result: Mechanical failures are significantly higher at the end of engine cycles.")
    print("Analysis: The data proves that engines are much more likely to fail in their 7th race.")
else:
    print("Result: Engine cycle position does not significantly affect the DNF rate.")
    print("Analysis: There is no statistical proof that older engines in the 7-race cycle fail more often.")

# Grafik 1: Rakım Etkisi (Boxplot - Rakım dağılımını gösterir)
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='is_mechanical_dnf', y='Altitude', palette='coolwarm')
plt.title("Hypothesis 1: Track Altitude vs Mechanical Failure")
plt.xlabel("0 = Finished, 1 = Mechanical DNF")
plt.ylabel("Altitude (meters)")
plt.savefig("graph1_altitude_effect.png")

# Grafik 2: Pist Tipi Etkisi (Barplot - Cadde vs Kalıcı Pist)
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='Type', y='is_mechanical_dnf', palette='viridis')
plt.title("Hypothesis 2: Mechanical DNF Rate by Circuit Type")
plt.ylabel("DNF Probability")
plt.savefig("graph2_circuit_type.png")

# Grafik 3: Sezon Boyunca Yorgunluk (Lineplot - Round ilerledikçe artış)
plt.figure(figsize=(10, 5))
round_rates = df.groupby('Round')['is_mechanical_dnf'].mean().reset_index()
sns.lineplot(data=round_rates, x='Round', y='is_mechanical_dnf', marker='o', color='red')
plt.title("Hypothesis 3: Motor Fatigue Over the Season")
plt.savefig("graph3_seasonal_fatigue.png")

# Grafik 4: Motor Döngüsü (Barplot - Mod-7 sistemindeki risk)
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Engine_Cycle_Pos', y='is_mechanical_dnf', palette='magma')
plt.title("Hypothesis 4: DNF Rate by Engine Lifecycle (7-Race Cycle)")
plt.xlabel("Race Number in the Cycle (1-7)")
plt.savefig("graph4_engine_lifecycle.png")

# --- 6. EXPORT (KAYDETME) ---
df.to_csv('f1_term_project_data.csv', index=False)
print("\nFormal Log: Data successfully serialized to f1_term_project_data.csv.")
plt.show()

#