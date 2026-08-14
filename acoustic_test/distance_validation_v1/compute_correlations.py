import csv
import numpy as np

# Load recording summary
rows = []
with open('acoustic_test/distance_validation_v1/analysis/results/distance_recording_summary.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Compute correlation between distance and median_delay_ms (per recording)
d = []
m = []
for r in rows:
    if r['median_delay_ms'] != '' and int(r['distance_cm']) > 0:
        d.append(int(r['distance_cm']))
        m.append(float(r['median_delay_ms']))

d = np.array(d)
m = np.array(m)
corr = np.corrcoef(d, m)[0, 1]
print(f'Per-recording correlation (distance vs median_delay): {corr:.4f}')
print(f'n={len(d)}')

# Also compute using per-condition medians
cond_medians = {}
for r in rows:
    if r['median_delay_ms'] != '' and int(r['distance_cm']) > 0:
        cond = r['condition']
        cond_medians.setdefault(cond, []).append(float(r['median_delay_ms']))

cd = []
cm = []
for cond in ['10cm', '20cm', '30cm', '40cm', '50cm']:
    if cond in cond_medians:
        cd.append(int(cond.replace('cm', '')))
        cm.append(np.median(cond_medians[cond]))

cd = np.array(cd)
cm = np.array(cm)
corr2 = np.corrcoef(cd, cm)[0, 1]
print(f'Per-condition median correlation: {corr2:.4f}')
print(f'Per-condition medians: {dict(zip(cd, cm))}')

# Also compute using per-condition means
cm_mean = []
for cond in ['10cm', '20cm', '30cm', '40cm', '50cm']:
    if cond in cond_medians:
        cm_mean.append(np.mean(cond_medians[cond]))

cm_mean = np.array(cm_mean)
corr3 = np.corrcoef(cd, cm_mean)[0, 1]
print(f'Per-condition mean correlation: {corr3:.4f}')
print(f'Per-condition means: {dict(zip(cd, cm_mean))}')

# Also compute using the condition_metrics.csv
print()
print('From condition_metrics.csv:')
with open('acoustic_test/distance_validation_v1/analysis/results/distance_condition_metrics.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['distance_cm'] != '0' and row['median_delay_ms'] != '':
            print(f"  {row['condition']}: distance={row['distance_cm']} "
                  f"expected={row['expected_delay_ms']} "
                  f"median_delay={row['median_delay_ms']} "
                  f"detection_rate={row['detection_rate_mean']}")