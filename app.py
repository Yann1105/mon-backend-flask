from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from random import randint
import pulp
import joblib

app = Flask(__name__)
CORS(app)  # Résout le problème CORS

# Paramètres globaux
DAYS = 60
NUM_HOSPITALS = 10
HOURS_PER_DAY = 24

# Charger le modèle (si utilisé)
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

# Génération des hôpitaux
def generate_hospitals():
    hospitals_data = []
    for i in range(NUM_HOSPITALS):
        ressources_totales = {"lit_rea": 15, "respirateur": 12, "scanner": 15, "lit": 20}
        urgence_ressources = {"lit_rea": 5, "respirateur": 4}
        occupation_historique = {h: {} for h in range(DAYS * HOURS_PER_DAY)}
        hospitals_data.append({
            "id": i,
            "ressources_totales": ressources_totales,
            "urgence_ressources": urgence_ressources,
            "occupation_historique": occupation_historique
        })
    return pd.DataFrame(hospitals_data)

def generate_transport_matrix():
    matrix = np.zeros((NUM_HOSPITALS, NUM_HOSPITALS), dtype=int)
    for i in range(NUM_HOSPITALS):
        for j in range(NUM_HOSPITALS):
            if i != j:
                matrix[i][j] = randint(10, 50)
                matrix[j][i] = matrix[i][j]
    return matrix

def get_available_resources(hospitals_df, hour):
    available = []
    for _, row in hospitals_df.iterrows():
        total = row["ressources_totales"].copy()
        occupation = row["occupation_historique"].get(hour, {})
        for res, qty in occupation.items():
            total[res] = max(0, total.get(res, 0) - qty)
        available.append({"total": total, "urgence": row["urgence_ressources"]})
    return available

hospitals_df = generate_hospitals()
transport_matrix = generate_transport_matrix()

def assign_patients_pl(patients_data, day):
    patients_df = pd.DataFrame(patients_data)
    if patients_df.empty:
        print("Aucun patient à traiter")
        return []

    prob = pulp.LpProblem("Patient_Assignment", pulp.LpMaximize)
    patients_day = patients_df.index.tolist()

    x = pulp.LpVariable.dicts("Assign", [(p, h) for p in patients_day for h in hospitals_df.index], cat="Binary")
    t = pulp.LpVariable.dicts("Transfer", [(p, h1, h2) for p in patients_day for h1 in hospitals_df.index for h2 in hospitals_df.index if h1 != h2], cat="Binary")

    prob += pulp.lpSum([((6 - patients_df.loc[p, "esi_level"]) ** 2) * x[(p, h)] for p in patients_day for h in hospitals_df.index]) - \
            0.01 * pulp.lpSum([t[(p, h1, h2)] for p in patients_day for h1 in hospitals_df.index for h2 in hospitals_df.index if h1 != h2])

    for p in patients_day:
        prob += pulp.lpSum([x[(p, h)] for h in hospitals_df.index]) <= 1
        for h1 in hospitals_df.index:
            prob += pulp.lpSum([t[(p, h1, h2)] for h2 in hospitals_df.index if h2 != h1]) <= x[(p, h1)]

    for h in hospitals_df.index:
        prob += pulp.lpSum([x[(p, h)] - pulp.lpSum([t[(p, h, h2)] for h2 in hospitals_df.index if h2 != h]) for p in patients_day]) <= 5

    for h in hospitals_df.index:
        for hr in range(patients_df["heure_arrivée"].min(), patients_df["heure_fin_soin"].max()):
            available = get_available_resources(hospitals_df, hr)[h]["total"]
            for res in ["lit_rea", "respirateur", "lit"]:
                prob += pulp.lpSum([patients_df.loc[p, "needs"].get(res, 0) * (x[(p, h)] - pulp.lpSum([t[(p, h, h2)] for h2 in hospitals_df.index if h2 != h]))
                                   for p in patients_day if patients_df.loc[p, "heure_arrivée"] <= hr < patients_df.loc[p, "heure_fin_soin"]]) <= available.get(res, 0)
            if hr % HOURS_PER_DAY == 0:
                prob += pulp.lpSum([patients_df.loc[p, "needs"].get("scanner", 0) * (x[(p, h)] - pulp.lpSum([t[(p, h, h2)] for h2 in hospitals_df.index if h2 != h]))
                                   for p in patients_day if patients_df.loc[p, "jour_arrivée"] == hr // HOURS_PER_DAY]) <= available.get("scanner", 0) * 20

        for p in patients_day:
            for h1 in hospitals_df.index:
                for h2 in hospitals_df.index:
                    if h1 != h2:
                        prob += t[(p, h1, h2)] * transport_matrix[h1][h2] <= patients_df.loc[p, "wait_window"]

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))

    allocation = {}
    transfers = []
    for p in patients_day:
        for h in hospitals_df.index:
            if pulp.value(x[(p, h)]) >= 0.99:
                allocation[p] = h
                for h1 in hospitals_df.index:
                    for h2 in hospitals_df.index:
                        if h1 != h2 and pulp.value(t[(p, h1, h2)]) >= 0.99:
                            transfers.append((p, h1, h2, transport_matrix[h1][h2]))
                            allocation[p] = h2
                            break
                    else:
                        continue
                    break
                break

    results = []
    for p in patients_day:
        initial_chu = allocation.get(p, None)
        transfer_info = next((t for t in transfers if t[0] == p), None)
        results.append({
            "id": patients_df.loc[p, "id"],
            "jour": int(patients_df.loc[p, "jour_arrivée"]),
            "esi": int(patients_df.loc[p, "esi_level"]),
            "pathologie": patients_df.loc[p, "pathologie"],
            "chu_initial": f"CHU {initial_chu}" if initial_chu is not None else "Non assigné",
            "chu_transfere": f"CHU {transfer_info[2]}" if transfer_info else "Aucun",
            "statut": "Assigné" if initial_chu is not None else "En attente"
        })
    return results

@app.route('/assign-patients', methods=['POST'])
def assign_patients():
    try:
        data = request.get_json()
        print("Données reçues :", data)
        patients = data.get("patients", [])
        day = data.get("day", 0)

        patients_data = []
        for i, p in enumerate(patients):
            heure_arrivée = day * HOURS_PER_DAY + randint(0, HOURS_PER_DAY - 1)
            duration = p.get("duration", 48)
            patients_data.append({
                "id": p.get("id", f"P{i+1}"),
                "jour_arrivée": day,
                "heure_arrivée": heure_arrivée,
                "esi_level": p["esi"],
                "pathologie": p["pathologie"],
                "needs": p["needs"],
                "wait_window": p["wait_window"],
                "durée_soin": duration,
                "heure_fin_soin": heure_arrivée + duration
            })

        print("Patients data processed:", patients_data)
        results = assign_patients_pl(patients_data, day)
        print("Results:", results)
        return jsonify({"results": results})
    except Exception as e:
        print(f"Erreur dans assign_patients : {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    app.run(host="0.0.0.0", port=7860)
    app.run()