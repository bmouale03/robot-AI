import streamlit as st
import numpy as np
import csv
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
import io
import time

# Paramètres Q-learning
gamma = 0.75
alpha = 0.9

# Etats
location_to_state = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11
}
state_to_location = {state: location for location, state in location_to_state.items()}

# Enregistrement CSV
def save_route_to_csv(start_point, end_point, route, travel_time_seconds=None, filename="optimal_routes.csv"):
    try:
        with open(filename, 'r'):
            pass
    except FileNotFoundError:
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Timestamp", "Start Point", "End Point",
                "Optimal Route", "Number of Steps", "Travel Time (s)"
            ])

    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        route_str = " -> ".join(route)
        writer.writerow([
            timestamp, start_point, end_point,
            route_str, len(route) - 1, travel_time_seconds
        ])

def calculate_travel_time(route, time_per_step=5):
    steps = len(route) - 1
    total_seconds = steps * time_per_step
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return minutes, seconds, total_seconds

def route(starting_location, ending_location):
    R = np.array([
        [0,1,0,0,0,0,0,0,0,0,0,0],
        [1,0,1,0,0,1,0,0,0,0,0,0],
        [0,1,0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,0,0,0],
        [0,1,0,0,0,0,0,0,0,1,0,0],
        [0,0,1,0,0,0,0,1,0,0,0,0],
        [0,0,0,1,0,0,1,0,0,0,0,1],
        [0,0,0,0,1,0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0,0,1,0,1,0],
        [0,0,0,0,0,0,0,0,0,1,0,1],
        [0,0,0,0,0,0,0,1,0,0,1,0]
    ])

    ending_state = location_to_state[ending_location]
    R[ending_state, ending_state] = 1000
    Q = np.zeros([12, 12])

    for _ in range(1000):
        current_state = np.random.randint(0, 12)
        playable_actions = [j for j in range(12) if R[current_state, j] > 0]
        next_state = np.random.choice(playable_actions)
        TD = R[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] += alpha * TD

    route_path = [starting_location]
    next_location = starting_location
    while next_location != ending_location:
        starting_state = location_to_state[next_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route_path.append(next_location)

    travel_time = calculate_travel_time(route_path)
    save_route_to_csv(starting_location, ending_location, route_path, travel_time_seconds=travel_time[2])
    return route_path

def best_route(starting_location, ending_location, intermediary_location):
    route1 = route(starting_location, intermediary_location)
    route2 = route(intermediary_location, ending_location)[1:]
    full_route = route1 + route2
    travel_time = calculate_travel_time(full_route)
    save_route_to_csv(starting_location, ending_location, full_route, travel_time_seconds=travel_time[2])
    return full_route

def draw_route_graph(route):
    G = nx.Graph()
    edges = [
        ('A', 'B'), ('B', 'C'), ('B', 'F'), ('C', 'G'),
        ('F', 'J'), ('G', 'H'), ('H', 'D'), ('H', 'L'),
        ('J', 'I'), ('J', 'K'), ('K', 'L'), ('I', 'E'),
    ]
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=42)

    node_colors = ['green' if node == route[0] else
                   'red' if node == route[-1] else
                   'orange' if node in route else 'lightgray'
                   for node in G.nodes()]

    route_edges = set(zip(route, route[1:])) | set(zip(route[1:], route))
    edge_colors = ['blue' if edge in route_edges else 'gray' for edge in G.edges()]

    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors,
            node_size=800, width=2, font_weight='bold')
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf

def animate_route(route):
    G = nx.Graph()
    edges = [
        ('A', 'B'), ('B', 'C'), ('B', 'F'), ('C', 'G'),
        ('F', 'J'), ('G', 'H'), ('H', 'D'), ('H', 'L'),
        ('J', 'I'), ('J', 'K'), ('K', 'L'), ('I', 'E'),
    ]
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=42)

    progress_bar = st.progress(0)
    status_text = st.empty()
    image_slot = st.empty()

    total_steps = len(route)
    for i in range(1, total_steps + 1):
        current_route = route[:i]
        route_edges = list(zip(current_route, current_route[1:]))

        fig, ax = plt.subplots(figsize=(7, 5))
        nx.draw(G, pos, with_labels=True, node_size=800, font_weight='bold',
                node_color=['green' if node == route[0] else
                            'red' if node == route[-1] else
                            'orange' if node in current_route else 'lightgray'
                            for node in G.nodes()],
                edge_color=['blue' if (e in route_edges or (e[1], e[0]) in route_edges) else 'gray' for e in G.edges()],
                width=2)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        status_text.text(f"\u00c9tape {i}/{total_steps} : {current_route[-1]}")
        image_slot.image(buf)
        progress_bar.progress(i / total_steps)
        time.sleep(0.7)
def dijkstra_route(start, end):
    G = nx.Graph()
    edges = [
        ('A', 'B'), ('B', 'C'), ('B', 'F'), ('C', 'G'),
        ('F', 'J'), ('G', 'H'), ('H', 'D'), ('H', 'L'),
        ('J', 'I'), ('J', 'K'), ('K', 'L'), ('I', 'E'),
    ]
    G.add_edges_from(edges)
    shortest_path = nx.shortest_path(G, source=start, target=end)
    travel_time = calculate_travel_time(shortest_path)
    save_route_to_csv(start, end, shortest_path, travel_time_seconds=travel_time[2], filename="dijkstra_routes.csv")
    return shortest_path
#Dijkstra avec etape intermediaire
def dijkstra_best_route(start, mid, end):
    G = nx.Graph()
    edges = [
        ('A', 'B'), ('B', 'C'), ('B', 'F'), ('C', 'G'),
        ('F', 'J'), ('G', 'H'), ('H', 'D'), ('H', 'L'),
        ('J', 'I'), ('J', 'K'), ('K', 'L'), ('I', 'E'),
    ]
    G.add_edges_from(edges)

    path1 = nx.shortest_path(G, source=start, target=mid)
    path2 = nx.shortest_path(G, source=mid, target=end)[1:]
    full_path = path1 + path2

    travel_time = calculate_travel_time(full_path)
    save_route_to_csv(start, end, full_path, travel_time_seconds=travel_time[2], filename="dijkstra_routes.csv")
    return full_path

def plot_comparison_chart(q_time, d_time):
    labels = ['Q-learning', 'Dijkstra']
    values = [q_time, d_time]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=['orange', 'skyblue'])
    ax.set_ylabel("Temps (secondes)")
    ax.set_title("Comparaison des temps estimés")

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, f'{int(yval)}s', ha='center', va='bottom', fontsize=10)

    st.pyplot(fig)
def save_comparison_to_csv(start, end, method, route, travel_time, filename="comparison_routes.csv"):
    try:
        with open(filename, 'r'):
            pass
    except FileNotFoundError:
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Timestamp", "Méthode", "Point de départ", "Point d’arrivée",
                "Trajet", "Nombre d'étapes", "Temps total (s)"
            ])

    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        route_str = " -> ".join(route)
        writer.writerow([
            timestamp, method, start, end, route_str, len(route) - 1, travel_time
        ])

# Interface utilisateur Streamlit
st.set_page_config(page_title="Optimisation Entrepôt", layout="centered")

# --- TITRE PRINCIPAL ---
st.title("OPTIMISATION DES FLUX DANS UN ENTREPÔT")
st.subheader("Trajets intelligents grâce à l’apprentissage par renforcement")

# --- DESCRIPTION DU PROJET (optionnelle) ---
with st.expander("Description du projet (cliquez pour afficher)"):
    st.markdown("""
    Cette application utilise l'intelligence artificielle pour optimiser les déplacements à l'intérieur d'un entrepôt.
    Elle repose sur un algorithme de **Q-learning** pour apprendre automatiquement les meilleurs itinéraires.

    #### Objectifs :
    - Réduction du temps de trajet.
    - Optimisation des parcours entre les zones.
    - Visualisation dynamique des trajets.

    #### Fonctions principales :
    - Itinéraires directs ou avec étape intermédiaire.
    - Estimation du temps selon la vitesse et la distance.
    - Affichage graphique & animation du chemin.
    - Enregistrement automatique des trajets.

    > Ce projet est un exemple d’intégration de l’IA dans la logistique intelligente (AGV, robotique, entrepôt automatisé).
    """)

st.markdown("---")

# --- PARAMÈTRES UTILISATEUR ---
st.header("Paramètres de simulation")
col1, col2 = st.columns(2)
with col1:
    time_per_step = st.slider("Temps pour une étape (secondes)", 1, 30, 5)
with col2:
    robot_speed = st.slider("Vitesse du robot (mètres/seconde)", 0.5, 5.0, 1.0, step=0.1)

distance_per_step = 10  # mètres

# --- CHOIX DE L’ITINÉRAIRE ---
st.markdown("### Choix de l’itinéraire")
option = st.radio("Quel type de trajet souhaitez-vous calculer ?", ["Route directe", "Route avec étape intermédiaire", "Comparer Q-learning vs Dijkstra"])

locations = list(location_to_state.keys())

if option == "Route directe":
    col1, col2 = st.columns(2)
    with col1:
        start = st.selectbox("Point de départ", locations, index=0)
    with col2:
        end = st.selectbox("Point d’arrivée", locations, index=1)

    if st.button("Calculer la route optimale"):
        if start != end:
            result = route(start, end)
            st.success(f"✅ Route optimale : **{' → '.join(result)}**")

            minutes, seconds, total = calculate_travel_time(result, time_per_step)
            st.info(f"Temps estimé : **{minutes} min {seconds} s** ({total} sec)")

            total_distance = (len(result) - 1) * distance_per_step
            speed_secs = total_distance / robot_speed
            st.info(f"Durée à {robot_speed} m/s : **{int(speed_secs//60)} min {int(speed_secs%60)} s** pour {total_distance} m")

            st.image(draw_route_graph(result), caption="🗺️ Graphe du trajet")

            if st.button("Lancer l’animation"):
                animate_route(result)
        else:
            st.warning("Le point de départ et d’arrivée doivent être différents.")

elif option == "Route avec étape intermédiaire":
    col1, col2, col3 = st.columns(3)
    with col1:
        start = st.selectbox("Départ", locations, key="start")
    with col2:
        mid = st.selectbox("⏸Étape intermédiaire", locations, key="mid")
    with col3:
        end = st.selectbox("Arrivée", locations, key="end")

    if st.button("Calculer la meilleure route avec étape"):
        if len({start, mid, end}) == 3:
            result = best_route(start, end, mid)
            st.success(f"✅ Meilleure route via {mid} : **{' → '.join(result)}**")

            minutes, seconds, total = calculate_travel_time(result, time_per_step)
            st.info(f"Temps estimé : **{minutes} min {seconds} s** ({total} sec)")

            total_distance = (len(result) - 1) * distance_per_step
            speed_secs = total_distance / robot_speed
            st.info(f"Durée à {robot_speed} m/s : **{int(speed_secs//60)} min {int(speed_secs%60)} s** pour {total_distance} m")

            st.image(draw_route_graph(result), caption="Graphe du trajet")

            if st.button("Lancer l’animation"):
                animate_route(result)
        else:
            st.warning("Les trois points doivent être différents.")
            
elif option == "Comparer Q-learning vs Dijkstra": 
    col1, col2 = st.columns(2)
    with col1:
        start = st.selectbox("Départ", locations, key="comp_start")
    with col2:
        end = st.selectbox("Arrivée", locations, key="comp_end")

    if st.button("Comparer les algorithmes"):
        if start != end:
            st.subheader("Q-learning")
            q_route = route(start, end)
            q_minutes, q_seconds, q_total = calculate_travel_time(q_route, time_per_step)
            st.success(f"Q-learning : **{' → '.join(q_route)}**")
            st.info(f"Temps estimé : {q_minutes} min {q_seconds} s")
            st.image(draw_route_graph(q_route), caption="Trajet Q-learning")

            st.subheader("Dijkstra")
            d_route = dijkstra_route(start, end)
            d_minutes, d_seconds, d_total = calculate_travel_time(d_route, time_per_step)
            st.success(f"Dijkstra : **{' → '.join(d_route)}**")
            st.info(f"Temps estimé : {d_minutes} min {d_seconds} s")
            st.image(draw_route_graph(d_route), caption="Trajet Dijkstra")

            if len(q_route) < len(d_route):
                st.success("Q-learning a trouvé un chemin plus court en nombre d'étapes.")
            elif len(q_route) > len(d_route):
                st.info("Dijkstra a trouvé un chemin plus court en nombre d'étapes.")
            else:
                st.warning("Les deux algorithmes ont trouvé un chemin de même longueur.")
        
        else:
            st.warning("Le point de départ et d’arrivée doivent être différents.")
            # Graphe comparatif des durées
    st.markdown("---")
    st.markdown("### Comparaison avec étape intermédiaire")

    col1, col2, col3 = st.columns(3)
    with col1:
        start_cmp = st.selectbox("Départ", locations, key="cmp_start")
    with col2:
        mid_cmp = st.selectbox("Étape", locations, key="cmp_mid")
    with col3:
        end_cmp = st.selectbox("Arrivée", locations, key="cmp_end")

    if st.button("Comparer les deux méthodes avec étape"):
        if len({start_cmp, mid_cmp, end_cmp}) == 3:
            st.subheader("Q-learning avec étape")
            q_route = best_route(start_cmp, end_cmp, mid_cmp)
            q_minutes, q_seconds, q_total = calculate_travel_time(q_route, time_per_step)
            st.success(f"Q-learning : **{' → '.join(q_route)}**")
            st.info(f"Temps estimé : {q_minutes} min {q_seconds} s")
            st.image(draw_route_graph(q_route), caption="Trajet Q-learning")
            save_comparison_to_csv(start_cmp, end_cmp, "Q-learning", q_route, q_total)

            st.subheader("Dijkstra avec étape")
            d_route = dijkstra_best_route(start_cmp, mid_cmp, end_cmp)
            d_minutes, d_seconds, d_total = calculate_travel_time(d_route, time_per_step)
            st.success(f"Dijkstra : **{' → '.join(d_route)}**")
            st.info(f"Temps estimé : {d_minutes} min {d_seconds} s")
            st.image(draw_route_graph(d_route), caption="Trajet Dijkstra")
            save_comparison_to_csv(start_cmp, end_cmp, "Dijkstra", d_route, d_total)
            plot_comparison_chart(q_total, d_total)

            if len(q_route) < len(d_route):
                st.success("Q-learning a trouvé un chemin plus court.")
            elif len(q_route) > len(d_route):
                st.info("Dijkstra a trouvé un chemin plus court.")
            else:
                st.warning("Les deux ont trouvé un chemin de même longueur.")
        else:
            st.warning("Les trois points doivent être différents.")
            #plot_comparison_chart(q_total, d_total)
             
# --- VISUEL ENTREPÔT ---
if st.checkbox("Afficher la photo de l’environnement d’étude"):
    st.image("photo-entrepot.png", caption="Photo de l’environnement d’étude",use_container_width=True)
    st.markdown("""
        <style>
        .zoom-container img:hover {
            transform: scale(1.5);
            z-index: 100;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<p style='font-style: italic; text-align: center;'>Survolez pour zoomer</p>", unsafe_allow_html=True)


# --- PIED DE PAGE ---
st.markdown("---")
st.caption("© 2025 - IA développée par Dr. MOUALE")
