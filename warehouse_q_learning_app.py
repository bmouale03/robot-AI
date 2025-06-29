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

# Interface utilisateur Streamlit
st.set_page_config(page_title="Optimisation Entrepôt", layout="centered")

st.title("OPTIMISATION DES FLUX DANS UN ENTREPÔT ")
st.write("Trouvez les itinéraires optimaux dans votre entrepôt à l’aide de l’intelligence artificielle.")

option = st.radio("Choisissez une option :", ["Route directe", "Route avec étape intermédiaire"])
locations = list(location_to_state.keys())

time_per_step = st.slider("Temps estimé pour parcourir une étape (en secondes)", 1, 30, 5)
robot_speed = st.slider("Vitesse du robot (en mètres/seconde)", 0.5, 5.0, 1.0, step=0.1)
distance_per_step = 10

if option == "Route directe":
    col1, col2 = st.columns(2)
    with col1:
        start = st.selectbox("Point de départ", locations, index=0)
    with col2:
        end = st.selectbox("Point d’arrivée", locations, index=1)

    if st.button("Trouver la route optimale"):
        if start != end:
            result = route(start, end)
            st.success(f"Route optimale : {' -> '.join(result)}")
            minutes, seconds, total = calculate_travel_time(result, time_per_step)
            st.info(f"Temps estimé : {minutes} min {seconds} s ({total} sec)")
            total_distance = (len(result) - 1) * distance_per_step
            speed_secs = total_distance / robot_speed
            st.info(f"Temps à {robot_speed} m/s : {int(speed_secs//60)} min {int(speed_secs%60)} s pour {total_distance} m")
            st.image(draw_route_graph(result))
            if st.button("Lancer l’animation"):
                animate_route(result)
        else:
            st.warning("Le point de départ et d’arrivée doivent être différents.")

elif option == "Route avec étape intermédiaire":
    col1, col2, col3 = st.columns(3)
    with col1:
        start = st.selectbox("Point de départ", locations, key="start")
    with col2:
        mid = st.selectbox("Point intermédiaire", locations, key="mid")
    with col3:
        end = st.selectbox("Point d’arrivée", locations, key="end")

    if st.button("Trouver la meilleure route avec étape"):
        if len({start, mid, end}) == 3:
            result = best_route(start, end, mid)
            st.success(f"Meilleure route via {mid} : {' -> '.join(result)}")
            minutes, seconds, total = calculate_travel_time(result, time_per_step)
            st.info(f"Temps estimé : {minutes} min {seconds} s ({total} sec)")
            total_distance = (len(result) - 1) * distance_per_step
            speed_secs = total_distance / robot_speed
            st.info(f"Temps à {robot_speed} m/s : {int(speed_secs//60)} min {int(speed_secs%60)} s pour {total_distance} m")
            st.image(draw_route_graph(result))
            if st.button("Lancer l’animation"):
                animate_route(result)
        else:
            st.warning("Les trois points doivent être différents.")

if st.checkbox("Afficher la photo de l’environnement d’étude"):
    st.image("photo-entrepot.png", caption="Photo de l’environnement d’étude", use_container_width=True)
    st.markdown("""
        <style>
        .zoom-container img:hover {
            transform: scale(1.5);
            z-index: 100;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<p style='font-style: italic; text-align: center;'>Survolez pour zoomer</p>", unsafe_allow_html=True)

st.markdown("---")
st.caption("© 2025 - IA réalisée par Dr. MOUALE")
