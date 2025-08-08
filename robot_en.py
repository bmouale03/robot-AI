import streamlit as st
import numpy as np
import csv
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
import io
import time
import base64
from PIL import Image
import pyautogui
import tempfile

# === AJOUTS POUR LE TEMPS D'EXÉCUTION & SYSTÈME ===
import platform
import psutil
import time as t  # renommé pour éviter conflit avec 'time.sleep'
start_time = t.time()

# Q-learning parameters
gamma = 0.75
alpha = 0.9

# States
location_to_state = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11
}
state_to_location = {state: location for location, state in location_to_state.items()}
# === NOUVELLE FONCTION SYSTEM INFO ===
#def get_system_info():
    #"""Récupère les informations système de la machine"""
    #info = {
        #"Version OS": platform.version(),
       # "Architecture": platform.machine(),
        #"Processeur": platform.processor(),
        #"Cœurs logiques": psutil.cpu_count(logical=True),
        #"Cœurs physiques": psutil.cpu_count(logical=False),
        #"RAM totale (Go)": round(psutil.virtual_memory().total / (1024**3), 2),
        #"Utilisation RAM (%)": psutil.virtual_memory().percent,
        #"Utilisation CPU (%)": psutil.cpu_percent(interval=1),
    #}
    #return info
# =======================================
# Function to capture screen

# === NEW: Get system info function ===
def get_system_info():
    """Returns system specifications"""
    info = {
        "Operating System": platform.system(),
        "OS Version": platform.version(),
        "Architecture": platform.machine(),
        "Processor": platform.processor(),
        "Logical Cores": psutil.cpu_count(logical=True),
        "Physical Cores": psutil.cpu_count(logical=False),
        "Total RAM (GB)": round(psutil.virtual_memory().total / (1024**3), 2),
        "RAM Usage (%)": psutil.virtual_memory().percent,
        "CPU Usage (%)": psutil.cpu_percent(interval=1),
    }
    return info
# =====================================
def capture_interface():
    """Capture the entire screen and return the image as bytes"""
    try:
        # Take screenshot
        screenshot = pyautogui.screenshot()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            screenshot.save(tmpfile.name, "PNG")
            with open(tmpfile.name, "rb") as f:
                img_bytes = f.read()
        
        return img_bytes
    except Exception as e:
        st.error(f"Error during capture: {e}")
        return None

# Function to create download link
def get_binary_download_link(bin_data, file_label, file_name):
    """Generate a link to download a binary file"""
    bin_str = base64.b64encode(bin_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{file_name}">{file_label}</a>'
    return href

# Utility function to create image download link
def get_image_download_link(img_buffer, filename="graph.png", text="Download image"):
    """Generate a link to download the image"""
    b64 = base64.b64encode(img_buffer.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

# CSV logging
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
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf, fig

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

        status_text.text(f"Step {i}/{total_steps}: {current_route[-1]}")
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
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Estimated Time Comparison")

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, f'{int(yval)}s', ha='center', va='bottom', fontsize=10)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def save_comparison_to_csv(start, end, method, route, travel_time, filename="comparison_routes.csv"):
    try:
        with open(filename, 'r'):
            pass
    except FileNotFoundError:
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Timestamp", "Method", "Start Point", "End Point",
                "Route", "Number of Steps", "Total Time (s)"
            ])

    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        route_str = " -> ".join(route)
        writer.writerow([
            timestamp, method, start, end, route_str, len(route) - 1, travel_time
        ])

# Streamlit user interface
st.set_page_config(page_title="Warehouse Optimization", layout="centered")

# --- MAIN TITLE ---
st.title("WAREHOUSE FLOW OPTIMIZATION")
st.subheader("Smart routing through reinforcement learning")

# --- PROJECT DESCRIPTION ---
with st.expander("Project description (click to expand)"):
    st.markdown("""
    This application uses artificial intelligence to optimize movements within a warehouse.
    It relies on a **Q-learning** algorithm to automatically learn the best routes.

    #### Objectives:
    - Reduce travel time
    - Optimize paths between zones
    - Dynamic route visualization

    #### Main features:
    - Direct routes or with intermediate stops
    - Time estimation based on speed and distance
    - Graphical display & route animation
    - Automatic route logging

    > This project demonstrates AI integration in smart logistics (AGV, robotics, automated warehouse).
    """)

st.markdown("---")

# --- USER PARAMETERS ---
st.header("Simulation parameters")
col1, col2 = st.columns(2)
with col1:
    time_per_step = st.slider("Time per step (seconds)", 1, 30, 5)
with col2:
    robot_speed = st.slider("Robot speed (meters/second)", 0.5, 5.0, 1.0, step=0.1)

distance_per_step = 10  # meters

# --- ROUTE SELECTION ---
st.markdown("### Route selection")
option = st.radio("What type of route would you like to calculate?", 
                 ["Direct route", "Route with intermediate stop", "Compare Q-learning vs Dijkstra"])

locations = list(location_to_state.keys())

if option == "Direct route":
    col1, col2 = st.columns(2)
    with col1:
        start = st.selectbox("Starting point", locations, index=0)
    with col2:
        end = st.selectbox("Destination", locations, index=1)

    if st.button("Calculate optimal route"):
        if start != end:
            result = route(start, end)
            st.success(f"✅ Optimal route: **{' → '.join(result)}**")

            minutes, seconds, total = calculate_travel_time(result, time_per_step)
            st.info(f"Estimated time: **{minutes} min {seconds} s** ({total} sec)")

            total_distance = (len(result) - 1) * distance_per_step
            speed_secs = total_distance / robot_speed
            st.info(f"Duration at {robot_speed} m/s: **{int(speed_secs//60)} min {int(speed_secs%60)} s** for {total_distance} m")

            graph_buffer, _ = draw_route_graph(result)
            st.image(graph_buffer, caption="🗺️ Route graph")
            st.markdown(get_image_download_link(graph_buffer, 
                                              filename=f"route_{start}_to_{end}.png",
                                              text="📥 Download this graph"),
                        unsafe_allow_html=True)

            if st.button("Launch animation"):
                animate_route(result)
        else:
            st.warning("Starting point and destination must be different.")

elif option == "Route with intermediate stop":
    col1, col2, col3 = st.columns(3)
    with col1:
        start = st.selectbox("Start", locations, key="start")
    with col2:
        mid = st.selectbox("⏸Intermediate stop", locations, key="mid")
    with col3:
        end = st.selectbox("End", locations, key="end")

    if st.button("Calculate best route with stop"):
        if len({start, mid, end}) == 3:
            result = best_route(start, end, mid)
            st.success(f"✅ Best route via {mid}: **{' → '.join(result)}**")

            minutes, seconds, total = calculate_travel_time(result, time_per_step)
            st.info(f"Estimated time: **{minutes} min {seconds} s** ({total} sec)")

            total_distance = (len(result) - 1) * distance_per_step
            speed_secs = total_distance / robot_speed
            st.info(f"Duration at {robot_speed} m/s: **{int(speed_secs//60)} min {int(speed_secs%60)} s** for {total_distance} m")

            graph_buffer, _ = draw_route_graph(result)
            st.image(graph_buffer, caption="Route graph")
            st.markdown(get_image_download_link(graph_buffer,
                                              filename=f"route_{start}_via_{mid}_to_{end}.png",
                                              text="📥 Download this graph"),
                        unsafe_allow_html=True)

            if st.button("Launch animation"):
                animate_route(result)
        else:
            st.warning("All three points must be different.")
            
elif option == "Compare Q-learning vs Dijkstra": 
    col1, col2 = st.columns(2)
    with col1:
        start = st.selectbox("Start", locations, key="comp_start")
    with col2:
        end = st.selectbox("End", locations, key="comp_end")

    if st.button("Compare algorithms"):
        if start != end:
            st.subheader("Q-learning")
            q_route = route(start, end)
            q_minutes, q_seconds, q_total = calculate_travel_time(q_route, time_per_step)
            st.success(f"Q-learning: **{' → '.join(q_route)}**")
            st.info(f"Estimated time: {q_minutes} min {q_seconds} s")
            
            q_graph_buffer, _ = draw_route_graph(q_route)
            st.image(q_graph_buffer, caption="Q-learning route")
            st.markdown(get_image_download_link(q_graph_buffer,
                                              filename=f"qlearning_{start}_to_{end}.png",
                                              text="📥 Download Q-learning graph"),
                        unsafe_allow_html=True)

            st.subheader("Dijkstra")
            d_route = dijkstra_route(start, end)
            d_minutes, d_seconds, d_total = calculate_travel_time(d_route, time_per_step)
            st.success(f"Dijkstra: **{' → '.join(d_route)}**")
            st.info(f"Estimated time: {d_minutes} min {d_seconds} s")
            
            d_graph_buffer, _ = draw_route_graph(d_route)
            st.image(d_graph_buffer, caption="Dijkstra route")
            st.markdown(get_image_download_link(d_graph_buffer,
                                              filename=f"dijkstra_{start}_to_{end}.png",
                                              text="📥 Download Dijkstra graph"),
                        unsafe_allow_html=True)

            if len(q_route) < len(d_route):
                st.success("Q-learning found a shorter path in number of steps.")
            elif len(q_route) > len(d_route):
                st.info("Dijkstra found a shorter path in number of steps.")
            else:
                st.warning("Both algorithms found paths of the same length.")
        
        else:
            st.warning("Starting point and destination must be different.")

    st.markdown("---")
    st.markdown("### Comparison with intermediate stop")

    col1, col2, col3 = st.columns(3)
    with col1:
        start_cmp = st.selectbox("Start", locations, key="cmp_start")
    with col2:
        mid_cmp = st.selectbox("Stop", locations, key="cmp_mid")
    with col3:
        end_cmp = st.selectbox("End", locations, key="cmp_end")

    if st.button("Compare both methods with stop"):
        if len({start_cmp, mid_cmp, end_cmp}) == 3:
            st.subheader("Q-learning with stop")
            q_route = best_route(start_cmp, end_cmp, mid_cmp)
            q_minutes, q_seconds, q_total = calculate_travel_time(q_route, time_per_step)
            st.success(f"Q-learning: **{' → '.join(q_route)}**")
            st.info(f"Estimated time: {q_minutes} min {q_seconds} s")
            
            q_graph_buffer, _ = draw_route_graph(q_route)
            st.image(q_graph_buffer, caption="Q-learning route")
            st.markdown(get_image_download_link(q_graph_buffer,
                                              filename=f"qlearning_{start_cmp}_via_{mid_cmp}_to_{end_cmp}.png",
                                              text="📥 Download Q-learning graph"),
                        unsafe_allow_html=True)
            save_comparison_to_csv(start_cmp, end_cmp, "Q-learning", q_route, q_total)

            st.subheader("Dijkstra with stop")
            d_route = dijkstra_best_route(start_cmp, mid_cmp, end_cmp)
            d_minutes, d_seconds, d_total = calculate_travel_time(d_route, time_per_step)
            st.success(f"Dijkstra: **{' → '.join(d_route)}**")
            st.info(f"Estimated time: {d_minutes} min {d_seconds} s")
            
            d_graph_buffer, _ = draw_route_graph(d_route)
            st.image(d_graph_buffer, caption="Dijkstra route")
            st.markdown(get_image_download_link(d_graph_buffer,
                                              filename=f"dijkstra_{start_cmp}_via_{mid_cmp}_to_{end_cmp}.png",
                                              text="📥 Download Dijkstra graph"),
                        unsafe_allow_html=True)
            save_comparison_to_csv(start_cmp, end_cmp, "Dijkstra", d_route, d_total)

            comparison_buffer = plot_comparison_chart(q_total, d_total)
            st.image(comparison_buffer, caption="Time comparison")
            st.markdown(get_image_download_link(comparison_buffer,
                                              filename=f"comparison_{start_cmp}_via_{mid_cmp}_to_{end_cmp}.png",
                                              text="📥 Download comparison chart"),
                        unsafe_allow_html=True)

            if len(q_route) < len(d_route):
                st.success("Q-learning found a shorter path.")
            elif len(q_route) > len(d_route):
                st.info("Dijkstra found a shorter path.")
            else:
                st.warning("Both found paths of the same length.")
        else:
            st.warning("All three points must be different.")
             
# --- WAREHOUSE VISUAL ---
if st.checkbox("Show study environment photo"):
    st.image("photo-entrepot.png", caption="Study environment photo", use_container_width=True)
    st.markdown("""
        <style>
        .zoom-container img:hover {
            transform: scale(1.5);
            z-index: 100;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<p style='font-style: italic; text-align: center;'>Hover to zoom</p>", unsafe_allow_html=True)

# --- SCREENSHOT ---
st.markdown("---")
st.markdown("### Full interface screenshot")

if st.button("📸 Capture full interface"):
    with st.spinner("Capturing..."):
        screenshot = capture_interface()
        
        if screenshot:
            st.success("Capture successful!")
            
            # Show preview
            img = Image.open(io.BytesIO(screenshot))
            st.image(img, caption="Screenshot preview", width=400)
            
            # Download button
            st.markdown(
                get_binary_download_link(
                    screenshot,
                    "⬇️ Download screenshot",
                    f"warehouse_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                ),
                unsafe_allow_html=True
            )
        else:
            st.error("Capture failed")
# === NOUVELLE SECTION TEMPS & SYSTÈME ===
#st.markdown("---")
#st.header("🔍 Détails techniques de l'exécution")

#if st.button("Afficher les infos système et le temps d'exécution"):
    #end_time = t.time()
    #exec_duration = round(end_time - start_time, 2)
    
    #st.info(f"⏱️ Temps écoulé depuis le lancement : **{exec_duration} secondes**")

    #sys_info = get_system_info()
    #st.markdown("### 🖥️ Informations système :")
    #for key, value in sys_info.items():
        #st.write(f"- **{key}** : {value}")
# =========================================
# === SYSTEM INFO SECTION ===
st.markdown("---")
st.header("🔍 Technical Details")

if st.button("Show system info & execution time"):
    end_time = t.time()
    exec_duration = round(end_time - start_time, 2)

    st.info(f"⏱️ Time since app launch: **{exec_duration} seconds**")

    sys_info = get_system_info()
    st.markdown("### 🖥️ System Information:")
    for key, value in sys_info.items():
        st.write(f"- **{key}**: {value}")
# ===========================
# --- FOOTER ---
st.markdown("---")
st.caption("© 2025 - AI developed by Dr. MOUALE")