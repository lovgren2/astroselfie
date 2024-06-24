import io
from PIL import Image
import os
import multiprocessing
from skyfield.api import load, Star
from skyfield.data import hipparcos
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from concurrent import futures
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Get the current date and time to create a unique file name for the GIF file
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
current_year = int(datetime.now().strftime("%Y"))

# Load star data and get the 100 brightest stars
with load.open(hipparcos.URL) as f:
    df = hipparcos.load_dataframe(f)
bright_stars = df.sort_values('magnitude').head(100)

# Load planetary ephemeris
planets = load('de440s.bsp')

# Create planet names and colors
planet_names = ['MERCURY BARYCENTER', 'VENUS BARYCENTER', 'EARTH BARYCENTER', 'MARS BARYCENTER', 'JUPITER BARYCENTER', 'SATURN BARYCENTER', 'URANUS BARYCENTER', 'NEPTUNE BARYCENTER', 'PLUTO BARYCENTER', 'SUN']
planet_colors = matplotlib.colormaps['tab20'].colors[:len(planet_names)]

# Define a dictionary for solar system object radii in meters
planet_radii = {
    'MERCURY BARYCENTER': 2439.7e3,
    'VENUS BARYCENTER': 6051.8e3,
    'EARTH BARYCENTER': 6371e3,
    'MARS BARYCENTER': 3389.5e3,
    'JUPITER BARYCENTER': 69911e3,
    'SATURN BARYCENTER': 58232e3,
    'URANUS BARYCENTER': 25362e3,
    'NEPTUNE BARYCENTER': 24622e3,
    'PLUTO BARYCENTER': 1188.3e3,
    'SUN': 6.9634e8
}

# Number of sols per year on each planet
sols_per_planet_year = [88, 225, 365, 687, 4333, 10759, 30687, 60190, 90520, 365]

# Get the current time
ts = load.timescale()
t = ts.now()

def get_planet_observation(planet, year, ts):
    days = sols_per_planet_year[planet]
    planet_obs_from = planets[planet_names[planet]]
    all_obs = []

    for day in range(days):
        t_day = ts.utc(year, 1, day + 1)
        for j in range(len(planet_names)):
            if j != planet:
                planet_to_observe = planets[planet_names[j]]
                observation = planet_obs_from.at(t_day).observe(planet_to_observe).apparent()
                ra, dec, _ = observation.radec()
                obs_tuple = (year, day, j, ra.hours, dec.degrees)
                all_obs.append(obs_tuple)

    return all_obs

def plot_orbit(planet_name, year, days_per_year, ax, planet_obs_from, ts):
    orbit_points = []
    for day in range(days_per_year):
        t_day = ts.utc(year, 1, day + 1)
        astrometric = planet_obs_from.at(t_day).observe(planets[planet_name])
        ra, dec, _ = astrometric.radec()
        orbit_points.append((ra.hours, dec.degrees))
    filtered_orbit_points = []
    prev_ra = None
    for ra, dec in orbit_points:
        if prev_ra is not None and abs(ra - prev_ra) > 12:
            filtered_orbit_points.append((None, None))
        filtered_orbit_points.append((ra, dec))
        prev_ra = ra
    filtered_orbit_points = list(zip(*filtered_orbit_points))
    ax.plot(filtered_orbit_points[0], filtered_orbit_points[1], color='gray', linestyle='dashed', linewidth=0.5)

def update_plot(frame, all_obs, ax, planet_colors, days, planet_obs_from, planet_observer):
    ax.clear()
    ax.set_ylim(-34, 34)
    ax.set_xlim(-1, 25)
    ax.set_xlabel('Right Ascension (hours)')
    ax.set_ylabel('Declination (degrees)')

    year = all_obs[frame * (len(planet_names) - 1)][0]
    day = all_obs[frame * (len(planet_names) - 1)][1]
    plotted_planets = []

    # Plot planets
    for i in range(len(planet_names) - 1):
        observation = all_obs[frame * (len(planet_names) - 1) + i]
        _, _, planet_index, ra, dec = observation
        planet_radius = planet_radii[planet_names[planet_index]]
        scaled_size = 4.5 * (planet_radius / planet_radii['EARTH BARYCENTER'])
        planet_marker = ax.scatter(ra, dec, s=scaled_size, color=planet_colors[planet_index])
        plotted_planets.append((planet_marker, planet_names[planet_index][:20]))
        ax.annotate(planet_names[planet_index][:7], (ra, dec), textcoords="offset points", xytext=(10, 10), ha='center')

    # Plot orbits
    for planet_name in planet_names:
        plot_orbit(planet_name, current_year, days, ax, planet_obs_from, ts)

    # Plot stars
    for _, star_row in bright_stars.iterrows():
        star = Star.from_dataframe(star_row)
        astrometric = planet_obs_from.at(ts.utc(year, 1, day + 1)).observe(star)
        ra, dec, _ = astrometric.radec()
        magnitude = star_row['magnitude']
        base_size = 25
        scaled_size = base_size * 10 ** (-magnitude)
        ax.scatter(ra.hours, dec.degrees, color='black', s=scaled_size)

    ax.set_title(f'From {planet_names[planet_observer]} on Day {day + 1} in {year}')

    if plotted_planets:
        markers, labels = zip(*plotted_planets)
        ax.legend(markers, labels, loc='upper right', fontsize='small')

def generate_frame(args):
    frame, all_obs, planet_colors, days, planet_data = args
    planet_name, ts, planet_observer = planet_data
    planet_obs_from = planets[planet_name]

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    update_plot(frame, all_obs, ax, planet_colors, days, planet_obs_from, planet_observer)
    
    img_folder = 'img'
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    
    file_path = os.path.join(img_folder, f"frame_{frame:04d}.png")
    plt.savefig(file_path, format='png')
    plt.close(fig)
    return file_path

def plot_sky_parallel(planet_observer):
    days = sols_per_planet_year[planet_observer]
    planet_name = planet_names[planet_observer]
    planet_data = (planet_name, ts, planet_observer)

    all_obs = get_planet_observation(planet_observer, current_year, ts)
    all_obs_tuple = tuple(all_obs)

    with futures.ProcessPoolExecutor() as executor:
        frames_paths = list(executor.map(generate_frame, [(frame, all_obs_tuple, planet_colors, days, planet_data) for frame in range(days)]))
    
    frames = [Image.open(path) for path in frames_paths]
    frames[0].save(f'{current_datetime}_{planet_name}_astroselfie.gif', 
                   save_all=True, 
                   append_images=frames[1:], 
                   optimize=False, 
                   duration=200, 
                   loop=0)
    logging.info(f"Created file: {current_datetime}_{planet_name}_astroselfie.gif")

    # Cleanup: Delete the img files and folder
    for path in frames_paths:
        os.remove(path)
    os.rmdir('img')
    logging.info("Removed temporary folder: img")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    planet_observer = 0  # 0 for Mercury, change this to observe from different planets
    plot_sky_parallel(planet_observer)
    logging.info("Done.")