import io
from PIL import Image
import os
import tempfile
import multiprocessing
from skyfield.api import load, Star
from skyfield.data import hipparcos
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from concurrent import futures

# Get the current date and time to create a unique file name for the GIF file
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
current_year = datetime.now().strftime("%Y")

# Load star data and get the 100 brightest stars
with load.open(hipparcos.URL) as f:
    df = hipparcos.load_dataframe(f)
bright_stars = df.sort_values('magnitude').head(100)

# Load planetary ephemeris and get the current year
planets = load('de440s.bsp')
years = [int(current_year)]

# Create a figure object
fig, ax = plt.subplots(figsize=(12.8, 7.2))

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

    for day in range(1, days + 1):
        t_day = ts.utc(year, 1, day)
        for j in range(len(planet_names)):
            if j != planet:
                planet_to_observe = planets[planet_names[j]]
                observation = planet_obs_from.at(t_day).observe(planet_to_observe).apparent()
                ra, dec, _ = observation.radec()
                obs_tuple = (year, day, j, ra.hours, dec.degrees)  # Convert observation to tuple
                all_obs.append(obs_tuple)

    return all_obs

def update_plot(frame, all_obs, ax, planet_colors, days, planet_obs_from):
    ax.clear()
    ax.set_ylim(-34, 34)
    ax.set_xlim(1, 24)
    ax.set_xlabel('Right Ascension (hours)')
    ax.set_ylabel('Declination (degrees)')

    year = all_obs[frame // days][0]
    day = frame % days
    plotted_planets = []

    for observation in all_obs:
        if observation[0] == year and observation[1] == day:
            _, _, planet_index, ra, dec = observation
            planet_radius = planet_radii[planet_names[planet_index]]
            scaled_size = 4.5 * (planet_radius / planet_radii['EARTH BARYCENTER'])
            planet_marker = ax.scatter(ra, dec, s=scaled_size, color=planet_colors[planet_index])
            plotted_planets.append((planet_marker, planet_names[planet_index][:20]))
            ax.annotate(planet_names[planet_index][:7], (ra, dec), textcoords="offset points", xytext=(10, 10), ha='center')

    for _, star_row in bright_stars.iterrows():
        star = Star.from_dataframe(star_row)
        astrometric = planet_obs_from.at(t).observe(star)
        ra, dec, _ = astrometric.radec()
        magnitude = star_row['magnitude']
        base_size = 25
        scaled_size = base_size * 10 ** (-magnitude)
        ax.scatter(ra.hours, dec.degrees, color='black', s=scaled_size)

    ax.set_title(f'From {planet_obs_from} in {years[0]}')

    if plotted_planets:
        markers, labels = zip(*plotted_planets)
        ax.legend(markers, labels, loc='upper right', fontsize='small')

def generate_frames(args):
    frame, all_obs, planet_colors, days, planet_data = args
    planet_name, ts = planet_data
    planet_obs_from = planets[planet_name]

    fig, ax = plt.subplots(figsize=(12.8, 7.2))  # Create Axes object inside the process
    update_plot(frame, all_obs, ax, planet_colors, days, planet_obs_from)
    
    # Create the img folder if it doesn't exist
    img_folder = 'img'
    if not os.path.exists(img_folder):
        print("Created temporary folder: img")
        os.makedirs(img_folder)
    
    # Save the figure to the img folder
    file_path = os.path.join(img_folder, f"frame_{frame}.png")
    plt.savefig(file_path, format='png')
    
    plt.close(fig)  # Close the figure to avoid memory leaks
    return file_path

def plot_sky_parallel(planet_observer):
    days = sols_per_planet_year[planet_observer]
    planet_name = planet_names[planet_observer]
    planet_data = (planet_name, ts)

    all_obs = []
    for year in years:
        all_obs.extend(get_planet_observation(planet_observer, year, ts))

    # Convert all_obs to a tuple to ensure picklability
    all_obs_tuple = tuple(all_obs)

    with futures.ProcessPoolExecutor() as executor:
        frames_paths = list(executor.map(generate_frames, [(frame, all_obs_tuple, planet_colors, days, planet_data) for frame in range(len(years) * days)]))
    
    # Load images and create GIF
    frames = [Image.open(path) for path in frames_paths]
    frames[0].save('astroselfie-parallel.gif', save_all=True, append_images=frames[1:], optimize=False, duration=200, loop=0)
    print("Created file: astroselfie-parallel.gif")

    # Cleanup: Delete the img files and folder
    img_folder = 'img'
    for path in frames_paths:
        os.remove(path)
    os.rmdir(img_folder)
    print("Removed temporary folder: img")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    plot_sky_parallel(0)
    print("Done.")