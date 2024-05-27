"""Create virtual python environment: pip install virtualenv"""
"""Name the virtual environment astro-env: python3 -m venv astro-env"""
"""Activate the virtual environment: source astro-env/bin/activate"""
"""Save the files in the astro-env directory"""
"""Install required libraries: pip install -r requirements.txt"""
"""Run the script using python3 astroselfie.py"""
"""Generated gif will show the nightsky from the perspective of an observer on a planet. """
"""0 = Mercury, 1 = Venus, 2 = Earth, 3 = Mars, 4 = Jupiter, 5 = Saturn, 6 = Uranus, 7 = Neptune, 8 = Pluto, 9 = Sun"""
"""Pick a new planet by changing the number in the plot_sky() function"""
"""The output file will be saved in the same directory as the script"""

"""Import required libraries"""
from skyfield.api import load, Star
from skyfield.data import hipparcos
import matplotlib as matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
rc('animation',html='jshtml')
import matplotlib.animation as animation
from datetime import datetime

"""Get current date and time"""
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
current_year = datetime.now().strftime("%Y")

"""Convert datetime obj to string"""
str_current_datetime = str(current_datetime)

"""Load star data and get the 100 brightest stars"""
with load.open(hipparcos.URL) as f:
    df = hipparcos.load_dataframe(f)
bright_stars = df.sort_values('magnitude').head(100)

"""Load planetary ephemeris and get current year"""
planets = load('de440s.bsp')
years = [int(current_year)]

"""Create a figure object"""
fig, ax = plt.subplots(figsize=(12.8,7.2))

"""Create planet names and colors"""
planet_names = ['MERCURY BARYCENTER', 'VENUS BARYCENTER', 'EARTH BARYCENTER', 'MARS BARYCENTER', 'JUPITER BARYCENTER', 'SATURN BARYCENTER', 'URANUS BARYCENTER', 'NEPTUNE BARYCENTER', 'PLUTO BARYCENTER', 'SUN']
planet_colors = matplotlib.colormaps['tab20'].colors[:len(planet_names)]

"""Define a dictionary for solar object radii in meters"""
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

"""Define the number of sols per planet year"""
sols_per_planet_year = [88, 225, 365, 687, 4333, 10759, 30687, 60190, 90520, 365]

"""Create a timescale object and get the current time"""
ts = load.timescale()
t = ts.now()

def planets_obs_matrix(planetarg, yeararg):
    """Get the observations of all planets from a given planet for a given year

    Args:
        planetarg (int): The index of the planet from which the observations are made
        yeararg (int): The year for which the observations are made

    Returns:
        list: The list of observations for all planets from the given planet for the given year
    """
    daily_obs = []
    days = sols_per_planet_year[planetarg]
    planet_obs_from = planets[planet_names[planetarg]]

    for day in range(1, days + 1):
        t_day = ts.utc(yeararg, 1, day)

        for j in range(len(planet_names)):
            if j != planetarg:
                planet = planets[planet_names[j]]
                observation = planet_obs_from.at(t_day).observe(planet).apparent()
                ra, dec, _ = observation.radec()
                obs_list = [yeararg, day, j, ra.hours, dec.degrees]
                daily_obs.append(obs_list)

    return daily_obs

def plot_orbit(planet_name, years, days_per_year, ax, planet_obs_from, ts):
    """Plot the orbit path of a planet

    Args:
        planet_name (string): The name of the planet
        years (int): The years for which the orbit is plotted
        days_per_year (int): The number of days in a year
        ax (Axes): The axis object containing right ascension and declination for each observed planet
        planet_obs_from (int): The index of the planet from which the observations are made
        ts (Time): The timescale object
    """
    orbit_points = []

    for year in years:
        for day in range(1, days_per_year + 1):
            t_day = ts.utc(year, 1, day)
            astrometric = planet_obs_from.at(t_day).observe(planets[planet_name])
            ra, dec, _ = astrometric.radec()
            orbit_points.append((ra.hours, dec.degrees))

    orbit_points = list(zip(*orbit_points))
    ax.plot(orbit_points[0], orbit_points[1], color='gray', linestyle='dashed', linewidth=0.5)

def update_plot(frame, all_obs, ax, planet_colors, days, planet_obs_from, planet_observer):
    """Update the plot with the observations of all planets from a given planet for a given year

    Args:
        frame (int): The frame number
        all_obs (list): The list of all observations
        ax (Axes): The axis object containing right ascension and declination for each observed planet
        planet_colors (string): The colors of the planets
        days (int): The number of days in a year
        planet_obs_from (int): The name of the planet from which the observations are made
        planet_observer (int): The index of the planet from which the observations are made
    """
    ax.clear()
    ax.set_ylim(-34, 34)
    ax.set_xlim(0, 24)
    ax.set_xlabel('Right Ascension (hours)')
    ax.set_ylabel('Declination (degrees)')

    year = all_obs[frame // days][0]
    day = frame % days
    
    """The list to store the planets that have been plotted"""
    plotted_planets = [] 

    for observation in all_obs:
        if observation[0] == year and observation[1] == day:
            _, _, planet_index, ra, dec = observation
            planet_radius = planet_radii[planet_names[planet_index]]
            scaled_size = 4.5 * (planet_radius / planet_radii['EARTH BARYCENTER'])  # Adjust scale based on your preference
            planet_marker = ax.scatter(ra, dec, s=scaled_size, color=planet_colors[planet_index])
            plotted_planets.append((planet_marker, planet_names[planet_index][:20]))
            ax.annotate(planet_names[planet_index][:7], (ra, dec), textcoords="offset points", xytext=(10, 10), ha='center')

    for planet_name in planet_names:
        plot_orbit(planet_name, years, days, ax, planets[planet_names[planet_observer]], ts)

    for _, star_row in bright_stars.iterrows():
        star = Star.from_dataframe(star_row)
        astrometric = planet_obs_from.at(t).observe(star)
        ra, dec, _ = astrometric.radec()
        ax.scatter(ra.hours, dec.degrees, color='black', s=20)

    ax.set_title(f'From {planet_names[planet_observer]} in {years[0]}')
    
    if plotted_planets:
        markers, labels = zip(*plotted_planets)
        ax.legend(markers, labels, loc='upper right', fontsize='small')

def plot_sky(planet_observer):
    """Plot the sky from the perspective of an observer on a planet

    Args:
        planet_observer (int): Index of the planet from which the observations are made
    """
    days = sols_per_planet_year[planet_observer]
    planet_name = planet_names[planet_observer]
    planet_obs_from = planets[planet_name]

    """ Load star data """
    with load.open(hipparcos.URL) as f:
        df = hipparcos.load_dataframe(f)
        df = df[df['magnitude'] <= 5.0]
        bright_stars = df  
    all_obs = []
    for year in years:
        daily_obs = planets_obs_matrix(planet_observer, year)
        all_obs.extend(daily_obs)
    
    """Create an animation object"""
    ani = animation.FuncAnimation(fig, update_plot, frames=range(1, len(years) * days), fargs=(all_obs, ax, planet_colors, days, planet_obs_from, planet_observer), repeat=True, interval=200)

    """Save the animation object as a gif file"""
    ani.save(str_current_datetime+'_'+str(planet_names[planet_observer])+'_astroselfie.gif')

plot_sky(0)
print("Done.")
