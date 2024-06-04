import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.animation as animation
from skyfield.api import load, Star
from skyfield.data import hipparcos
from datetime import datetime

# Load the Hipparcos star data
with load.open(hipparcos.URL) as f:
    df = hipparcos.load_dataframe(f)

# Get the 25 brightest stars
bright_stars = df.sort_values('magnitude').head(25)

# Load the DE440 planetary ephemeris
planets = load('de440s.bsp')

# Get the current date and year
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
current_year = int(datetime.now().strftime("%Y"))

# Create a list of planet names and colors
planet_names = ['MERCURY BARYCENTER', 'VENUS BARYCENTER', 'EARTH BARYCENTER', 'MARS BARYCENTER', 'JUPITER BARYCENTER', 'SATURN BARYCENTER', 'URANUS BARYCENTER', 'NEPTUNE BARYCENTER', 'PLUTO BARYCENTER', 'SUN']
planet_colors = plt.cm.tab20.colors[:len(planet_names)]

# Radii of the planets in meters
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

# Days per year on each planet
sols_per_planet_year = [88, 225, 365, 687, 4333, 10759, 30687, 60190, 90520, 365]

# Create a timescale object and get the current time
ts = load.timescale()
t = ts.now()

def planets_obs_matrix(planetarg, yeararg):
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
    orbit_points = []
    for year in years:
        for day in range(1, days_per_year + 2):
            t_day = ts.utc(year, 1, day)
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
    for planet_name in planet_names:
        plot_orbit(planet_name, [current_year], days, ax, planets[planet_names[planet_observer]], ts)
    for _, star_row in bright_stars.iterrows():
        star = Star.from_dataframe(star_row)
        astrometric = planet_obs_from.at(t).observe(star)
        ra, dec, _ = astrometric.radec()
        magnitude = star_row['magnitude']
        base_size = 1
        scaled_size = base_size * 10 ** (-magnitude)
        ax.scatter(ra.hours, dec.degrees, color='black', s=scaled_size)
    ax.set_title(f'From {planet_names[planet_observer]} in {current_year}')
    if plotted_planets:
        markers, labels = zip(*plotted_planets)
        ax.legend(markers, labels, loc='upper right', fontsize='small')

def plot_sky(planet_observer):
    days = sols_per_planet_year[planet_observer]
    planet_name = planet_names[planet_observer]
    planet_obs_from = planets[planet_name]
    with load.open(hipparcos.URL) as f:
        df = hipparcos.load_dataframe(f)
        df = df[df['magnitude'] <= 3.0]
        bright_stars = df
    all_obs = []
    for year in [current_year]:
        daily_obs = planets_obs_matrix(planet_observer, year)
        all_obs.extend(daily_obs)
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ani = animation.FuncAnimation(fig, update_plot, frames=range(1, len([current_year]) * days), fargs=(all_obs, ax, planet_colors, days, planet_obs_from, planet_observer), repeat=True, interval=200)
    ani.save(f'{current_datetime}_{planet_name}_astroselfie.gif')

plot_sky(0)  # Use 0 for Mercury, you can change this to any planet index
print("Done.")
