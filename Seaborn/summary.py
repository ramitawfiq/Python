import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.tri import Triangulation
from sklearn.datasets import load_iris, load_digits
from sklearn.manifold import Isomap
from sklearn.gaussian_process import GaussianProcess
from scipy.stats import gaussian_kde
from netCDF4 import Dataset, date2index
from datetime import datetime
import seaborn as sns

# Set global style
plt.style.use('seaborn-whitegrid')

# =========================
# Section 1: 3D Visualizations
# =========================
def plot_3d_visualizations():
    fig = plt.figure(figsize=(15, 10))
    # 3D Line and Scatter Plot
    ax = fig.add_subplot(231, projection='3d')
    ax.set_title('3D Line and Scatter Plot')
    zline = np.linspace(0, 15, 1000)
    xline = np.sin(zline)
    yline = np.cos(zline)
    ax.plot3D(xline, yline, zline, 'gray', label='Spiral Line')
    zdata = 15 * np.random.random(100)
    xdata = np.sin(zdata) + 0.1*np.random.randn(100)
    ydata = np.cos(zdata) + 0.1*np.random.randn(100)
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens', label='Points')
    ax.legend()

    # 3D Contour plot
    ax2 = plt.subplot(232, projection='3d')
    ax2.set_title('3D Contour Plot')
    def f(x, y): return np.sin(np.sqrt(x**2 + y**2))
    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    ax2.contour3D(X, Y, Z, 50, cmap='binary')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.view_init(60, 35)

    # Wireframe surface
    ax3 = plt.subplot(233, projection='3d')
    ax3.set_title('Wireframe Surface')
    ax3.plot_wireframe(X, Y, Z, color='black')

    # Surface plot
    ax4 = plt.subplot(234, projection='3d')
    ax4.set_title('Surface Plot')
    ax4.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    # Partial polar grid for surface
    r = np.linspace(0, 6, 20)
    theta = np.linspace(-0.9*np.pi, 0.8*np.pi, 40)
    R, Theta = np.meshgrid(r, theta)
    X_polar = R * np.sin(Theta)
    Y_polar = R * np.cos(Theta)
    Z_polar = f(X_polar, Y_polar)
    ax5 = plt.subplot(235, projection='3d')
    ax5.set_title('Polar Surface')
    ax5.plot_surface(X_polar, Y_polar, Z_polar, cmap='viridis', edgecolor='none')

    # Triangulation with random points
    theta_rand = 2*np.pi*np.random.random(1000)
    r_rand = 6*np.random.random(1000)
    x_rand = r_rand * np.sin(theta_rand)
    y_rand = r_rand * np.cos(theta_rand)
    z_rand = f(x_rand, y_rand)
    ax6 = plt.subplot(236, projection='3d')
    ax6.set_title('Triangulated Surface')
    ax6.scatter3D(x_rand, y_rand, z_rand, c=z_rand, cmap='viridis', linewidth=0.5)
    tri = Triangulation(x_rand, y_rand)
    ax6.plot_trisurf(x_rand, y_rand, z_rand, triangles=tri.triangles, cmap='viridis', linewidths=0.2)

    plt.tight_layout()
    plt.show()

    # Möbius strip visualization
    fig2 = plt.figure(figsize=(8, 8))
    ax_mobius = fig2.add_subplot(111, projection='3d')
    ax_mobius.set_title('Möbius Strip')
    w = np.linspace(-0.25, 0.25, 8)
    theta = np.linspace(0, 2*np.pi, 30)
    W, Theta = np.meshgrid(w, theta)
    phi = 0.5 * Theta
    r = 1 + W * np.cos(phi)
    x = r * np.cos(Theta)
    y = r * np.sin(Theta)
    z = W * np.sin(phi)
    tri = Triangulation(W.flatten(), Theta.flatten())
    ax_mobius.plot_trisurf(x.flatten(), y.flatten(), z.flatten(),
                           triangles=tri.triangles, cmap='viridis', linewidths=0.2)
    ax_mobius.set_xlim(-1, 1)
    ax_mobius.set_ylim(-1, 1)
    ax_mobius.set_zlim(-1, 1)
    plt.show()

# =========================
# Section 2: Basic Wave Plot
# =========================
def plot_simple_wave():
    x = np.linspace(0, 10, 1000)
    y = np.sin(x)
    plt.figure()
    plt.plot(x, y, color='blue', linestyle='-', label='sin(x)')
    plt.title('A Simple Sine Wave')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.legend()
    plt.show()

# =========================
# Section 3: Geographic Maps & Maps Visualization
# =========================
def plot_maps_and_geography():
    from mpl_toolkits.basemap import Basemap

    # Global map with coastlines
    plt.figure(figsize=(8, 8))
    m = Basemap(projection='ortho', resolution=None, lat_0=50, lon_0=-100)
    m.bluemarble(scale=0.5)
    plt.title("Blue Marble Globe")
    plt.show()

    # Regional map with etopo relief and city marker
    plt.figure(figsize=(8, 8))
    m = Basemap(projection='lcc', resolution=None, width=8E6, height=8E6, lat_0=45, lon_0=-100)
    m.etopo(scale=0.5, alpha=0.5)
    x, y = m(-122.3, 47.6)
    plt.plot(x, y, 'ok', markersize=5)
    plt.text(x, y, ' Seattle', fontsize=12)
    plt.title("Regional Map with Topography and Seattle")
    plt.show()

    # California cities with population and area
    try:
        cities = pd.read_csv('data/california_cities.csv')
        lat = cities['latd'].values
        lon = cities['longd'].values
        population = cities['population_total'].values
        area = cities['area_total_km2'].values
        plt.figure(figsize=(8,8))
        m = Basemap(projection='lcc', resolution='h', lat_0=37.5, lon_0=-119, width=1E6, height=1.2E6)
        m.shadedrelief()
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        m.scatter(lon, lat, latlon=True, c=np.log10(population), s=area, cmap='Reds', alpha=0.5)
        plt.colorbar(label=r'$\log_{10}({\rm population})$')
        for a in [100, 300, 500]:
            plt.scatter([], [], c='k', alpha=0.5, s=a, label=str(a) + ' km$^2$')
        plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='lower left')
        plt.title("California Cities by Population and Area")
        plt.show()
    except:
        pass  # Data file not available

    # Temperature anomaly over North America (NetCDF data)
    try:
        data = Dataset('gistemp250.nc')
        timeindex = date2index(datetime(2014, 1, 15), data.variables['time'])
        lat = data.variables['lat'][:]
        lon = data.variables['lon'][:]
        lon, lat = np.meshgrid(lon, lat)
        temp_anomaly = data.variables['tempanomaly'][timeindex]
        plt.figure(figsize=(10,8))
        m = Basemap(projection='lcc', resolution='c', width=8E6, height=8E6, lat_0=45, lon_0=-100)
        m.shadedrelief(scale=0.5)
        m.pcolormesh(lon, lat, temp_anomaly, latlon=True, cmap='RdBu_r')
        plt.clim(-8, 8)
        m.drawcoastlines(color='lightgray')
        plt.title('January 2014 Temperature Anomaly')
        plt.colorbar(label='temperature anomaly (°C)')
        plt.show()
    except:
        pass

    # Map projections demonstration
    projections = ['cyl', 'moll', 'ortho', 'lcc']
    for proj in projections:
        plt.figure(figsize=(8,6))
        m = Basemap(projection=proj, resolution='h', lat_0=0, lon_0=0)
        def draw_map_features(m):
            m.shadedrelief()
            m.drawparallels(np.linspace(-90, 90, 13), linewidth=0.5)
            m.drawmeridians(np.linspace(-180, 180, 13), linewidth=0.5)
        draw_map_features(m)
        plt.title(f"{proj.upper()} projection")
        plt.show()

    # Boundaries at different resolutions
    fig, axes = plt.subplots(1, 2, figsize=(12,8))
    for i, res in enumerate(['l', 'h']):
        m = Basemap(projection='gnom', lat_0=57.3, lon_0=-6.2, width=90000, height=120000, resolution=res, ax=axes[i])
        m.fillcontinents(color="#FFDDCC", lake_color='#DDEEFF')
        m.drawmapboundary(fill_color="#DDEEFF")
        m.drawcoastlines()
        axes[i].set_title(f"Resolution '{res}'")
    plt.show()

# =========================
# Section 4: Scatter Plot Examples
# =========================
def plot_scatter_examples():
    # Markers with plt.plot()
    x = np.linspace(0, 10, 30)
    plt.figure(figsize=(10, 6))
    for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
        plt.plot(np.random.rand(5), np.random.rand(5), marker, label=f"marker='{marker}'")
    plt.legend(numpoints=1)
    plt.title('Scatter Markers with plt.plot()')
    plt.show()

    # Connected scatter plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, np.sin(x), '-ok')
    plt.title('Connected Scatter Plot with plt.plot()')
    plt.ylim(-1.2, 1.2)
    plt.show()

    # Random scatter with color and size
    np.random.seed(0)
    x_scatter = np.random.randn(100)
    y_scatter = np.random.randn(100)
    plt.figure(figsize=(10, 6))
    plt.scatter(x_scatter, y_scatter, c=np.random.rand(100), s=1000*np.random.rand(100), alpha=0.3, cmap='viridis')
    plt.colorbar()
    plt.title('Random Scatter Plot with Colors and Sizes')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # Iris dataset scatter
    iris = load_iris()
    features = iris.data.T
    plt.figure(figsize=(10, 6))
    plt.scatter(features[0], features[1], alpha=0.2, s=100*features[3], c=iris.target, cmap='viridis')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title('Iris Data Scatter Plot with Multiple Features')
    plt.colorbar(label='Species')
    plt.show()

# =========================
# Section 5: Error Bars and Gaussian Process Regression
# =========================
def plot_error_bars():
    # Error bars with y-error
    x = np.linspace(0, 10, 50)
    dy = 0.8
    y = np.sin(x) + dy * np.random.randn(50)
    plt.figure()
    plt.errorbar(x, y, yerr=dy, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
    plt.title('Error Bars Example')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # Gaussian Process regression with continuous error band
    def model(x): return x * np.sin(x)
    xdata = np.array([1, 3, 5, 6, 8])
    ydata = model(xdata)
    try:
        gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1E-1, random_start=100)
        gp.fit(xdata[:, None], ydata)
        xfit = np.linspace(0, 10, 1000)
        yfit, MSE = gp.predict(xfit[:, None], eval_MSE=True)
        dyfit = 2 * np.sqrt(MSE)
        plt.figure()
        plt.plot(xdata, ydata, 'or', label='Data points')
        plt.plot(xfit, yfit, '-', color='gray', label='Gaussian process fit')
        plt.fill_between(xfit, yfit - dyfit, yfit + dyfit, color='gray', alpha=0.2, label='Uncertainty region')
        plt.xlim(0, 10)
        plt.title('Gaussian Process Regression with Continuous Error Region')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()
    except:
        pass

# =========================
# Section 6: Seaborn Distributions & Plots
# =========================
def plot_seaborn_distributions():
    sns.set()
    # Random walk data
    rng = np.random.RandomState(0)
    x = np.linspace(0, 10, 500)
    y = np.cumsum(rng.randn(500, 6), axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(x, y)
    plt.legend(['A', 'B', 'C', 'D', 'E', 'F'], ncol=2, loc='upper left')
    plt.title("Random Walk with Matplotlib")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(x, y)
    plt.legend(['A', 'B', 'C', 'D', 'E', 'F'], ncol=2, loc='upper left')
    plt.title("Random Walk with Seaborn Style")
    plt.show()

    # Multivariate normal data
    data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
    df = pd.DataFrame(data, columns=['x', 'y'])

    plt.figure(figsize=(12, 4))
    for i, col in enumerate(['x', 'y']):
        plt.subplot(1, 2, i+1)
        sns.histplot(df[col], kde=False, alpha=0.5)
        plt.title(f'Histogram of {col}')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    for i, col in enumerate(['x', 'y']):
        plt.subplot(1, 2, i+1)
        sns.kdeplot(df[col], shade=True)
        plt.title(f'KDE of {col}')
    plt.tight_layout()
    plt.show()

    # Joint KDE plot
    sns.jointplot(x='x', y='y', data=df, kind='kde', height=6)
    plt.suptitle("Joint KDE Plot", y=1.02)
    plt.show()

    # Pairplot of Iris dataset
    iris_df = sns.load_dataset("iris")
    sns.pairplot(iris_df, hue='species', height=2.5)
    plt.suptitle("Pairplot of Iris Dataset", y=1.02)
    plt.show()

    # Tips dataset: facet grid
    tips = sns.load_dataset('tips')
    tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']
    g = sns.FacetGrid(tips, row='sex', col='time', margin_titles=True)
    g.map(plt.hist, 'tip_pct', bins=np.linspace(0, 40, 15))
    plt.suptitle("Tip Percentage Distributions by Sex and Time", y=1.02)
    plt.show()

    # Boxplot
    sns.catplot(x='day', y='total_bill', hue='sex', data=tips, kind='box')
    plt.title("Boxplot of Total Bill by Day and Sex")
    plt.show()

    # Regression plot
    sns.jointplot(x='total_bill', y='tip', data=tips, kind='reg')
    plt.suptitle("Regression between Total Bill and Tip", y=1.02)
    plt.show()

    # Discrete distribution example
    planets = sns.load_dataset('planets')
    g = sns.catplot(x='year', kind='count', data=planets, aspect=2)
    g.set_xticklabels(step=5)
    plt.title("Number of Planets Discovered per Year")
    plt.show()

    # Distribution of split fractions
    split_fractions = np.random.randn(1000) * 0.2
    sns.histplot(split_fractions, kde=False)
    plt.axvline(0, color='k', linestyle='--')
    plt.title("Distribution of Split Fractions")
    plt.xlabel("Split Fraction")
    plt.show()

    # Violin plot by gender
    marathon_data = pd.DataFrame({
        'gender': np.random.choice(['M', 'W'], size=100),
        'split_frac': np.random.randn(100) * 0.2,
        'age': np.random.randint(20, 80, size=100)
    })
    marathon_data['age_dec'] = (marathon_data['age'] // 10) * 10
    sns.violinplot(x='gender', y='split_frac', data=marathon_data,
                   palette=['lightblue', 'lightpink'])
    plt.title("Split Fraction Distribution by Gender")
    plt.show()

    # Violin plot by age decile
    sns.violinplot(x='age_dec', y='split_frac', hue='gender', data=marathon_data,
                   split=True, inner='quartile', palette=['lightblue', 'lightpink'])
    plt.title("Split Fraction by Age Decade and Gender")
    plt.show()

    # Regression plot: final time vs split fraction
    df = marathon_data.copy()
    df['final_sec'] = np.random.uniform(7200, 18000, size=100)
    sns.lmplot(x='final_sec', y='split_frac', col='gender', data=df, markers='.')
    plt.axhline(0.1, color='k', ls=':')
    plt.suptitle("Final Time vs Split Fraction", y=1.02)
    plt.show()

# =========================
# Section 7: Contour Plot Variations
# =========================
def plot_contour_variations():
    def f(x, y): return np.sin(x)**10 + np.cos(10 + y * x) * np.cos(x)
    x = np.linspace(0, 5, 50)
    y = np.linspace(0, 5, 40)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # Line contours
    plt.figure(figsize=(6, 5))
    plt.contour(X, Y, Z, colors='black')
    plt.title('Line Contour Plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # Multiple levels with colormap
    plt.figure(figsize=(6, 5))
    plt.contour(X, Y, Z, 20, cmap='RdGy')
    plt.title('Colored Contour Plot with 20 Levels')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # Filled contours
    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, Z, 20, cmap='RdGy')
    plt.colorbar()
    plt.title('Filled Contour Plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # Image plot
    plt.figure(figsize=(6, 5))
    plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower', cmap='RdGy')
    plt.colorbar()
    plt.title('Image Plot with imshow()')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # Overlay contours and image
    plt.figure(figsize=(6, 5))
    contours = plt.contour(X, Y, Z, 3, colors='black')
    plt.clabel(contours, inline=True, fontsize=8)
    plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower', cmap='RdGy', alpha=0.5)
    plt.colorbar()
    plt.title('Overlay of Contours and Image')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# =========================
# Section 8: KDE and Histograms
# =========================
def plot_kde_histograms():
    data = np.random.randn(1000)

    # Basic histogram
    plt.figure(figsize=(8, 4))
    plt.hist(data)
    plt.title('Basic Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

    # Overlayed histograms
    x1 = np.random.normal(0, 0.8, 1000)
    x2 = np.random.normal(-2, 1, 1000)
    x3 = np.random.normal(3, 2, 1000)
    plt.figure(figsize=(8, 4))
    kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)
    plt.hist(x1, **kwargs)
    plt.hist(x2, **kwargs)
    plt.hist(x3, **kwargs)
    plt.title('Overlayed Histograms')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()

    # 2D histogram
    mean = [0, 0]
    cov = [[1, 1], [1, 2]]
    x, y = np.random.multivariate_normal(mean, cov, 10000).T
    plt.figure(figsize=(8, 6))
    plt.hist2d(x, y, bins=30, cmap='Blues')
    plt.colorbar(label='counts in bin')
    plt.title('2D Histogram with plt.hist2d')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # np.histogram2d + pcolormesh
    counts, xedges, yedges = np.histogram2d(x, y, bins=30)
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(xedges, yedges, counts.T, cmap='Blues')
    plt.colorbar(label='counts in bin')
    plt.title('2D Histogram with np.histogram2d and pcolormesh')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # Hexbin plot
    plt.figure(figsize=(8, 6))
    plt.hexbin(x, y, gridsize=30, cmap='Blues')
    plt.colorbar(label='count in bin')
    plt.title('Hexbin Plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # KDE estimation
    data_for_kde = np.vstack([x, y])
    kde = gaussian_kde(data_for_kde)
    xgrid = np.linspace(-3.5, 3.5, 40)
    ygrid = np.linspace(-6, 6, 40)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
    plt.figure(figsize=(8, 6))
    plt.imshow(Z.reshape(Xgrid.shape), origin='lower', aspect='auto',
               extent=[-3.5, 3.5, -6, 6], cmap='Blues')
    plt.colorbar(label='density')
    plt.title('Kernel Density Estimation (KDE)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# =========================
# Section 9: Trigonometric Functions and Legends
# =========================
def plot_trig_wave():
    x = np.linspace(0,10,100)
    y = np.sin(x)
    plt.figure()
    plt.plot(x, y, label='sin(x)')
    plt.title('Sine Wave')
    plt.legend()
    plt.show()

def plot_legend_examples():
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x + np.pi/4)

    # Basic plot with legend
    fig, ax = plt.subplots()
    ax.plot(x, y1, '-b', label='Sine')
    ax.plot(x, y2, '--r', label='Cosine')
    ax.axis('equal')
    ax.legend()
    plt.show()

    # Multiple columns legend
    fig, ax = plt.subplots()
    ax.plot(x, y1, label='Sine')
    ax.plot(x, y2, label='Cosine')
    ax.plot(x, y3, label='Sine shifted')
    ax.legend(frameon=False, loc='lower center', ncol=2)
    plt.show()

    # Legend with fancy box and shadow
    fig, ax = plt.subplots()
    ax.plot(x, y1, label='Sine')
    ax.plot(x, y2, label='Cosine')
    ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.show()

    # Selective handles in legend
    fig, ax = plt.subplots()
    lines = []
    lines += ax.plot(x, y1, label='Sine')
    lines += ax.plot(x, y2, label='Cosine')
    ax.legend(handles=lines[:2], labels=['first', 'second'])
    plt.show()

    # Labels set directly in plot
    fig, ax = plt.subplots()
    ax.plot(x, y1, label='Sine')
    ax.plot(x, y2, label='Cosine')
    ax.plot(x, y3)
    ax.legend()
    plt.show()

    # City population legend example
    cities = pd.DataFrame({
        'latd': np.random.uniform(32, 42, 10),
        'longd': np.random.uniform(-124, -114, 10),
        'population_total': np.random.randint(100000, 1000000, 10),
        'area_total_km2': np.random.uniform(50, 500, 10)
    })
    lat, lon = cities['latd'], cities['longd']
    population = cities['population_total']
    area = cities['area_total_km2']
    plt.figure(figsize=(8,6))
    plt.scatter(lon, lat, c=np.log10(population), cmap='viridis', s=area, linewidth=0, alpha=0.5)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar(label='log$_{10}$(population)')
    for sz in [100, 300, 500]:
        plt.scatter([], [], c='k', alpha=0.3, s=sz, label=str(sz) + ' km$^2$')
    plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')
    plt.show()

    # Multiple legends with different line styles
    fig, ax = plt.subplots()
    lines = []
    styles = ['-', '--', '-.', ':']
    for i in range(4):
        lines += ax.plot(x, np.sin(x - i*np.pi/2), styles[i], color='black')
    ax.axis('equal')
    ax.legend(lines[:2], ['Line A', 'Line B'], loc='upper right', frameon=False)
    second_leg = Legend(ax, lines[2:], ['Line C', 'Line D'], loc='lower right', frameon=False)
    ax.add_artist(second_leg)
    plt.show()

# =========================
# Section 10: Colormaps and Projections
# =========================
def plot_colormap_and_projection():
    # Basic heatmap
    x = np.linspace(0, 10, 1000)
    I = np.sin(x) * np.cos(x[:, None])
    plt.figure(figsize=(6, 4))
    plt.imshow(I)
    plt.colorbar()
    plt.title("Basic Heatmap with Colorbar")
    plt.show()

    # Gray colormap
    plt.figure(figsize=(6, 4))
    plt.imshow(I, cmap='gray')
    plt.title("Image with Gray Colormap")
    plt.colorbar()
    plt.show()

    # Grayscale colormap function
    from matplotlib.colors import LinearSegmentedColormap
    def grayscale_cmap(cmap_name):
        cmap = plt.cm.get_cmap(cmap_name)
        colors = cmap(np.arange(cmap.N))
        RGB_weight = [0.299, 0.587, 0.114]
        luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
        colors[:, :3] = luminance[:, np.newaxis]
        return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)

    # View colormap and grayscale
    def view_colormap(cmap_name):
        cmap = plt.cm.get_cmap(cmap_name)
        colors = cmap(np.arange(cmap.N))
        gray_cmap = grayscale_cmap(cmap_name)
        grayscale = gray_cmap(np.arange(gray_cmap.N))
        fig, axes = plt.subplots(2, figsize=(6, 2))
        axes[0].imshow([colors], extent=[0, 10, 0, 1])
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_title(f"{cmap_name} Colormap")
        axes[1].imshow([grayscale], extent=[0, 10, 0, 1])
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].set_title(f"{cmap_name} Grayscale")
        plt.show()

    for cmap_name in ['jet', 'viridis', 'cubehelix', 'RdBu']:
        view_colormap(cmap_name)

    # Demonstrate color limits and extensions
    np.random.seed(0)
    noisy_I = I.copy()
    speckles = np.random.random(I.shape) < 0.01
    noisy_I[speckles] = np.random.normal(0, 3, np.count_nonzero(speckles))
    plt.figure(figsize=(10, 3.5))
    plt.subplot(1, 2, 1)
    plt.imshow(noisy_I, cmap='RdBu')
    plt.colorbar()
    plt.title("Default Color Limits")
    plt.subplot(1, 2, 2)
    plt.imshow(noisy_I, cmap='RdBu')
    plt.colorbar(extend='both')
    plt.clim(-1, 1)
    plt.title("Customized Limits with Extensions")
    plt.show()

    # Discrete colormap
    plt.figure()
    plt.imshow(I, cmap=plt.cm.get_cmap('Blues', 6))
    plt.colorbar()
    plt.title("Discrete Colormap with 6 Bins")
    plt.clim(-1, 1)
    plt.show()

    # Handwritten digits and projection
    digits = load_digits(n_class=6)
    fig, axes = plt.subplots(8, 8, figsize=(6, 6))
    for i, axi in enumerate(axes.flat):
        axi.imshow(digits.images[i], cmap='binary')
        axi.set(xticks=[], yticks=[])
    plt.suptitle("Sample Digits")
    plt.show()

    iso = Isomap(n_components=2)
    projection = iso.fit_transform(digits.data)
    plt.figure(figsize=(8, 6))
    plt.scatter(projection[:, 0], projection[:, 1], lw=0.5, c=digits.target, cmap=plt.cm.get_cmap('cubehelix', 6))
    plt.colorbar(ticks=range(6), label='digit value')
    plt.clim(-0.5, 5.5)
    plt.title("Digits projected into 2D with Discrete Colormap")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

# =========================
# Section 11: Axes Layouts and Subplots
# =========================
def plot_axes_layouts():
    # Manual axes
    x = np.linspace(0, 10, 100)
    plt.figure()
    ax1 = plt.axes()
    ax1.plot(x, np.sin(x))
    ax1.set_title('Basic Axes')
    plt.show()

    # Inset axes
    fig = plt.figure()
    ax2 = fig.add_axes([0.65, 0.65, 0.2, 0.2])
    ax2.plot(x, np.cos(x))
    ax2.set_title('Inset Axes')
    plt.show()

    # Stacked axes
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(6, 8))
    ax3.plot(x, np.sin(x))
    ax3.set_title('Top Plot')
    ax4.plot(x, np.cos(x))
    ax4.set_title('Bottom Plot')
    plt.tight_layout()
    plt.show()

    # Grid of subplots
    fig3 = plt.figure()
    for i in range(1, 7):
        plt.subplot(2, 3, i)
        plt.text(0.5, 0.5, f"Subplot {i}", fontsize=14, ha='center')
    plt.suptitle('2x3 Grid with plt.subplot()')
    plt.tight_layout()
    plt.show()

    # Shared axes grid with GridSpec
    from matplotlib.gridspec import GridSpec
    fig5 = plt.figure(figsize=(8,8))
    gs = GridSpec(4, 4, hspace=0.4, wspace=0.4)
    ax_main = fig5.add_subplot(gs[:-1, 1:])
    ax_xhist = fig5.add_subplot(gs[-1, 1:], yticklabels=[], sharex=ax_main)
    ax_yhist = fig5.add_subplot(gs[:-1, 0], xticklabels=[], sharey=ax_main)
    # Generate data
    x2 = np.random.randn(1000)
    y2 = np.random.randn(1000)
    ax_main.scatter(x2, y2, s=10, alpha=0.5)
    ax_xhist.hist(x2, bins=30, orientation='vertical', color='gray', histtype='stepfilled')
    ax_xhist.invert_yaxis()
    ax_yhist.hist(y2, bins=30, orientation='horizontal', color='gray', histtype='stepfilled')
    ax_yhist.invert_xaxis()
    ax_main.set_title('Main Scatter Plot with Marginal Histograms')
    plt.suptitle('Complex Layout with GridSpec')
    plt.show()

# =========================
# Section 12: Birth Data Annotations
# =========================
def plot_births_annotations():
    # Load birth data
    births_url = 'https://raw.githubusercontent.com/jakevdp/data-CDCbirths/master/births.csv'
    births = pd.read_csv(births_url)
    quartiles = np.percentile(births['births'], [25, 50, 75])
    mu, sig = quartiles[1], 0.74 * (quartiles[2] - quartiles[0])
    births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')
    births['day'] = births['day'].astype(int)
    births.index = pd.to_datetime(
        10000 * births.year + 100 * births.month + births.day, format='%Y%m%d'
    )
    births_by_date = births.pivot_table('births', [births.index.month, births.index.day])
    births_by_date.index = [pd.to_datetime(f'2012-{m}-{d}') for (m, d) in births_by_date.index]
    fig, ax = plt.subplots(figsize=(12, 4))
    births_by_date.plot(ax=ax)
    style = dict(size=10, color='gray')
    # Annotate some holidays
    ax.text('2012-1-1', 3950, "New Year's Day", **style)
    ax.text('2012-7-4', 4250, "Independence Day", ha='center', **style)
    ax.text('2012-9-4', 4850, "Labor Day", ha='center', **style)
    ax.text('2012-10-31', 4600, "Halloween", ha='right', **style)
    ax.text('2012-11-25', 4450, "Thanksgiving", ha='center', **style)
    ax.text('2012-12-25', 3850, "Christmas", ha='right', **style)
    ax.set(title='USA births by day of year (1969-1988)', ylabel='average daily births')
    # Format x-axis with months
    import matplotlib as mpl
    ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
    ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'))
    plt.show()

# =========================
# Section 13: Text Transformations & Annotations
# =========================
def plot_text_annotations():
    # Text with different coordinate transforms
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.axis([0, 10, 0, 10])
    ax2.text(1, 5, ". Data: (1, 5)", transform=ax2.transData)
    ax2.text(0.5, 0.1, ". Axes: (0.5, 0.1)", transform=ax2.transAxes)
    ax2.text(0.2, 0.2, ". Figure: (0.2, 0.2)", transform=fig2.transFigure)
    plt.show()

    # Annotations with arrows
    x = np.linspace(0, 20, 1000)
    fig3, ax3 = plt.subplots()
    ax3.plot(x, np.cos(x))
    ax3.axis('equal')
    ax3.annotate('local maximum', xy=(6.28, 1), xytext=(10, 4),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    ax3.annotate('local minimum', xy=(5*np.pi, -1), xytext=(2, -6),
                 arrowprops=dict(arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=-90"))
    plt.show()

# =========================
# Section 14: Tick Formatting & Scales
# =========================
def plot_axis_tick_formatting():
    # Logarithmic axes with major and minor ticks
    fig1, ax1 = plt.subplots()
    ax1.set_title('Log-Scale with Major and Minor Ticks')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    x = np.logspace(0.1, 2, 100)
    y = np.exp(np.sqrt(x))
    ax1.plot(x, y)
    ax1.grid(which='both')
    print("Major locator:", ax1.xaxis.get_major_locator())
    print("Minor locator:", ax1.xaxis.get_minor_locator())
    print("Major formatter:", ax1.xaxis.get_major_formatter())
    print("Minor formatter:", ax1.xaxis.get_minor_formatter())
    plt.show()

    # Hide ticks and labels
    fig2, ax2 = plt.subplots()
    ax2.plot(np.random.rand(50))
    ax2.yaxis.set_major_locator(plt.NullLocator())
    ax2.xaxis.set_major_formatter(plt.NullFormatter())
    ax2.set_title('No Ticks or Labels')
    plt.show()

    # MaxNLocator to reduce tick count
    fig3, ax3 = plt.subplots(2, 2, sharex=True, sharey=True)
    for axi in ax3.flat:
        axi.plot(np.random.rand(10))
        axi.xaxis.set_major_locator(plt.MaxNLocator(3))
        axi.yaxis.set_major_locator(plt.MaxNLocator(3))
    plt.suptitle('Reduced Tick Density with MaxNLocator')
    plt.show()

    # Fancy tick formatting using multiples of pi
    fig4, ax4 = plt.subplots()
    x_vals = np.linspace(0, 3*np.pi, 1000)
    ax4.plot(x_vals, np.sin(x_vals), label='sin(x)')
    ax4.plot(x_vals, np.cos(x_vals), label='cos(x)')
    ax4.legend()
    ax4.grid(True)
    ax4.set_xlim(0, 3*np.pi)

    # Custom formatter for pi multiples
    def format_func(value, tick_number):
        N = int(np.round(2 * value / np.pi))
        if N == 0:
            return "0"
        elif N == 1:
            return r"$\pi/2$"
        elif N == 2:
            return r"$\pi$"
        elif N % 2 > 0:
            return r"${0}\pi/2$".format(N)
        else:
            return r"${0}\pi$".format(N // 2)

    ax4.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    plt.show()

# =========================
# Main execution: Run all sections
# =========================
def run_all():
    plot_3d_visualizations()
    plot_simple_wave()
    plot_maps_and_geography()
    plot_scatter_examples()
    plot_error_bars()
    plot_seaborn_distributions()
    plot_contour_variations()
    plot_kde_histograms()
    plot_trig_wave()
    plot_legend_examples()
    plot_colormap_and_projection()
    plot_axes_layouts()
    plot_births_annotations()
    plot_text_annotations()
    plot_axis_tick_formatting()

if __name__ == "__main__":
    run_all()
