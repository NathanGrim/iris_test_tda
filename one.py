import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from sklearn import datasets
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

# --- 1. Load and Prepare the Data ---

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # The 4D data
y = iris.target # The species labels (0, 1, 2)
target_names = iris.target_names

# Reduce dimensionality from 4D to 2D using PCA
pca = PCA(n_components=2)
points_2d = pca.fit_transform(X)

# --- 2. Set up the Interactive Plot ---

# Create a figure and axes for the plot with navigation toolbar enabled
fig, ax = plt.subplots(figsize=(12, 9))
plt.subplots_adjust(bottom=0.3, top=0.95)  # Make space for controls

# Enable the navigation toolbar (includes zoom, pan, etc.)
# This is enabled by default, but we make sure it's available
fig.canvas.toolbar_visible = True

# Define colors for the species
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Standard matplotlib colors

# Create the initial scatter plot
# We store the plot elements in a dictionary to easily access them later
scatter_plots = {}
for i, name in enumerate(target_names):
    # Select data for each species
    species_points = points_2d[y == i]
    # Create scatter plot for the species
    scatter_plots[name] = ax.scatter(
        species_points[:, 0], species_points[:, 1],
        c=colors[i], label=name, alpha=0.8, s=50  # Increased marker size slightly
    )

ax.set_title('Vietoris-Rips Complex for Iris Dataset (Radius ε = 0.00)', fontsize=14)
ax.set_xlabel("Principal Component 1", fontsize=12)
ax.set_ylabel("Principal Component 2", fontsize=12)
ax.legend(title="Species", loc='upper right')
ax.grid(True, alpha=0.3)  # Add grid for better reference

# Store initial axis limits for reset functionality
initial_xlim = ax.get_xlim()
initial_ylim = ax.get_ylim()

# Pre-calculate the pairwise distance matrix for efficiency
# This matrix stores the distance between every pair of points
dist_matrix = squareform(pdist(points_2d, 'euclidean'))

# --- 3. Create Controls (Slider and Zoom Buttons) ---

# Add an axes for the slider
ax_slider = plt.axes([0.25, 0.15, 0.5, 0.03], facecolor='lightgray')

# Create the slider widget
slider = Slider(
    ax=ax_slider,
    label='Radius (ε)',
    valmin=0,
    valmax=1.5,
    valinit=0
)

# Add zoom control buttons
ax_zoom_in = plt.axes([0.25, 0.05, 0.1, 0.04])
ax_zoom_out = plt.axes([0.36, 0.05, 0.1, 0.04])
ax_reset_zoom = plt.axes([0.47, 0.05, 0.1, 0.04])
ax_fit_data = plt.axes([0.58, 0.05, 0.1, 0.04])

btn_zoom_in = Button(ax_zoom_in, 'Zoom In')
btn_zoom_out = Button(ax_zoom_out, 'Zoom Out')
btn_reset_zoom = Button(ax_reset_zoom, 'Reset View')
btn_fit_data = Button(ax_fit_data, 'Fit Data')

# This list will hold all the artist objects (circles, lines) we draw
# so we can easily remove them on update
drawn_artists = []

# --- 4. Define Update and Zoom Functions ---

# This function will be called every time the slider value changes
def update(val):
    # Get the current radius value from the slider
    radius = slider.val

    # Clear the previous drawing (circles and lines)
    for artist in drawn_artists:
        artist.remove()
    drawn_artists.clear()

    # Find pairs of points that are close enough to connect
    # An edge exists if dist(p1, p2) <= 2 * radius
    edge_indices = np.argwhere(dist_matrix < 2 * radius)

    # Draw the new lines (edges)
    for i, j in edge_indices:
        # Avoid drawing self-loops and double-drawing edges
        if i < j:
            p1 = points_2d[i]
            p2 = points_2d[j]
            line, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.3, lw=0.5)
            drawn_artists.append(line)

    # Optional: Draw the circles around each point
    for i in range(len(points_2d)):
        circle = plt.Circle(points_2d[i], radius, color='gray', alpha=0.1, fill=True)
        ax.add_artist(circle)
        drawn_artists.append(circle)
        
    # Update the title with the current radius
    ax.set_title(f'Vietoris-Rips Complex for Iris Dataset (Radius ε = {radius:.2f})', fontsize=14)
    
    # Redraw the plot canvas
    fig.canvas.draw_idle()

# Zoom functions
def zoom_in(event):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Zoom in by reducing the range by 20%
    x_center = (xlim[0] + xlim[1]) / 2
    y_center = (ylim[0] + ylim[1]) / 2
    x_range = (xlim[1] - xlim[0]) * 0.4
    y_range = (ylim[1] - ylim[0]) * 0.4
    ax.set_xlim(x_center - x_range, x_center + x_range)
    ax.set_ylim(y_center - y_range, y_center + y_range)
    fig.canvas.draw_idle()

def zoom_out(event):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Zoom out by increasing the range by 25%
    x_center = (xlim[0] + xlim[1]) / 2
    y_center = (ylim[0] + ylim[1]) / 2
    x_range = (xlim[1] - xlim[0]) * 0.625
    y_range = (ylim[1] - ylim[0]) * 0.625
    ax.set_xlim(x_center - x_range, x_center + x_range)
    ax.set_ylim(y_center - y_range, y_center + y_range)
    fig.canvas.draw_idle()

def reset_zoom(event):
    ax.set_xlim(initial_xlim)
    ax.set_ylim(initial_ylim)
    fig.canvas.draw_idle()

def fit_data(event):
    # Auto-fit to show all data points with some padding
    x_min, x_max = points_2d[:, 0].min(), points_2d[:, 0].max()
    y_min, y_max = points_2d[:, 1].min(), points_2d[:, 1].max()
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    fig.canvas.draw_idle()

# Enable mouse wheel zoom
def on_scroll(event):
    if event.inaxes != ax:
        return
    
    # Get current mouse position
    x_mouse, y_mouse = event.xdata, event.ydata
    if x_mouse is None or y_mouse is None:
        return
    
    # Get current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Define zoom factor
    zoom_factor = 1.1 if event.button == 'down' else 0.9
    
    # Calculate new limits centered on mouse position
    x_range = (xlim[1] - xlim[0]) * zoom_factor
    y_range = (ylim[1] - ylim[0]) * zoom_factor
    
    # Calculate how much to shift based on mouse position
    x_shift = (x_mouse - (xlim[0] + xlim[1]) / 2) * (1 - zoom_factor)
    y_shift = (y_mouse - (ylim[0] + ylim[1]) / 2) * (1 - zoom_factor)
    
    new_x_center = (xlim[0] + xlim[1]) / 2 + x_shift
    new_y_center = (ylim[0] + ylim[1]) / 2 + y_shift
    
    ax.set_xlim(new_x_center - x_range/2, new_x_center + x_range/2)
    ax.set_ylim(new_y_center - y_range/2, new_y_center + y_range/2)
    fig.canvas.draw_idle()

# --- 5. Connect Controls and Show Plot ---

# Register the update function with the slider
slider.on_changed(update)

# Connect zoom buttons
btn_zoom_in.on_clicked(zoom_in)
btn_zoom_out.on_clicked(zoom_out)
btn_reset_zoom.on_clicked(reset_zoom)
btn_fit_data.on_clicked(fit_data)

# Connect mouse wheel zoom
fig.canvas.mpl_connect('scroll_event', on_scroll)

# Add some helpful text
fig.text(0.02, 0.02, 'Navigation: Use toolbar above, mouse wheel to zoom, or buttons below', 
         fontsize=10, style='italic', alpha=0.7)

# Display the plot
plt.show()