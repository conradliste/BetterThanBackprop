import matplotlib.pyplot as plt

def plot_trajectory(projected_params):
    # Separate components
    x = projected_params[:, 0]
    y = projected_params[:, 1]
    z = projected_params[:, 2]
    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    # Creating plot
    ax.scatter3D(x, y, z, color="green")
    plt.title("Projected Learning Trajectory")


def display_images(images, images_per_row=4):
    """
    Displays a grid of images
    
    Args:
        - images (Iterable): the set of images to display
        - images_per_row (int): the number of images per row
    
    Returns:
        The figure and axes
    """
    nrows = len(images) // images_per_row
    num_leftover = len(images) % images_per_row
    # Add another row in case we have a remainder
    if num_leftover > 0:
        nrows += 1
    # Create  the figure
    fig, axs = plt.subplots(nrows=nrows, ncols=images_per_row)
    # Display images in each subplot
    for i, image in enumerate(images):
        axs.flat[i].imshow(image, cmap='binary')
    return fig, axs
    
        