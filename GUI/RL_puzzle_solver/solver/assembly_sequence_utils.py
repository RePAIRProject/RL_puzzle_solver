#!/usr/bin/python3

from matplotlib import pyplot as plt
import matplotlib.patches as patches
# plt.ion()
import numpy as np
import math
import os

from shapely.geometry import Point, Polygon
from shapely import plotting
from GUI.RL_puzzle_solver.solver import scale_utils


def prepare_polygon_data(multipolygon):
    polygon_array = scale_utils.convert_shapely_multi_polygon_to_array_of_shapely_polygons(multipolygon)

    centroids = []
    polygons = []
    for poly in polygon_array:
        centroid_point = poly.centroid
        centroid_tuple = centroid_point.x, centroid_point.y
        centroids.append(centroid_tuple)
        polygons.append(list(poly.exterior.coords))

    assembly_data = scale_utils.combine_centroids_and_vertices(centroids, polygons)
    return assembly_data


def get_assembly_plan_snake(polygons_data, movement_delay=0.01, viz=False):
    # Extract centroids data from the dictionary
    centroids_data = [polygon['centroids']
                      for polygon in polygons_data.values()]
    # Extract vertices data from the dictionary
    vertices_data = [polygon['vertices']
                     for polygon in polygons_data.values()]

    # Calculate the larger fresco box
    all_vertices = [
        vertex for vertices in vertices_data for vertex in vertices]
    min_x = min(all_vertices, key=lambda x: x[0])[0]
    min_y = min(all_vertices, key=lambda x: x[1])[1]
    max_x = max(all_vertices, key=lambda x: x[0])[0]
    max_y = max(all_vertices, key=lambda x: x[1])[1]

    larger_box_width = max_x - min_x
    larger_box_height = max_y - min_y

    # Calculate the smaller box
    smallest_area = float('inf')
    for vertices in vertices_data:
        polygon_x = [vertex[0] for vertex in vertices]
        polygon_y = [vertex[1] for vertex in vertices]
        area = (max(polygon_x) - min(polygon_x)) * \
               (max(polygon_y) - min(polygon_y))
        if area < smallest_area:
            smallest_area = area
            smaller_box_width = max(polygon_x) - min(polygon_x)
            smaller_box_height = max(polygon_y) - min(polygon_y)

    if viz == True:
        # Create a figure and axis for plotting
        fig, ax = plt.subplots()

    # Create the larger box
    larger_box_x = min_x
    larger_box_y = min_y
    larger_box = patches.Rectangle((larger_box_x, larger_box_y), larger_box_width,
                                   larger_box_height, linewidth=1, edgecolor='r', facecolor='none')

    # Initialize the smaller box position
    smaller_box_x = larger_box_x
    smaller_box_y = larger_box_y

    # Define the direction flags for movement
    move_up = True
    move_right = True
    shift_right = False

    # Lists to store the x and y coordinates for plotting
    x_positions = []
    y_positions = []

    # Plot all fresco vertices
    all_x = [vertex[0] for vertex in all_vertices]
    all_y = [vertex[1] for vertex in all_vertices]

    # Plot all fresco centroids
    centroids_x = [vertex[0] for vertex in centroids_data]
    centroids_y = [vertex[1] for vertex in centroids_data]

    # Lists to store polygon IDs and centroids
    polygon_info = []

    # print("Generating assembly plan...")
    # Perform the snake like pattern until reaching the top-right corner
    while smaller_box_x + 0.005 < larger_box_x + larger_box_width:
        # Add current position to the lists
        x_positions.append(smaller_box_x)
        y_positions.append(smaller_box_y)

        if viz:
            # Clear and redraw the smaller box
            ax.clear()
            ax.add_patch(larger_box)
            smaller_box = patches.Rectangle((smaller_box_x, smaller_box_y), smaller_box_width,
                                            smaller_box_height, linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(smaller_box)

            plt.scatter(all_x, all_y, c='k', marker='*', label='vertices')
            plt.scatter(centroids_x, centroids_y, c='r',
                        marker='x', label='centroids')

            # Plot the current state
            plt.plot(x_positions, y_positions, 'g--')
            if movement_delay > 0:
                plt.pause(movement_delay)  # Pause to visualize each step

        # Check vertices inside the smaller box
        vertices_inside_box = []
        for vertex in all_vertices:
            x, y = vertex
            if (smaller_box_x <= x <= smaller_box_x + smaller_box_width) and (
                    smaller_box_y <= y <= smaller_box_y + smaller_box_height):
                vertices_inside_box.append(vertex)

        # Get unique polygon IDs and their centroids
        unique_polygon_info = set()
        for vertex in vertices_inside_box:
            for polygon_id, polygon_data in polygons_data.items():
                if vertex in polygon_data['vertices']:
                    centroid = tuple(polygon_data['centroids'])
                    # Check if the centroid is within the smaller box
                    if (smaller_box_x <= centroid[0] <= smaller_box_x + smaller_box_width) and (
                            smaller_box_y <= centroid[1] <= smaller_box_y + smaller_box_height):
                        unique_polygon_info.add((polygon_id, centroid))

        # Check if vertices_inside_box is empty
        if not vertices_inside_box:
            # Check if any centroids are inside the smaller box
            for polygon_id, polygon_data in polygons_data.items():
                centroid = tuple(polygon_data['centroids'])
                if (smaller_box_x <= centroid[0] <= smaller_box_x + smaller_box_width) and (
                        smaller_box_y <= centroid[1] <= smaller_box_y + smaller_box_height):
                    unique_polygon_info.add((polygon_id, centroid))

        # Sort the unique_polygon_info by centroid's x-coordinate and y-coordinate
        unique_polygon_info = sorted(
            unique_polygon_info, key=lambda x: (x[1][0], x[1][1]))

        # Append the polygon based on the specified rules
        if unique_polygon_info:
            if len(unique_polygon_info) == 1:
                # Consider bottom left centroid
                polygon_info.append(unique_polygon_info[0])
            else:
                # Consider bottom left centroid
                polygon_info.append(unique_polygon_info[0])

        # Move the smaller box vertically up
        if move_up:
            smaller_box_y += (smaller_box_height / 1.5)
            if smaller_box_y + smaller_box_height > larger_box_y + larger_box_height:
                smaller_box_y = larger_box_y + larger_box_height - smaller_box_height
                move_up = False  # Set when top hit
                move_right = True
        # Shift the smaller box horizontally right with a smaller offset
        elif move_right:
            if shift_right:
                smaller_box_x += (smaller_box_width / 1.5)
                shift_right = False
                move_up = True
            else:
                smaller_box_x += (smaller_box_width / 1.5)
                shift_right = True

            # Change direction after shifting right
            move_right = False
        # Move the smaller box vertically down
        else:
            smaller_box_y -= (smaller_box_height / 1.5)
            if smaller_box_y < larger_box_y:
                smaller_box_y = larger_box_y
                move_right = True

    # Get final assembly plan
    assembly_id = [item[0] for item in polygon_info]
    assembly_plan = []
    seen = set()
    for item in assembly_id:
        if item not in seen:
            assembly_plan.append(item)
            seen.add(item)
    # print("Assembly planing done.")
    print("Assembly planing order snake:", assembly_plan)
    return assembly_plan


def get_assembly_plan_spiral(polygons_data, spiral_radius_increment=10.0, num_iterations=1200, viz=False):
    # Extract vertices data from the dictionary
    vertices_data = [polygon['vertices']
                     for polygon in polygons_data.values()]
    centroids_data = [polygon['centroids']
                      for polygon in polygons_data.values()]
    no_of_frescos = len(centroids_data)

    points = [vertex for vertices in vertices_data for vertex in vertices]
    points = np.asarray(points)

    # Calculate the centroid of all points
    centroid = np.mean(points, axis=0)

    # Initialize variables for the spiral movement
    angle = 0
    radius_increment = spiral_radius_increment
    polygon_ids = []
    trace_x = []
    trace_y = []

    # Calculate the smaller box dimensions
    smallest_area = float('inf')
    for vertices in vertices_data:
        polygon_x = [vertex[0] for vertex in vertices]
        polygon_y = [vertex[1] for vertex in vertices]
        area = (max(polygon_x) - min(polygon_x)) * \
               (max(polygon_y) - min(polygon_y))
        if area < smallest_area:
            smallest_area = area
            rect_width = (max(polygon_x) - min(polygon_x)) / 2  # Use half width of the smallest fresco
            rect_length = (max(polygon_y) - min(polygon_y)) / 2  # Use half length of the smallest fresco

    # Create Shapely Polygon for the small rectangle
    rect_polygon = Polygon([
        (centroid[0] - rect_length / 2, centroid[1] - rect_width / 2),
        (centroid[0] + rect_length / 2, centroid[1] - rect_width / 2),
        (centroid[0] + rect_length / 2, centroid[1] + rect_width / 2),
        (centroid[0] - rect_length / 2, centroid[1] + rect_width / 2)
    ])

    # Create Shapely Polygons for each polygon's vertices
    polygons = [Polygon(vertices) for vertices in vertices_data]

    if viz == True:
        # Create a Matplotlib figure and axis
        fig, ax = plt.subplots()

    # Define a function to update the rectangle's position and visualize
    def update_rectangle():
        nonlocal angle
        nonlocal centroid
        nonlocal polygon_ids
        nonlocal trace_x, trace_y

        # Calculate the new position of the rectangle in a spiral pattern
        radius = angle * radius_increment
        x = centroid[0] + radius * math.cos(angle)
        y = centroid[1] + radius * math.sin(angle)

        # Update the rectangle's position
        rect_polygon_new = Polygon([
            (x - rect_length / 2, y - rect_width / 2),
            (x + rect_length / 2, y - rect_width / 2),
            (x + rect_length / 2, y + rect_width / 2),
            (x - rect_length / 2, y + rect_width / 2)
        ])
        angle += 0.1

        # Check if the rectangle is fully contained within any polygon
        for i, polygon in enumerate(polygons):
            if polygon.intersects(rect_polygon_new):
                centroid_tuple = (polygon.centroid.x, polygon.centroid.y)
                for polygon_id, curr_centroid in enumerate(centroids_data):
                    precision = 12
                    centroid_tuple = (round(centroid_tuple[0], precision), round(
                        centroid_tuple[1], precision))
                    curr_centroid = (round(curr_centroid[0], precision), round(
                        curr_centroid[1], precision))
                    if centroid_tuple == curr_centroid:
                        polygon_ids.append(polygon_id)

        # Append the current position to the trace
        trace_x.append(x)
        trace_y.append(y)

        if viz == True:
            # Clear the previous plot and redraw the plot with the updated polygons and rectangle
            ax.clear()
            ax.set_aspect('equal', adjustable='box')
            ax.set_title("Spiral Movement of Rectangle")

            # Plot polygons
            for polygon in polygons:
                x, y = polygon.exterior.xy
                ax.plot(x, y, color='red')

            # Plot the rectangle
            x, y = rect_polygon_new.exterior.xy
            ax.fill(x, y, color='blue', alpha=0.5)

            # Plot trace
            ax.plot(trace_x, trace_y, color='green',
                    linestyle='-', label='Trace')
            ax.legend()

            # Pause for visualization
            plt.pause(0.0001)

    # Animate the movement of the rectangle
    for i in range(num_iterations):
        update_rectangle()
        if no_of_frescos == len(set(polygon_ids)):
            break

    # Get final assembly plan
    assembly_plan = []
    seen = set()
    for item in polygon_ids:
        if item not in seen:
            assembly_plan.append(item)
            seen.add(item)
    # print("Assembly planing done.")
    print("Assembly planing order spiral:", assembly_plan)

    return assembly_plan


def plot_fresco_assembly(assembly_plan, polygons_data, fragment_names, folder_path, name, block_window: bool = True,
                         pause_duration=0.5):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', 'box')

    for i in range(len(assembly_plan) - 1):
        current_index = assembly_plan[i]
        next_index = assembly_plan[i + 1]

        current_data = polygons_data[current_index]
        next_data = polygons_data[next_index]

        current_centroid = Point(current_data['centroids'])
        next_centroid = Point(next_data['centroids'])

        current_polygon = Polygon(current_data['vertices'])
        next_polygon = Polygon(next_data['vertices'])

        # Extract exterior coordinates manually
        current_exterior = np.array(current_polygon.exterior.xy).T
        next_exterior = np.array(next_polygon.exterior.xy).T

        # Draw the polygons
        ax.fill(current_exterior[:, 0], current_exterior[:, 1], facecolor='blue', edgecolor='cyan')
        # plt.text(current_centroid.x, current_centroid.y, str(i) + "/id:" + str(fragment_names[current_index]),
        #          ha='center', va='center')

        ax.fill(next_exterior[:, 0], next_exterior[:, 1], facecolor='green', edgecolor='black')
        # plt.text(next_centroid.x, next_centroid.y, str(i + 1) + "/id:" + str(fragment_names[next_index]), ha='center',
        #          va='center')

    print("Save fresco plot ", name)
    plt.savefig(fname=os.path.join(folder_path, name + ".png"), dpi=300, format="png")
    # plt.savefig(fname=path+name+".svg", format="svg")


def plot_fresco_comparison_image(img_path, fresco_polygons_list, ref="fresco", name="", colors=["black", "C0"],
                                 visualize=False, save_plot=False):
    ax, fig = plot_fresco_image(img_path, fresco_polygons_list[0], ref=ref, name=name, color=colors[0])
    plot_fresco_image(img_path, fresco_polygons_list[1], ref=ref, name=name, color=colors[1], ax=ax, fig=fig,
                      visualize=visualize, save_plot=save_plot)


def plot_fresco_image(img_path, fresco_polygons, ref="fresco", name="", color="C0", line_width=2.0, plot_grid=True,
                      plot_axis_labels=True, plot_centroids=True, ax=None, fig=None, visualize=False, save_plot=False):
    if name == "":
        name = Path(img_path).stem
    if ax is None:
        # TODO: Get length and with from gt_data
        if ref == "fresco":
            fresco_length = 0.12
            fresco_width = 0.18
            worst_case_sf = 2.0
            worst_case_length = fresco_length * (1.0 + worst_case_sf)
            worst_case_width = fresco_width * (1.0 + worst_case_sf)
        # elif ref == "world":
        #     pass

        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'box')
        if ref == "fresco":
            ax.set_xlim(-worst_case_length / 2, worst_case_length / 2)
            ax.set_ylim(-worst_case_width / 2, worst_case_width / 2)
        # elif ref == "world":
        #     pass

    # Plot fresco
    plotting.plot_polygon(
        fresco_polygons,
        ax,
        alpha=0.5,
        add_points=False,
        color=color
    )
    # Plot centroids
    if plot_centroids:
        for fragment in fresco_polygons.geoms:
            plotting.plot_points(
                fragment.centroid,
                color="black"
            )

    if plot_axis_labels:
        ax.set_xlabel(r"$x$ [mm]")
        ax.set_ylabel("$y$ [mm]")
    else:
        ax.xticks([])
        ax.yticks([])
    ax.grid(plot_grid)
    ax.set_title(name)

    # if visualize:
    #     fig.show()
    # if save_plot:
    fig.savefig(img_path + ".png", dpi=300, bbox_inches='tight')

    return ax, fig