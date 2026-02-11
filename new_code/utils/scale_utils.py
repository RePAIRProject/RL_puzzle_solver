#!/usr/bin/python3

from shapely import (
    affinity,
    Polygon,
    MultiPolygon,
    MultiPoint,
)

def scale_fresco(fresco_polygons, scale_factor):
    # Scale up fresco
    sf = 1.0 + scale_factor
    scaled_polygons = affinity.scale(
        fresco_polygons, sf, sf
    )

    # Shift old fragments according to new centroids
    new_polygons = []
    new_centroids = []
    for i in range(len(fresco_polygons.geoms)):
        old_centroid = fresco_polygons.geoms[i].centroid
        new_centroid = scaled_polygons.geoms[i].centroid

        shifted_fragment = affinity.translate(
            fresco_polygons.geoms[i],
            new_centroid.x-old_centroid.x,
            new_centroid.y-old_centroid.y,
        )

        new_polygons.append(shifted_fragment)
        new_centroids.append(new_centroid)

    # group all polygons into a shapely multipolygon
    fresco_polygons_scaled = MultiPolygon(new_polygons)
    fresco_centroids_scaled = MultiPoint(new_centroids)

    return fresco_polygons_scaled, fresco_centroids_scaled
    
def combine_centroids_and_vertices(centroids, vertices):
    fresco_data = {}

    for i, (centroid_list, vertex_list) in enumerate(zip(centroids, vertices)):
        fresco_data[i] = {
            'centroids': centroid_list,
            'vertices': vertex_list
        }

    return fresco_data

def convert_fresco_array_to_shapely_multi_polygon(fresco_array):
    fresco_polygons = []

    for i in range(len(fresco_array)):
        fragment = Polygon(fresco_array[i])
        fresco_polygons.append(fragment)

    return MultiPolygon(fresco_polygons)  

def convert_shapely_multi_polygon_to_array_of_shapely_polygons(shapely_multi_polygon):
    polygon_array = []

    for i in range(len(shapely_multi_polygon.geoms)):
        polygon = Polygon(shapely_multi_polygon.geoms[i])
        polygon_array.append(polygon)

    return polygon_array

def inflate_fresco(multi_polygon, inflation_factor):
    multi_polygon_array = convert_shapely_multi_polygon_to_array_of_shapely_polygons(multi_polygon)
    inflated_fresco = []
    for fragment in multi_polygon_array:
        inflated_fragment = fragment.buffer(inflation_factor)
        inflated_fresco.append(inflated_fragment)

    inflated_fresco = MultiPolygon(inflated_fresco)
    return inflated_fresco

def check_for_fragment_overlap_in_fresco(multipolygon):
    # Extract individual polygons from the multipolygon
    polygons = convert_shapely_multi_polygon_to_array_of_shapely_polygons(multipolygon)
    
    # Iterate through each pair of polygons
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            # Check if the two polygons overlap
            if polygons[i].overlaps(polygons[j]):
                return True
    return False

def get_min_inflated_gt_fresco(gt_multi_polygon, inflation_width=0.00635, sf_factor = 0.001, visualize=False, save_plot=False):
    inflated_gt_multi_polygon = inflate_fresco(gt_multi_polygon, inflation_width)
    scaled_and_inflated_gt_multi_polygon = inflated_gt_multi_polygon
    scaled_gt_multi_polygon = gt_multi_polygon
    overlap = True
    while overlap == True:
        scaled_and_inflated_gt_multi_polygon, _ = scale_fresco(scaled_and_inflated_gt_multi_polygon, scale_factor=sf_factor)
        scaled_gt_multi_polygon, _ = scale_fresco(scaled_gt_multi_polygon, scale_factor=sf_factor)
        # Check if any shapes in the inflated fresco overlaps
        overlap = check_for_fragment_overlap_in_fresco(scaled_and_inflated_gt_multi_polygon)
    
    min_inflated_gt_fresco = scaled_gt_multi_polygon#scaled_and_inflated_gt_multi_polygon
    
    return min_inflated_gt_fresco