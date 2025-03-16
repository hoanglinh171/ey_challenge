import os
import yaml
import osmnx as ox
import networkx as nx
import momepy
from shapely.geometry import box, LineString
import numpy as np

with open("data_pipeline/config.yaml", "r") as file:
    config = yaml.safe_load(file)

COORDS = config['coords']


def load_engineer_street_data(savefolder):
    bbox = box(*COORDS)
    G = ox.graph_from_polygon(bbox, network_type='drive')
    G_proj = ox.project_graph(G, to_crs="EPSG:26918")

    # Node degree
    degree_dict = dict(G_proj.degree())
    nx.set_node_attributes(G_proj, degree_dict, "degree")
    
    # Centrality
    primal = momepy.closeness_centrality(G_proj, radius=1000, name="closeness_1000", distance="length", weight="length")
    primal = momepy.closeness_centrality(primal, radius=500, name="closeness_500", distance="length", weight="length")
    primal = momepy.closeness_centrality(primal, radius=100, name="closeness_100", distance="length", weight="length")
    # primal = momepy.closeness_centrality(primal, name="closeness_global", distance="length", weight="length")

    # primal = momepy.betweenness_centrality(primal, radius=1000, name="betweenness_node_1000", mode="nodes", weight="length")
    # primal = momepy.betweenness_centrality(primal, radius=500, name="betweenness_node_500", mode="nodes", weight="length")
    # primal = momepy.betweenness_centrality(primal, radius=100, name="betweenness_node_100", mode="nodes", weight="length")
    # primal = momepy.betweenness_centrality(primal, name="betweenness_node_global", mode="nodes", weight="length")

    # primal = momepy.betweenness_centrality(primal, radius=1000, name="betweenness_edge_1000", mode="edges", weight="length")
    # primal = momepy.betweenness_centrality(primal, radius=500, name="betweenness_edge_500", mode="edges", weight="length")
    # primal = momepy.betweenness_centrality(primal, radius=100, name="betweenness_edge_100", mode="edges", weight="length")
    # primal = momepy.betweenness_centrality(primal, name="betweenness_edge_global", mode="edges", weight="length")

    # Get node and edge
    nodes, edges = momepy.nx_to_gdf(primal)

    # Drop null geometry
    nodes = nodes.dropna(subset='geometry')
    edges = edges.dropna(subset='geometry')
    
    # Recalculate length
    # edges['length'] = edges.apply(lambda row: row['length'] * len(row['osmid']) 
    #                               if isinstance(row['osmid'], list) else row['length'],
    #                               axis=1)

    # Edge circuity
    edges['straight'] = edges.geometry.apply(lambda geom: geom.coords[0] 
                                             if geom is not None and isinstance(geom, LineString) else None)
    edges['straight_end'] = edges.geometry.apply(lambda geom: geom.coords[-1]
                                                 if geom is not None and isinstance(geom, LineString) else None)
    edges['straight_dist'] = edges.apply(
        lambda row: np.linalg.norm(np.array(row['straight']) - np.array(row['straight_end'])),
        axis=1
    )
    edges['circuity'] = edges['length'] / edges['straight_dist']
    edges['circuity'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Orientation
    edges['orientation'] = momepy.Orientation(edges).series

    nodes = nodes.to_crs(epsg=4326)
    edges = edges.to_crs(epsg=4326)

    nodes.to_file(savefolder + "nodes.geojson", driver="GeoJSON")
    edges.to_file(savefolder + "edges.geojson", driver="GeoJSON")



if __name__ == "__main__":
    SAVE_DIR = "data_pipeline/data/features_extracted/"
    load_engineer_street_data(SAVE_DIR)






