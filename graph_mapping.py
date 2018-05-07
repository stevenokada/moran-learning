import networkx as nx

class Constant():

	GRAPH_NAME_MAP = {'balanced_tree': nx.balanced_tree(2,5),
					'barbell_graph': nx.barbell_graph(50,50),
					'complete_graph': nx.complete_graph(100),
					'complete_multipartite_graph': nx.complete_multipartite_graph(25,25,25,25),
					'circular_ladder_graph': nx.circular_ladder_graph(50),
					'cycle_graph': nx.cycle_graph(100),
					'dorogovtsev_goltsev_mendes_graph': nx.dorogovtsev_goltsev_mendes_graph(5),
					'grid_2d_graph': nx.grid_2d_graph(10,10),
					'hypercube_graph': nx.hypercube_graph(6),
					'ladder_graph': nx.ladder_graph(50),
					'lollipop_graph': nx.lollipop_graph(50,50),
					'path_graph': nx.path_graph(100),
					'star_graph': nx.star_graph(100),
					'wheel_graph': nx.wheel_graph(100),

					'bull_graph': nx.bull_graph(),
					'chvatal_graph': nx.chvatal_graph(),
					'desargues_graph': nx.desargues_graph(),
					'diamond_graph': nx.diamond_graph(),
					'dodecahedral_graph': nx.dodecahedral_graph(),
					'frucht_graph': nx.frucht_graph(),
					'heawood_graph': nx.heawood_graph(),
					'house_x_graph': nx.house_x_graph(),
					'icosahedral_graph': nx.icosahedral_graph(),
					'krackhardt_kite_graph': nx.krackhardt_kite_graph(),
					'moebius_kantor_graph': nx.moebius_kantor_graph(),
					'octahedral_graph': nx.octahedral_graph(),
					'pappus_graph': nx.pappus_graph(),
					'petersen_graph': nx.petersen_graph(),
					'sedgewick_maze_graph': nx.petersen_graph(),
					'tetrahedral_graph': nx.tetrahedral_graph(),
					'truncated_cube_graph': nx.truncated_cube_graph(),
					'truncated_tetrahedron_graph': nx.truncated_tetrahedron_graph(),
					'tutte_graph': nx.tutte_graph(),

					'karate_club_graph': nx.karate_club_graph(),
					'davis_southern_women_graph': nx.davis_southern_women_graph(),
					'florentine_families_graph': nx.florentine_families_graph(),

					}


def populate_dict():
	for i in range(50):
		create_graph(i)

def string_to_graph():

