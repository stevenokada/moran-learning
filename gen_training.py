import subprocess
import glob
import simulate_moran as smoran
from pymongo import MongoClient

i_datapath = "data/karate/"
fitnesses = [2, 5, 10]
number_of_runs = 500

# for every file in karate
# call 


if __name__ == '__main__':

	try: 
		print("Starting to generate data")
		client = MongoClient("mongodb://skokada:12345@ds115350.mlab.com:15350/moran")
		db = client['moran']
		collection = db['data']


		insertions = []


		counter = 1

		for file in glob.glob(i_datapath + "*"):
			# command = ["python", "node2vec/src/main.py", "--input", file, "--output", file.split(".")[0] + ".rep"]
			# subprocess.call(command)

			# training_data_@fitness.csv

			# make row with file_name, feature vec, p_success @ given fitness, fixation_success @ given fitness


			for fitness in fitnesses:

				p_success, f_time = smoran.aggregate_run(file, fitness, number_of_runs)

				bson_obj = {"input_file": file, "dataset": "karate", "feature_vec": None, "f_val": fitness, "target_vec":{"p_success": p_success, "f_time":f_time}, "number_of_runs": number_of_runs }

				insertions.append(bson_obj)

			print("p_success: %d, f_time: %d" % (p_success, f_time))
			print("Graph %d processed" % counter)

			counter += 1


		collection.insert_many(insertions)




	finally:
		client.close()






