import os
import urllib.request
import gzip

def download_tsplib_data():
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    base_url = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/"
    
    instances = [
        {"prob": "tsp/ulysses22.tsp.gz", "tour": "STSP/ulysses22.opt.tour.gz", "name": "ulysses22"},
        {"prob": "tsp/eil51.tsp.gz", "tour": "STSP/eil51.opt.tour.gz", "name": "eil51"},
        {"prob": "tsp/kroA100.tsp.gz", "tour": "STSP/kroA100.opt.tour.gz", "name": "kroA100"}
    ]

    for item in instances:
        tsp_url = f"{base_url}{item['prob']}"
        tour_url = f"{base_url}{item['tour']}"
        
        tsp_local_gz = os.path.join(data_dir, f"{item['name']}.tsp.gz")
        tour_local_gz = os.path.join(data_dir, f"{item['name']}.opt.tour.gz")
        
        tsp_final = os.path.join(data_dir, f"{item['name']}.tsp")
        tour_final = os.path.join(data_dir, f"{item['name']}.opt.tour")

        try:
            print(f"Downloading {item['name']} problem dataset...")
            urllib.request.urlretrieve(tsp_url, tsp_local_gz)
            with gzip.open(tsp_local_gz, 'rb') as f_in:
                with open(tsp_final, 'wb') as f_out:
                    f_out.write(f_in.read())
            os.remove(tsp_local_gz)

            print(f"Downloading {item['name']} optimal validation tour...")
            urllib.request.urlretrieve(tour_url, tour_local_gz)
            with gzip.open(tour_local_gz, 'rb') as f_in:
                with open(tour_final, 'wb') as f_out:
                    f_out.write(f_in.read())
            os.remove(tour_local_gz)
            
            print(f"Successfully cached dataset pairs for: {item['name']}")
            
        except Exception as e:
            print(f"Failed processing {item['name']}: {e}")

if __name__ == "__main__":
    download_tsplib_data()
