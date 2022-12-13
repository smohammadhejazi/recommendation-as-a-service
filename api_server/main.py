import json
from flask import Flask, request
from kabirrec import RecommendationService

# TODO change .. to .
DEFAULT_DATASET_PATH = "../dataset/ml-100k/"

recommendation_service = RecommendationService()
cold_start = None
similar_items = None
user_specific = None


app = Flask(__name__)


@app.route('/load-csv', methods=["POST"])
def load_csv():
    options = request.get_json()
    path = options.get("path", DEFAULT_DATASET_PATH) + "/"
    recommendation_service.read_csv_data(
        user_info_path=path + "u.user",
        user_ratings_path=path + "u.data",
        item_info_path=path + "u.item",
        info_columns=["user_id", "age", "gender", "occupation", "zip_code"],
        ratings_columns=["user_id", "item_id", "rating", "timestamp"],
        item_columns=["movie_id", "movie_title", "release_date", "video_release_date", "imdb_url", "unknown",
                      "action", "adventure", "animation", "children's", "comedy", "crime", "documentary",
                      "drama", "fantasy", "film_noir", "horror", "musical", "mystery", "romance", "sci-fi",
                      "thriller", "war", "western"],
        info_sep="|", ratings_sep="\t", item_sep="|"
    )
    return json.dumps({"message": "data was successfully loaded"}), 200


@app.route('/start-coldstart', methods=["POST"])
def generate_cold_start():
    options = request.get_json()
    verbose = options.get("verbose", False)
    cold_start = recommendation_service.cold_start_module(options={"verbose": verbose})
    cold_start.fit()
    return json.dumps({"message": "data was successfully loaded"}), 200


@app.route('/coldstart')
def cold_start():
    options = request.get_json()
    verbose = options.get("n", 10)
    items = cold_start.recommend(5)
    item_list = []
    return json.dumps({"items_list"})


if __name__ == "__main__":
    # TODO change port later to 80
    # app.run(host='0.0.0.0', port=80)
    app.run(host='localhost', port=7878)
