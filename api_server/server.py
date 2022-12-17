import json
from flask import Flask, request
from kabirrec import RecommendationService
from kabirrec.services import ColdStart
from kabirrec.services import SimilarItems
from kabirrec.services import UserSpecific

# TODO change .. to .
DEFAULT_DATASET_PATH = "../dataset/ml-100k/"

recommendation_service = RecommendationService()
cold_start: ColdStart = None
similar_items: SimilarItems = None
user_specific: UserSpecific = None


app = Flask(__name__)


@app.route('/load-csv', methods=["POST"])
def load_csv():
    options = request.get_json(silent=True)
    if options is None:
        path = DEFAULT_DATASET_PATH
    else:
        path = options.get("path") + "/"

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
def start_cold_start():
    global cold_start
    options = request.get_json(silent=True)
    if options is None:
        verbose = False
    else:
        verbose = options.get("verbose")
    cold_start = recommendation_service.cold_start_module(options={"verbose": verbose})
    cold_start.fit()
    return json.dumps({"message": "ColdStart module is ready for use"}), 200


@app.route('/coldstart', methods=["POST"])
def cold_start():
    options = request.get_json(silent=True)
    n = int(options.get("n", 10))
    items = cold_start.recommend(n)
    return json.dumps({"items_list": items})


@app.route('/start-similaritems', methods=["POST"])
def start_similar_items():
    global similar_items
    options = request.get_json(silent=True)
    if options is None:
        verbose = False
    else:
        verbose = options.get("verbose")
    similar_items = recommendation_service.similar_items_module(options={"verbose": verbose})
    similar_items.fit()
    return json.dumps({"message": "SimilarItems module is ready for use"}), 200


@app.route('/similaritems', methods=["POST"])
def similar_items():
    options = request.get_json(silent=True)
    if options is None:
        return json.dumps({"message": "item_name is required"})
    item_name = options.get("item_name")
    n = int(options.get("n", 10))
    items = similar_items.recommend(item_name, n)
    return json.dumps({"items_list": items})


@app.route('/start-userspecific', methods=["POST"])
def start_user_specific():
    global user_specific
    options = request.get_json(silent=True)
    if options is None:
        verbose = False
        k = None
        top_n = None
    else:
        verbose = options.get("verbose", False)
        k = int(options.get("k", None))
        top_n = int(options.get("top_n", None))

    user_specific = recommendation_service.user_specific_module(options={"verbose": verbose, "k": k, "top_n": top_n})
    user_specific.fit()
    return json.dumps({"message": "UserSpecific module is ready for use"}), 200


@app.route('/userspecific', methods=["POST"])
def user_specific():
    options = request.get_json(silent=True)
    if options is None:
        return json.dumps({"message": "item_name is required"})
    userid = options.get("userid")
    n = int(options.get("n", 10))
    items = user_specific.recommend(userid, n)
    return json.dumps({"items_list": items})


if __name__ == "__main__":
    # TODO change port later to 80
    # app.run(host='0.0.0.0', port=8080)
    app.run(host='localhost', port=7878)
