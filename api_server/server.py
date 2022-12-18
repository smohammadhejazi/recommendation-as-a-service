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


def get_value(dictionary, key, key_type, default):
    if dictionary is None:
        return default
    value = dictionary.get(key)
    if isinstance(value, key_type):
        return value
    return default


@app.route('/load-csv', methods=["POST"])
def load_csv():
    options = request.get_json(silent=True)
    path = get_value(options, "path", str, DEFAULT_DATASET_PATH)

    try:
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
    except FileNotFoundError as e:
        return json.dumps({"message": "path is invalid"}), 400
    return json.dumps({"message": "data was successfully loaded"}), 200


@app.route('/start-coldstart', methods=["POST"])
def start_cold_start():
    global cold_start
    options = request.get_json(silent=True)
    verbose = get_value(options, "verbose", bool, False)

    cold_start = recommendation_service.cold_start_module(options={"verbose": verbose})
    cold_start.fit()
    return json.dumps({"message": "ColdStart module is ready for use"}), 200


@app.route('/coldstart', methods=["POST"])
def cold_start():
    options = request.get_json(silent=True)
    n = get_value(options, "n", int, 10)

    items = cold_start.recommend(n)
    return json.dumps({"items_list": items})


@app.route('/start-similaritems', methods=["POST"])
def start_similar_items():
    global similar_items
    options = request.get_json(silent=True)
    verbose = get_value(options, "verbose", bool, False)

    similar_items = recommendation_service.similar_items_module(options={"verbose": verbose})
    similar_items.fit()
    return json.dumps({"message": "SimilarItems module is ready for use"}), 200


@app.route('/similaritems', methods=["POST"])
def similar_items():
    options = request.get_json(silent=True)
    if options is None or "item_name" not in options:
        return json.dumps({"message": "item_name is required"})
    item_name = get_value(options, "item_name", str, None)
    n = get_value(options, "n", int, 10)

    if item_name is None:
        return json.dumps({"message": "item_name is invalid"}), 400
    try:
        items = similar_items.recommend(item_name, n)
    except Exception as e:
        return json.dumps({"message": "item_name is invalid"}), 400
    return json.dumps({"items_list": items})


@app.route('/start-userspecific', methods=["POST"])
def start_user_specific():
    global user_specific
    options = request.get_json(silent=True)
    verbose = get_value(options, "verbose", bool, False)
    k = get_value(options, "k", int, None)
    k_start = get_value(options, "k_start", int, None)
    k_end = get_value(options, "k_end", int, None)
    if k is None:
        if k_start is not None and k_start < 2:
            return json.dumps({"k_start": "k_start should should be bigger than 1"}), 400
        if k_end is not None and k_end > recommendation_service.user_info.shape[0]:
            return json.dumps({"k_end": "k_start should should be smaller than number of users"}), 400
        if k_start is not None and k_end is not None and k_start >= k_end:
            return json.dumps({"message": "k_start should be smaller than k_end"}), 400

    user_specific = recommendation_service.user_specific_module(options={"verbose": verbose, "k": k})
    user_specific.fit(k_start=k_start, k_end=k_end)
    return json.dumps({"message": "UserSpecific module is ready for use"}), 200


@app.route('/userspecific', methods=["POST"])
def user_specific():
    options = request.get_json(silent=True)
    if options is None or "userid" not in options:
        return json.dumps({"message": "userid is required"})
    userid = get_value(options, "userid", int, None)
    n = get_value(options, "n", int, 10)

    if userid is None:
        return json.dumps({"message": "userid is invalid"}), 400
    try:
        items = user_specific.recommend(userid, n)
    except Exception as e:
        return json.dumps({"message": "userid is invalid"}), 400
    return json.dumps({"items_list": items})


if __name__ == "__main__":
    # TODO change port later to 80
    # app.run(host='0.0.0.0', port=8080)
    app.run(host='localhost', port=7878)
