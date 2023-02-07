import json
import requests
import zipfile
import os
import pickle
from flask import Flask, request
from kabirrec import RecommendationService

CONFIG = {}
app = Flask(__name__)


def read_config():
    global CONFIG
    with open("./config.json") as file:
        CONFIG = json.load(file)


def get_value(dictionary, key, key_type, default):
    if dictionary is None:
        return default
    value = dictionary.get(key)
    if isinstance(value, key_type):
        return value
    return default


def is_token_valid(userid, token):
    # TODO check auth token with core and return false if its not valid
    return True


def load_model(userid, model_name):
    try:
        with open(f"./models/{userid}_{model_name}.obj", mode="rb") as model_file:
            return pickle.load(model_file)
    except FileNotFoundError:
        return None


def save_model(userid, model_name, model_object):
    with open(f"./models/{userid}_{model_name}.obj", mode="wb") as model_file:
        pickle.dump(model_object, model_file)


def is_none(param_list):
    for param in param_list:
        if param is None:
            return True
    return False


def print_log(verbose, message):
    if verbose:
        print(message)


@app.route('/create-model', methods=["POST"])
def create_model():
    # parameters
    options = request.get_json(silent=True)
    userid = get_value(options, "userid", int, None)
    token = request.headers.get("Authorization", None)
    model_name = get_value(options, "model_name", str, None)
    verbose = get_value(options, "verbose", bool, False)

    # parameters check
    if is_none([userid, token, model_name]):
        print_log(verbose, f"Error: userid/token/model_name was invalid | userid:{userid} - model_name:{model_name}")
        return json.dumps({"message": "userid/token/model_name is required"}), 400

    # auth check
    if not is_token_valid(userid, token):
        print_log(verbose, f"Error: Invalid token | userid:{userid} - token:{token}")
        return json.dumps({"message": "token is invalid"}), 400

    # creating model
    recommendation_service = RecommendationService()
    save_model(userid, model_name, recommendation_service)
    print_log(verbose, f"Log: New model created | userid:{userid} - model_name:{model_name}")
    return json.dumps({"message": f"({model_name}) model was created"}), 200


@app.route('/load-csv', methods=["POST"])
def load_csv():
    recommendation_service = None

    # parameters
    options = request.get_json(silent=True)
    userid = get_value(options, "userid", int, None)
    token = request.headers.get("Authorization", None)
    model_name = get_value(options, "model_name", str, None)
    verbose = get_value(options, "verbose", bool, False)
    path = get_value(options, "path", str, CONFIG["datasets_path"]).strip("/")
    name = get_value(options, "name", str, CONFIG["dataset_name"]).strip("/")
    url = get_value(options, "url", str, None)
    extract = get_value(options, "extract", bool, False)
    # CSV
    user_info_path = get_value(options, "user_info_path", str, "u.user")
    user_ratings_path = get_value(options, "user_ratings_path", str, "u.data")
    item_info_path = get_value(options, "item_info_path", str, "u.item")
    user_info_columns = get_value(options, "user_info_columns", str, ["user_id", "age", "gender", "occupation", "zip_code"])
    user_ratings_columns = get_value(options, "ratings_columns", str, ["user_id", "item_id", "rating", "timestamp"])
    item_columns = get_value(options, "item_columns", str, ["movie_id", "movie_title", "release_date",
                                                            "video_release_date", "imdb_url", "unknown",
                                                            "action", "adventure", "animation", "children's",
                                                            "comedy", "crime", "documentary", "drama", "fantasy",
                                                            "film_noir", "horror", "musical", "mystery", "romance",
                                                            "sci-fi", "thriller", "war", "western"])
    user_info_sep = get_value(options, "info_sep", str, "|")
    user_ratings_sep = get_value(options, "ratings_sep", str, "\t")
    item_sep = get_value(options, "item_sep", str, "|")

    # parameters check
    if is_none([userid, token, model_name]):
        print_log(verbose, f"Error: userid/token/model_name was invalid | userid:{userid} - model_name:{model_name}")
        return json.dumps({"message": "userid/token/model_name is required"}), 400

    # auth check
    if not is_token_valid(userid, token):
        print_log(verbose, f"Error: Invalid token | userid:{userid} - token:{token}")
        return json.dumps({"message": "token is invalid"}), 400

    # load model
    recommendation_service = load_model(userid, model_name)
    if recommendation_service is None:
        print_log(verbose, f"Error: Invalid model_name | userid:{userid}")
        return json.dumps({"message": "model_name is invalid"}), 400

    # dataset loading
    print_log(verbose, f"Log: Loading database | userid:{userid} - model_name:{model_name}")
    try:
        if url is not None:
            print_log(verbose, f"Log: Downloading dataset | userid:{userid} - model_name:{model_name}")
            response = requests.get(url)
            zip_path = path + "/" + os.path.basename(url)
            extract_path = path + "/"
            open(zip_path, "wb").write(response.content)
            print_log(verbose, f"Log: Download done | userid:{userid} - model_name:{model_name}")
            if extract:
                print_log(verbose, f"Log: Extracting | userid:{userid} - model_name:{model_name}")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                print_log(verbose, f"Log: Extraction done | userid:{userid} - model_name:{model_name}")
        base = path + "/" + name + "/"
        recommendation_service.read_csv_data(
            user_info_path=base + user_info_path,
            user_ratings_path=base + user_ratings_path,
            item_info_path=base + item_info_path,
            user_info_columns=user_info_columns,
            user_ratings_columns=user_ratings_columns,
            item_columns=item_columns,
            user_info_sep=user_info_sep, user_ratings_sep=user_ratings_sep, item_sep=item_sep
        )
        save_model(userid, model_name, recommendation_service)
        print_log(verbose, f"Log: Database loaded | userid:{userid} - model_name:{model_name}")
        return json.dumps({"message": "data was successfully loaded"}), 200
    except FileNotFoundError as e:
        print_log(verbose, f"Error: Invalid database path/parameters | userid:{userid} - model_name:{model_name}")
        return json.dumps({"message": "invalid path/parameters"}), 400


@app.route('/start-coldstart', methods=["POST"])
def start_cold_start():
    recommendation_service = None

    # parameters
    options = request.get_json(silent=True)
    userid = get_value(options, "userid", int, None)
    token = request.headers.get("Authorization", None)
    model_name = get_value(options, "model_name", str, None)
    verbose = get_value(options, "verbose", bool, False)

    # parameters check
    if is_none([userid, token, model_name]):
        print_log(verbose, f"Error: userid/token/model_name was invalid | userid:{userid} - model_name:{model_name}")
        return json.dumps({"message": "userid/token/model_name is required"}), 400

    # auth check
    if not is_token_valid(userid, token):
        print_log(verbose, f"Error: Invalid token | userid:{userid} - token:{token}")
        return json.dumps({"message": "token is invalid"}), 400

    # load model
    recommendation_service = load_model(userid, model_name)
    if recommendation_service is None:
        print_log(verbose, f"Error: Invalid model_name | userid:{userid}")
        return json.dumps({"message": "model_name is invalid"}), 400

    # dataset check
    if not recommendation_service.data_loaded:
        print_log(verbose, f"Error: Dataset is not loaded yet | userid:{userid} - model_name:{model_name}")
        return json.dumps({"message": "dataset is not loaded yet"}), 400

    # fitting
    print_log(verbose, f"Log: Fitting ColdStart | userid:{userid} - model_name:{model_name}")
    recommendation_service.cold_start_module(options={"verbose": verbose})
    recommendation_service.cold_start.fit()
    print_log(verbose, f"Log: ColdStart is fit | userid:{userid} - model_name:{model_name}")
    save_model(userid, model_name, recommendation_service)
    return json.dumps({"message": f"ColdStart module for model:({model_name}) is ready for use"}), 200


@app.route('/coldstart', methods=["POST"])
def cold_start():
    recommendation_service = None

    # parameters
    options = request.get_json(silent=True)
    userid = get_value(options, "userid", int, None)
    token = request.headers.get("Authorization", None)
    model_name = get_value(options, "model_name", str, None)
    verbose = get_value(options, "verbose", bool, False)
    n = get_value(options, "n", int, 10)

    # parameters check
    if is_none([userid, token, model_name]):
        print_log(verbose, f"Error: userid/token/model_name was invalid | userid:{userid} - model_name:{model_name}")
        return json.dumps({"message": "userid/token/model_name is required"}), 400

    # auth check
    if not is_token_valid(userid, token):
        print_log(verbose, f"Error: Invalid token | userid:{userid} - token:{token}")
        return json.dumps({"message": "token is invalid"}), 400

    # load model
    recommendation_service = load_model(userid, model_name)
    if recommendation_service is None:
        print_log(verbose, f"Error: Invalid model_name | userid:{userid}")
        return json.dumps({"message": "model_name is invalid"}), 400

    # dataset check
    if not recommendation_service.data_loaded:
        print_log(verbose, f"Error: Dataset is not loaded yet | userid:{userid} - model_name:{model_name}")
        return json.dumps({"message": "dataset is not loaded yet"}), 400

    # recommendation
    try:
        items = recommendation_service.cold_start.recommend(n)
        print_log(verbose, f"Log: ColdStart used | userid:{userid} - model_name:{model_name}")
        return json.dumps({"items_list": items}), 200
    except Exception as e:
        print_log(verbose, f"Error: ColdStart is not fit yet| userid:{userid} - model_name:{model_name}")
        return json.dumps({"message": "ColdStart is not fit yet"}), 200


@app.route('/start-similaritems', methods=["POST"])
def start_similar_items():
    recommendation_service = None

    # parameters
    options = request.get_json(silent=True)
    userid = get_value(options, "userid", int, None)
    token = request.headers.get("Authorization", None)
    model_name = get_value(options, "model_name", str, None)
    verbose = get_value(options, "verbose", bool, False)
    n = get_value(options, "n", int, 10)
    algo = get_value(options, "algo", str, "knnbasic")
    sim_options = get_value(options, "sim_options", dict, {"name": "pearson_baseline", "user_based": False})
    bsl_options = get_value(options, "bsl_options", dict, {})
    k = get_value(options, "k", int, 40)
    min_k = get_value(options, "min_k", int, 1)

    # parameters check
    if is_none([userid, token, model_name]):
        print_log(verbose, f"Error: userid/token/model_name was invalid | userid:{userid} - model_name:{model_name}")
        return json.dumps({"message": "userid/token/model_name is required"}), 400

    # auth check
    if not is_token_valid(userid, token):
        print_log(verbose, f"Error: Invalid token | userid:{userid} - token:{token}")
        return json.dumps({"message": "token is invalid"}), 400

    # load model
    recommendation_service = load_model(userid, model_name)
    if recommendation_service is None:
        print_log(verbose, f"Error: Invalid model_name | userid:{userid}")
        return json.dumps({"message": "model_name is invalid"}), 400

    # dataset check
    if not recommendation_service.data_loaded:
        print_log(verbose, f"Error: Dataset is not loaded yet | userid:{userid} - model_name:{model_name}")
        return json.dumps({"message": "dataset is not loaded yet"}), 400

    # fitting
    print_log(verbose, f"Log: Fitting SimilarItem | userid:{userid} - model_name:{model_name}")
    recommendation_service.similar_items_module(
        options={"algo": algo, "k": k, "min_k": min_k,
                 "sim_options": sim_options, "bsl_options": bsl_options, "verbose": verbose})
    recommendation_service.similar_items.fit()
    print_log(verbose, f"Log: SimilarItem is fit | userid:{userid} - model_name:{model_name}")
    save_model(userid, model_name, recommendation_service)
    return json.dumps({"message": f"SimilarItems module for model:({model_name}) is ready for use"}), 200


@app.route('/similaritems', methods=["POST"])
def similar_items():
    recommendation_service = None

    # parameters
    options = request.get_json(silent=True)
    userid = get_value(options, "userid", int, None)
    token = request.headers.get("Authorization", None)
    model_name = get_value(options, "model_name", str, None)
    verbose = get_value(options, "verbose", bool, False)
    item_name = get_value(options, "item_name", str, None)
    n = get_value(options, "n", int, 10)

    # parameters check
    if is_none([userid, token, model_name, item_name]):
        print_log(verbose, f"Error: userid/token/model_name/item_name was invalid | userid:{userid} - model_name:{model_name}")
        return json.dumps({"message": "userid/token/model_name/item_name is required"}), 400

    # auth check
    if not is_token_valid(userid, token):
        print_log(verbose, f"Error: Invalid token | userid:{userid} - token:{token}")
        return json.dumps({"message": "token is invalid"}), 400

    # load model
    recommendation_service = load_model(userid, model_name)
    if recommendation_service is None:
        print_log(verbose, f"Error: Invalid model_name | userid:{userid}")
        return json.dumps({"message": "model_name is invalid"}), 400

    # dataset check
    if not recommendation_service.data_loaded:
        print_log(verbose, f"Error: Dataset is not loaded yet | userid:{userid} - model_name:{model_name}")
        return json.dumps({"message": "dataset is not loaded yet"}), 400

    # recommendation
    try:
        items = recommendation_service.similar_items.recommend(item_name, n)
        print_log(verbose, f"Log: SimilarItems used | userid:{userid} - model_name:{model_name}")
        return json.dumps({"items_list": items})
    except Exception as e:
        print_log(verbose, f"Error: invalid item_name/SimilarItems is not fit | userid:{userid} - model_name:{model_name}")
        return json.dumps({"message": "item_name is invalid/SimilarItems is not fit yet"}), 400


@app.route('/start-userspecific', methods=["POST"])
def start_user_specific():
    recommendation_service = None

    # parameters
    options = request.get_json(silent=True)
    userid = get_value(options, "userid", int, None)
    token = request.headers.get("Authorization", None)
    model_name = get_value(options, "model_name", str, None)
    verbose = get_value(options, "verbose", bool, False)
    k = get_value(options, "k", int, None)
    k_start = get_value(options, "k_start", int, None)
    k_end = get_value(options, "k_end", int, None)

    # parameters check
    if is_none([userid, token, model_name]):
        print_log(verbose, f"Error: userid/token/model_name was invalid | userid:{userid} - model_name:{model_name}")
        return json.dumps({"message": "userid/token/model_name/item_name is required"}), 400

    # auth check
    if not is_token_valid(userid, token):
        print_log(verbose, f"Error: Invalid token | userid:{userid} - token:{token}")
        return json.dumps({"message": "token is invalid"}), 400

    # load model
    recommendation_service = load_model(userid, model_name)
    if recommendation_service is None:
        print_log(verbose, f"Error: Invalid model_name | userid:{userid}")
        return json.dumps({"message": "model_name is invalid"}), 400

    # dataset check
    if not recommendation_service.data_loaded:
        print_log(verbose, f"Error: Dataset is not loaded yet | userid:{userid} - model_name:{model_name}")
        return json.dumps({"message": "dataset is not loaded yet"}), 400

    # fitting
    if k is None:
        if k_start is not None and k_start < 2:
            return json.dumps({"message": "k_start should should be bigger than 1"}), 400
        if k_end is not None and k_end > recommendation_service.user_info.shape[0]:
            return json.dumps({"message": "k_start should should be smaller than number of users"}), 400
        if k_start is not None and k_end is not None and k_start >= k_end:
            return json.dumps({"message": "k_start should be smaller than k_end"}), 400

    print_log(verbose, f"Log: Fitting UserSpecific | userid:{userid} - model_name:{model_name}")
    recommendation_service.user_specific_module(options=
                                                {"verbose": verbose, "k": k, "k_start": k_start, "k_end": k_end})
    recommendation_service.user_specific.fit()
    print_log(verbose, f"Log: UserSpecific is fit | userid:{userid} - model_name:{model_name}")
    save_model(userid, model_name, recommendation_service)
    return json.dumps({"message": f"UserSpecific module for model:({model_name}) is ready for use"}), 200


@app.route('/userspecific', methods=["POST"])
def user_specific():
    recommendation_service = None

    # parameters
    options = request.get_json(silent=True)
    userid = get_value(options, "userid", int, None)
    token = request.headers.get("Authorization", None)
    model_name = get_value(options, "model_name", str, None)
    verbose = get_value(options, "verbose", bool, False)
    dataset_userid = get_value(options, "dataset_userid", int, None)
    n = get_value(options, "n", int, 10)

    # parameters check
    if is_none([userid, token, model_name, dataset_userid]):
        print_log(verbose, f"Error: userid/token/dataset_userid was invalid | userid:{userid} - model_name:{model_name}")
        return json.dumps({"message": "userid/token/model_name/dataset_userid is required"}), 400

    # auth check
    if not is_token_valid(userid, token):
        print_log(verbose, f"Error: Invalid token | userid:{userid} - token:{token}")
        return json.dumps({"message": "token is invalid"}), 400

    # load model
    recommendation_service = load_model(userid, model_name)
    if recommendation_service is None:
        print_log(verbose, f"Error: Invalid model_name | userid:{userid}")
        return json.dumps({"message": "model_name is invalid"}), 400

    # dataset check
    if not recommendation_service.data_loaded:
        print_log(verbose, f"Error: Dataset is not loaded yet | userid:{userid} - model_name:{model_name}")
        return json.dumps({"message": "dataset is not loaded yet"}), 400

    try:
        print_log(verbose, f"Log: UserSpecific used | userid:{userid} - model_name:{model_name}")
        items = recommendation_service.user_specific.recommend(dataset_userid, n)
        return json.dumps({"items_list": items})
    except Exception as e:
        print_log(verbose, f"Error: invalid dataset_userid/UserSpecific is not fit | userid:{userid} - model_name:{model_name}")
        return json.dumps({"message": "dataset_userid is invalid/UserSpecific is not fit"}), 400


if __name__ == "__main__":
    read_config()
    app.run(host=CONFIG["host"], port=CONFIG["port"])
