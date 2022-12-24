![KabirRec](https://github.com/smohammadhejazi/recommendation-as-a-service/raw/main/logo.svg "KabirRec")

# Overview
**KabirRec** is a Python recommendation service that uses **surprise** and **scikit-learn**.

It has three main services:
- Cold Start Recommendation: Recomnends the most polular items
- Similar Items Recommendation: Recomnends similar items to the item given
- User Specific Recommendation: Recommends the best items to a user based on their history.

# Getting started
Its easy to use each service, first you need to create the RecommendationService object. Then read your csv data. We use ml-100k which has user info, user ratings, item info. We need to specify the columns as well as each csv seperators.
```
recommendation_service = RecommendationService()

recommendation_service.read_csv_data(
    user_info_path="../dataset/ml-100k/u.user",
    user_ratings_path="../dataset/ml-100k/u.data",
    item_info_path="../dataset/ml-100k/u.item",
    info_columns=["user_id", "age", "gender", "occupation", "zip_code"],
    ratings_columns=["user_id", "item_id", "rating", "timestamp"],
    item_columns=["movie_id", "movie_title", "release_date", "video_release_date", "imdb_url", "unknown",
    "action", "adventure", "animation", "children's", "comedy", "crime", "documentary",
    "drama", "fantasy", "film_noir", "horror", "musical", "mystery", "romance", "sci-fi",
    "thriller", "war", "western"],
    info_sep="|", ratings_sep="\t", item_sep="|"
)
```

Then you can call on generators for services to get the service object. First call the fit function and then you can use their prediction services.
## User Specific Example
```
user_specific = recommendation_service.user_specific_module(options={"verbose": True, "k": 26})
user_specific.fit(20, 30)
user_specific.draw_clusters_graph()
prediction_rating = user_specific.recommend(1, 4)
print(prediction_rating)
prediction_rating = user_specific.predict_rating(1, 1)
print(prediction_rating.est)
```
Output:
```
Fitting the algorithm...
Finding optimal cluster...
Optimal cluster k=23 found.
Clustering with k=23...
Clustering done.
Building tables...
Tables are built.
Fitting is done.
[(1189, 'Prefontaine (1997)', 5), (1500, 'Santa with Muscles (1996)', 5), (814, 'Great Day in Harlem, A (1994)', 5), (1536, 'Aiqing wansui (1994)', 5)]
3.603819651096398
```
![Clusters Info](https://github.com/smohammadhejazi/recommendation-as-a-service/raw/main/examples_output/user_specific_plot.png "Clusters Info")
## Similar Item Example
```
similar_items = recommendation_service.similar_items_module(options={"verbose": True})
similar_items.fit()
items = similar_items.recommend("Toy Story (1995)", k=5)
for movie in items:
    print(movie)
```
Output:
```
Fitting the algorithm...
Estimating biases using als...
Computing the pearson_baseline similarity matrix...
Done computing similarity matrix.
Fitting is done.
Finding 5 nearest items...
5 nearest items found.
Beauty and the Beast (1991)
Raiders of the Lost Ark (1981)
That Thing You Do! (1996)
Lion King, The (1994)
Craft, The (1996)
```
## Cold Start Example
```
cold_start = recommendation_service.cold_start_module(options {"verbose": True})
cold_start.fit()
items = cold_start.recommend(5)
for movie in items:
    print(movie)
```
Output:
```
Ranking items...
All items are ranked.
     item_id  rating
49        50    2541
99       100    2111
180      181    2032
257      258    1936
173      174    1786
```
## API Server
**/api_server** folder contains an API server desinged to use Kabirrec as a live service. It can load database, fit the algorithm and use them live by sending HTTP request to it's different routes.

Each route takes a post request with a JSON object as it's options and reponds a JSON object. There are lots of options that can be used to further optimize the algorithm to your own needs but if none is given, the default setting will be applyed.

These routes and their JSON options are as follows:
Notes: 
- All the routes begin with the server name:port and then the route
- All the routes require POST request.
- The default value is written in parentheses.

For configuring sim_options and bsl_options please check the following [link](https://surprise.readthedocs.io/en/stable/prediction_algorithms.html# "configuration options").

Routes and options:
- **/load-csv** : Load csv database (it should be in the format of MovieLens ml-100k)
	- path: path of the database ("./dataset/ml-100k/")
	- verbose: output logs (false)
    
	returns JSON object {"message": "message content"}


- **/start-coldstart** : Start ColdStart module
	- verbose: output logs (false)

	returns JSON object {"message": "message content"}


- **/coldstart**: Use ColdStart module
	- n: number of item recommendations (10)
	- verbose: output logs (false)

	returns JSON object {"items_list": [list of popular items]}


- **/start-similaritems** : Start SimilarItems module
	- algo: path of the database ("knnbasic")
	- k: The (max) number of neighbors to take into account for aggregation (40)
	- min_k: The minimum number of neighbors to take into account foraggregation. If there are not enough neighbors, the prediction is set to the global mean of all ratings (1)
	- sim_options: A dictionary of options for the similarity measure ({"name": "pearson_baseline", "user_based": False})
	- bsl_options: A dictionary of options for the baseline estimates computation. Only when algo is KNNBaseline (empty dict {})
	- verbose: output logs (false)

	returns JSON object {"message": "message content"}


- **/similaritems** : Use SimilarItems module
	- item_name: name of the item (must be given)
	- n: number of item recommendations (10)

	returns JSON object {"items_list": [list of popular items]}


- **/start-userspecific** : Start UserSpecific module
	- k: number of the manual clusters, if given k_start and k_end are not taken into account (None)
	- k_start: start number of the range to look for optimal cluster, range[k_start, k_end] (2)
	- k_end: end number of the range to look for optimal cluster, range[k_start, k_end] (number of users / 2)
	- verbose: output logs (false)

	returns JSON object {"message": "message content"}


- **/userspecific** : Use UserSpecific module
	- item_name: name of the user (must be given)
	- n: number of item recommendations (10)

	returns JSON object {"items_list": [list of popular items]}


## More Information
For more information on how to use the library, look at comments in the codes.

# Installation
You can use pip (you'll need a C compiler and numpy library):
```
$ pip install kabirrec
```
You can also clone the repo and build the library:
```
$ pip install numpy cython
$ git clone https://github.com/smohammadhejazi/recommendation-as-a-service
$ cd recommendation-as-a-service
$ python setup.py install
```


