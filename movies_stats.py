# movies.py
import numpy as np

id_to_movie = {}
cats = [[] for _ in range(19)]

def make_dict_and_cats():
    data = np.genfromtxt("movies.txt", dtype='str', delimiter="\t")

    for x in data:
        id_to_movie[int(x[0])] = x[1]
        counter = -1
        for y in x[2:]:
            counter += 1
            if y == "1":
                cats[counter].append(int(x[0]))

def get_most_popular(n_pop):
    num_of_reviews = {}
    pop_ids = []
    data = np.genfromtxt("data.txt", dtype='int', delimiter="\t")
    for x in data:
        if not (x[1] in num_of_reviews.keys()):
            num_of_reviews[x[1]] = 1
        else:
            num_of_reviews[x[1]] += 1
    for x in range(n_pop):
        temp1 = max(num_of_reviews, key= (lambda k: (num_of_reviews[k])))
        pop_ids.append(temp1)
        num_of_reviews[temp1] = 0
    return pop_ids

def random(n_movies):
    return np.random.choice(max(id_to_movie.keys()), n_movies)


 #  Could use or just for reference   
    # unknown = cats[0]
    # action = cats[1]
    # adventure = cats[2]
    # animation = cats[3]
    # chlidrens = cats[4]
    # comedy = cats[5]
    # crime = cats[6]
    # documentary = cats[7]
    # drama = cats[8]
    # fantasy = cats[9]
    # film_noir = cats[10]
    # horror = cats[11]
    # musical = cats[12]
    # mystery = cats[13]
    # romance = cats[14]
    # sci_fi = cats[15]
    # thriller = cats[16]
    # war = cats[17]
    # western = cats[18]

def get_random_from_genre(genre, n_movies):
    return np.random.choice(cats[genre], n_movies)

def get_highest_rated(n_rated, min_number_of_ratings):
    num_of_reviews = {}
    sum_of_reviews = {}
    pop_ids = []
    data = np.genfromtxt("data.txt", dtype='float', delimiter="\t")
    for x in data:
        if not (x[1] in num_of_reviews.keys()):
            num_of_reviews[x[1]] = 1.0
            sum_of_reviews[x[1]] = x[2]
        else:
            num_of_reviews[x[1]] += 1.0
            sum_of_reviews[x[1]] += x[2]
    for x in sum_of_reviews.keys():
        if(num_of_reviews[x] < min_number_of_ratings):
            sum_of_reviews[x] = 0
        else:
            sum_of_reviews[x] = sum_of_reviews[x] / num_of_reviews[x]
    for x in range(n_rated):
        temp1 = max(sum_of_reviews, key= (lambda k: (sum_of_reviews[k])))
        pop_ids.append(temp1)
        sum_of_reviews[temp1] = 0
    return pop_ids

if __name__ == "__main__":
    make_dict_and_cats()
    get_highest_rated(10, 10)
    get_most_popular(10)
    for x in get_highest_rated(10, 10):
        print id_to_movie[x]