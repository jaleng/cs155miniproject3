# matrix_facorization_vis.py
import numpy as np
import prob2utils as util
import pkl_help as pk
import projections
import getUV as uv
import movies_stats as ms
import matplotlib.pyplot as plt



def make_plot(v1, v2, movies, title):
    x_axis = []
    y_axis = []
    labels = [ms.id_to_movie[i] for i in movies]

    for x in movies:
        x_axis.append(v1[int(x) - 1])
        y_axis.append(v2[int(x) - 1])

    plt.scatter(x_axis, y_axis)
    for label, x, y in zip(labels, x_axis, y_axis):
        plt.annotate(
            label,
            xy=(x, y), textcoords='offset points', xytext=(x-len(label)*3, y+5))

    plt.title(title)
    plt.axhline(0, color='black', linestyle='dashed')
    plt.axvline(0, color='black', linestyle='dashed')
    plt.show()

if __name__ == "__main__":
    ms.make_dict_and_cats()
    
    UV = uv.gettingUV_from_pkl("saved_objs/U_k_20_withbias", "saved_objs/V_k_20_withbias")
    U = UV[0]
    V = UV[1]
    U_proj, V_proj = uv.get_proj_U_V(U, V)
    V_proj = np.array(V_proj)

    make_plot(V_proj[0, :], V_proj[1, :], ms.random(10), "10 Random Movies")
    make_plot(V_proj[0, :], V_proj[1, :], ms.get_most_popular(10), "10 Most Popular Movies")
    make_plot(V_proj[0, :], V_proj[1, :], ms.get_highest_rated(10, 10), "10 Best Movies with more than 10 Reviews")
    make_plot(V_proj[0, :], V_proj[1, :], ms.get_random_from_genre(3, 10), "10 Random Movies from Animation Genre")
    make_plot(V_proj[0, :], V_proj[1, :], ms.get_random_from_genre(7, 10), "10 Random Movies from Documentary Genre")
    make_plot(V_proj[0, :], V_proj[1, :], ms.get_random_from_genre(8, 10), "10 Random Movies from Drama Genre")
