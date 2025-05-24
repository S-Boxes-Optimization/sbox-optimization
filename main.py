from ddt import ddt, get_discrete_vector_representation, display_ddt, evaluate, evaluate_table
from sbox import AsconSBox, DESBox, AESSBox
from search.local_search import Search as LocalSearch
import continuous
import ga
import numpy as np

# DES
# for i in range(1, 9):
#     box = DESBox(i)
#     print(f"{i}:", evaluate(box))
#
# genetic_alg = ga.SBoxGA(n=6, m=4, population_size=400, ngen=150, mutpb=0.25, bijectivity_penalty=0)
# best = genetic_alg.run()
# t = ddt(best, 6, 4)
# print("New one:", evaluate_table(t, 6, 4))

# # ASCON
# box = AsconSBox()
# print("Original:", evaluate(box))
#
# genetic_alg = ga.SBoxGA(n=5, m=5, population_size=400, ngen=150, mutpb=0.25, bijectivity_penalty=0)
# best = genetic_alg.run()
# t = ddt(best, 5, 5)
# print("New one:", evaluate_table(t, 5, 5))

# # AES
box = AESSBox()
print("Original:", evaluate(box))

genetic_alg = ga.SBoxGA(n=8, m=8, population_size=400, ngen=150, mutpb=0.25, bijectivity_penalty=0)
best = genetic_alg.run()
t = ddt(best, 8, 8)
print("New one:", evaluate_table(t, 8, 8))

"""
print(t)
print(np.max(t[1:, :]))
print(np.count_nonzero(t[1:, 0]))
print(np.std(t[1:, 1:]))


t, score = ddt(search.representation, box.input_size, box.output_size)
print(len(search.representation), t.shape)
print(t)
print(np.max(t[1:, :]))
print(np.count_nonzero(t[1:, 0]))
print(np.std(t[1:, 1:]))
#DESBox.print_from_representation(search.representation)
#display_ddt(table)
search.search(100)
final_ddt, final_ddt  = ddt(search.best, box.input_size, box.output_size)
print("Std:", final_ddt, "Final Sbox:")
#display_ddt(final_ddt)

print("Sbox:")
#DESBox.print_from_representation(search.best)
prob_matrix = continuous.sbox_to_prob_matrix_np(search.representation, box.input_size, box.output_size)
print(prob_matrix)
original = continuous.prob_matrix_to_sbox(prob_matrix)
print(continuous.calculate_ddt_prob(prob_matrix))

P_optimizada = continuous.gd_minimize_variance(prob_matrix, 1000, 0.1, 1)
print(P_optimizada)
optimized_sbox = continuous.prob_matrix_to_sbox(P_optimizada)
print(continuous.calculate_ddt_prob(P_optimizada))
print(optimized_sbox)
print(ddt(optimized_sbox, box.input_size, box.output_size))
"""