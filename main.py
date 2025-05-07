from ddt import ddt, get_discrete_vector_representation, display_ddt
from sbox import AsconSBox, DESBox
from search.local_search import Search as LocalSearch

box = DESBox()

search = LocalSearch(box)
DESBox.print_from_representation(search.representation)
display_ddt(ddt(search.representation, box.input_size, box.output_size)[0])
search.search(100)
final_ddt, final_std = ddt(search.best, box.input_size, box.output_size)
display_ddt(final_ddt)

print("Sbox:")
DESBox.print_from_representation(search.best)
