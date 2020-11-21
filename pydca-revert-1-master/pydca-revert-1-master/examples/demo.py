"""pydca demo

Author: Mehari B. Zerihun
"""

# import pydca modules
from pydca.plmdca import plmdca
from pydca.meanfield_dca import meanfield_dca
from pydca.sequence_backmapper import sequence_backmapper
from pydca.msa_trimmer import msa_trimmer
from pydca.contact_visualizer import contact_visualizer
from pydca.dca_utilities import dca_utilities
import time
from pydca.fasta_reader import fasta_reader

st_time = time.time()

#rna_msa_file = '/home/at/work/senior-design/pydca-revert-1-master/pydca-revert-1-master/examples/1B70_A_1B70_B.fas'
#rna_msa_file = '/home/at/work/senior-design/pydca-revert-1-master/pydca-revert-1-master/examples/MSA_RF00167.fa'
rna_msa_file  = '/home/at/work/senior-design/pydca-revert-1-master/pydca-revert-1-master/examples/Untitled.fasta'
rna_refseq_file = '/home/at/work/senior-design/pydca-revert-1-master/pydca-revert-1-master/examples/Untitled_refseq.fasta'
    #'/home/at/work/senior-design/pydca-revert-1-master/pydca-revert-1-master/examples/ref_RF00167.fa'
#'/home/at/work/senior-design/pydca-revert-1-master/pydca-revert-1-master/examples/MSA_RF00167_Trimmed.fa'

# create MSATrimmer instance
trimmer = msa_trimmer.MSATrimmer(
    rna_msa_file, biomolecule='protein', refseq_file=rna_refseq_file,
    )
#refseq_file=rna_refseq_file,
trimmed_data = trimmer.get_msa_trimmed_by_refseq(remove_all_gaps=True)
#trimmed_data = trimmer.get_msa_trimmed_by_gap_size(remove_all_gaps=True)
#trimmed_data = trimmer.get_msa()

#write trimmed msa to file in FASTA format
trimmed_data_outfile = 'MSA_RF00167_Trimmed.fa'
with open(trimmed_data_outfile, 'w') as fh:
    for seqid, seq in trimmed_data:
        fh.write('>{}\n{}\n'.format(seqid, seq))
# Compute DCA scores using Pseudolikelihood maximization algorithm

plmdca_inst = plmdca.PlmDCA(
    trimmed_data_outfile,
    'protein'
    '',
    seqid = 0.8,
    lambda_h = 1.0,
    lambda_J = 20.0,
    num_threads = 1,
    max_iterations = 500,
)

# compute DCA scores summarized by Frobenius norm and average product corrected
plmdca_FN_APC = plmdca_inst.compute_sorted_FN_APC()
#for site_pair, score in plmdca_FN_APC[:5]:
 #   print(site_pair, score)
#create mean-field DCA instance
mfdca_inst = meanfield_dca.MeanFieldDCA(
    trimmed_data_outfile,
    'protein',
    pseudocount = 0.5,
    seqid = 0.8,
)

# Compute average product corrected Frobenius norm of the couplings
#mfdca_FN_APC = mfdca_inst.compute_sorted_FN_APC()
#for site_pair, score in mfdca_FN_APC[:5]:
 #   print(site_pair, score)

#print("mfdca_FN_APC")
#for obj in mfdca_FN_APC:
 #   print(obj[0][0],obj[0][1],obj[1])

#print("plmdca_FN_APC")
for obj in plmdca_FN_APC:
    print((obj[0][0]+1),'\t', (obj[0][1]+1),'\t', obj[1])

print(time.time()-st_time)

#exit(0)

#plmdca_visualizer = contact_visualizer.DCAVisualizer('rna', 'x', '1y26',
plmdca_visualizer = contact_visualizer.DCAVisualizer('protein', 'b', '4fwi',
    refseq_file = rna_refseq_file,
    sorted_dca_scores = plmdca_FN_APC,
    linear_dist = 4,
    contact_dist = 8.0,
)
contact_map_data = plmdca_visualizer.plot_contact_map()
tp_rate_data = plmdca_visualizer.plot_true_positive_rates()
mfdca_visualizer = contact_visualizer.DCAVisualizer('protein', 'x', '1y26',
    refseq_file = rna_refseq_file,
    sorted_dca_scores = mfdca_FN_APC,
    linear_dist = 4,
    contact_dist = 8.0,
)
contact_map_data = mfdca_visualizer.plot_contact_map()
tp_rate_data = mfdca_visualizer.plot_true_positive_rates()
fields_plm, couplings_plm = plmdca_inst.compute_params(ranked_by='di_apc', num_site_pairs=100)

