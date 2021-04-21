import torch
import torch.nn as nn

import pdb
import argparse
import random
from tqdm import tqdm


from common import models


class NetCrossInteractionLayersz(models.net_crossInteraction):

    def __init__(self, lambda_l1, lambda_fused, lambda_group, lambda_bind):
        super().__init__(lambda_l1=lambda_l1, lambda_fused=lambda_fused, lambda_group=lambda_group, lambda_bind=lambda_bind)

        self.name = "ArghaCustomModel"

    def forward_embedding(self, prot_data2):
        # protein embedding 2
        aminoAcid_embedding2 = self.mod_aminoAcid_embedding(prot_data2)

        b, i, j, d = aminoAcid_embedding2.size()
        prot_seq_embedding2 = aminoAcid_embedding2.reshape(b * i, j, d)
        prot_seq_embedding2, _ = self.gru0(prot_seq_embedding2)
        prot_seq_embedding2 = prot_seq_embedding2.reshape(b * j, i, d)
        prot_seq_embedding2, _ = self.gru1(prot_seq_embedding2)
        prot_seq_embedding2 = prot_seq_embedding2.reshape(b, i * j, d)

        return prot_seq_embedding2
