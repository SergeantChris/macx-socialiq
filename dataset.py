import h5py
from tqdm import tqdm
import random

import numpy as np
import torch
from torch.utils.data import Dataset

import folds


class SocialIQ(Dataset):

    def __init__(self, root, phase, mods, a4=False):
        self.root = root

        self.h_qa = None
        if not a4:
            self.h_qa = h5py.File(self.root + '/socialiq/preextr/SOCIAL-IQ_QA_BERT_LASTLAYER_BINARY_CHOICE.csd', 'r')
        else:
            self.h_qa = h5py.File(self.root + '/Social-IQ/code/socialiq/SOCIAL-IQ_QA_BERT_MULTIPLE_CHOICE.csd', 'r')

        self.h_v = None
        if 'v' in mods:
            self.h_v = h5py.File(self.root + '/socialiq/preextr/SOCIAL_IQ_DENSENET161_1FPS.csd', 'r')

        self.h_t = None
        if 't' in mods:
            self.h_t = h5py.File(self.root + '/Social-IQ/code/socialiq/SOCIAL_IQ_TRANSCRIPT_RAW_CHUNKS_BERT.csd', 'r')

        self.h_ac = None
        if 'ac' in mods:
            self.h_ac = h5py.File(self.root + '/Social-IQ/code/socialiq/SOCIAL_IQ_COVAREP.csd', 'r')

        if phase == 'train':
            fold = folds.standard_train_fold
        else:
            fold = folds.standard_valid_fold

        self.samples = []
        for sid in tqdm(fold):

            feats = ()

            if 'ac' in mods:
                acfeats = torch.from_numpy(
                    np.nan_to_num(self.h_ac['SOCIAL_IQ_COVAREP']['data'][sid]['features'][()], posinf=1e3, neginf=-1e3))
                ac_1ps = []
                prev_ps = acfeats.shape[0] // 60
                for i in range(0, acfeats.shape[0], prev_ps):
                    ac_1ps.append(acfeats[i:i + prev_ps].mean(0))
                acfeats = torch.stack(ac_1ps[:60])
                feats = (acfeats,) + feats

            if 'v' in mods:
                visfeats = torch.from_numpy(self.h_v['SOCIAL_IQ_DENSENET161_1FPS']['data'][sid]['features'][()])
                feats = (visfeats,) + feats

            if 't' in mods:
                subtitles = torch.from_numpy(
                    self.h_t['SOCIAL_IQ_TRANSCRIPT_RAW_CHUNKS_BERT']['data'][sid]['features'][()]).view(-1, 9216)[:512,
                            -768:]
                toks = subtitles.shape[0]
                if toks < 512:
                    subtitles = torch.cat([subtitles, torch.zeros((512 - toks, subtitles.shape[1]))], dim=0)
                feats = (subtitles,) + feats

            if not a4:
                qa_embs = torch.from_numpy(
                    self.h_qa['SOCIAL-IQ_QA_BERT_LASTLAYER_BINARY_CHOICE']['data'][sid]['features'][0])
            else:
                qa_embs = torch.from_numpy(self.h_qa['SOCIAL-IQ_QA_BERT_MULTIPLE_CHOICE']['data'][sid]['features'][0])
            for i_q, question in enumerate(qa_embs):
                for i_c, combination in enumerate(question):
                    enum = list(enumerate(combination[1:]))
                    random.shuffle(enum)
                    ini_ind, combi = zip(*enum)
                    label = ini_ind.index(0)
                    self.samples.append((combination[0], *combi, label,) + feats)

        if self.h_qa is not None:
            self.h_qa.close()
        if self.h_v is not None:
            self.h_v.close()
        if self.h_t is not None:
            self.h_t.close()
        if self.h_ac is not None:
            self.h_ac.close()

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)
