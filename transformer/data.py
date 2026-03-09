# transformer/data.py
import re, random
from collections import Counter
from torch.utils.data import Dataset
import torch

PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<pad>", "<sos>", "<eos>", "<unk>"
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
MAX_SEQ_LEN = 30

PAIRS = [
    ("the cat is sleeping .", "le chat dort ."),
    ("the dog is running .", "le chien court ."),
    ("a man is eating .", "un homme mange ."),
    ("a woman is reading .", "une femme lit ."),
    ("the boy is playing .", "le garçon joue ."),
    ("the girl is singing .", "la fille chante ."),
    ("a child is crying .", "un enfant pleure ."),
    ("the bird is flying .", "l'oiseau vole ."),
    ("a horse is walking .", "un cheval marche ."),
    ("the fish is swimming .", "le poisson nage ."),
    ("a man is sleeping .", "un homme dort ."),
    ("the woman is running .", "la femme court ."),
    ("a dog is eating .", "un chien mange ."),
    ("the cat is playing .", "le chat joue ."),
    ("a girl is reading .", "une fille lit ."),
    ("the boy is crying .", "le garçon pleure ."),
    ("a bird is singing .", "un oiseau chante ."),
    ("the child is walking .", "l'enfant marche ."),
    ("a horse is running .", "un cheval court ."),
    ("the woman is eating .", "la femme mange ."),
    ("the man is reading a book .", "l'homme lit un livre ."),
    ("a woman is eating an apple .", "une femme mange une pomme ."),
    ("the boy is drinking water .", "le garçon boit de l'eau ."),
    ("a girl is drawing a picture .", "une fille dessine un tableau ."),
    ("the cat is drinking milk .", "le chat boit du lait ."),
    ("a dog is chasing a ball .", "un chien chasse une balle ."),
    ("the child is riding a bike .", "l'enfant fait du vélo ."),
    ("a man is driving a car .", "un homme conduit une voiture ."),
    ("the woman is holding a bag .", "la femme tient un sac ."),
    ("a horse is eating grass .", "un cheval mange de l'herbe ."),
    ("two men are running .", "deux hommes courent ."),
    ("three children are playing .", "trois enfants jouent ."),
    ("two women are talking .", "deux femmes parlent ."),
    ("a man and a woman are eating .", "un homme et une femme mangent ."),
    ("two dogs are running in the park .", "deux chiens courent dans le parc ."),
    ("the children are playing in the garden .", "les enfants jouent dans le jardin ."),
    ("a woman is sitting on a chair .", "une femme est assise sur une chaise ."),
    ("the man is standing near the door .", "l'homme se tient près de la porte ."),
    ("a cat is sleeping on the sofa .", "un chat dort sur le canapé ."),
    ("the dog is running in the park .", "le chien court dans le parc ."),
    ("the cat is black .", "le chat est noir ."),
    ("the dog is big .", "le chien est grand ."),
    ("the girl is happy .", "la fille est heureuse ."),
    ("the boy is tired .", "le garçon est fatigué ."),
    ("the apple is red .", "la pomme est rouge ."),
    ("the sky is blue .", "le ciel est bleu ."),
    ("the grass is green .", "l'herbe est verte ."),
    ("the book is old .", "le livre est vieux ."),
    ("the car is fast .", "la voiture est rapide ."),
    ("the house is big .", "la maison est grande ."),
    ("i see a cat .", "je vois un chat ."),
    ("i see a dog .", "je vois un chien ."),
    ("i eat an apple .", "je mange une pomme ."),
    ("i read a book .", "je lis un livre ."),
    ("i drink water .", "je bois de l'eau ."),
    ("i drive a car .", "je conduis une voiture ."),
    ("i like cats .", "j'aime les chats ."),
    ("i like dogs .", "j'aime les chiens ."),
    ("i see a bird .", "je vois un oiseau ."),
    ("i ride a bike .", "je fais du vélo ."),
    ("she is reading a book .", "elle lit un livre ."),
    ("he is eating an apple .", "il mange une pomme ."),
    ("she is drinking water .", "elle boit de l'eau ."),
    ("he is driving a car .", "il conduit une voiture ."),
    ("she is holding a bag .", "elle tient un sac ."),
    ("he is riding a bike .", "il fait du vélo ."),
    ("she is singing a song .", "elle chante une chanson ."),
    ("he is playing football .", "il joue au football ."),
    ("she is drawing a picture .", "elle dessine un tableau ."),
    ("he is sleeping on the sofa .", "il dort sur le canapé ."),
    ("the man is happy .", "l'homme est heureux ."),
    ("the woman is tired .", "la femme est fatiguée ."),
    ("the cat is small .", "le chat est petit ."),
    ("the dog is fast .", "le chien est rapide ."),
    ("the child is young .", "l'enfant est jeune ."),
    ("the horse is big .", "le cheval est grand ."),
    ("the bird is small .", "l'oiseau est petit ."),
    ("the book is new .", "le livre est nouveau ."),
    ("the apple is sweet .", "la pomme est sucrée ."),
    ("the water is cold .", "l'eau est froide ."),
]


def simple_tokenize(text):
    return re.sub(r"([.!?,;:])", r" \1 ", text.lower().strip()).split()


def preprocess_input(sentence):
    s = sentence.lower().strip()
    if s and s[-1] not in '.!?':
        s += ' .'
    return ' '.join(re.sub(r"([.!?,;:])", r" \1 ", s).split())


def build_vocab(sentences, min_freq=1):
    counter = Counter(tok for s in sentences for tok in simple_tokenize(s))
    vocab = {t: i for i, t in enumerate([PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN])}
    for w, freq in counter.items():
        if freq >= min_freq and w not in vocab:
            vocab[w] = len(vocab)
    return vocab, {i: w for w, i in vocab.items()}


def encode(sentence, vocab):
    return [SOS_IDX] + [vocab.get(t, UNK_IDX) for t in simple_tokenize(sentence)] + [EOS_IDX]


def decode(indices, inv_vocab):
    return " ".join(inv_vocab.get(i, UNK_TOKEN) for i in indices
                    if i not in (PAD_IDX, SOS_IDX, EOS_IDX))


class TranslationDataset(Dataset):
    def __init__(self, pairs, sv, tv):
        self.data = [(encode(en, sv), encode(fr, tv)) for en, fr in pairs
                     if len(encode(en, sv)) <= MAX_SEQ_LEN and len(encode(fr, tv)) <= MAX_SEQ_LEN]

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


def collate_fn(batch):
    sb, tb = zip(*batch)
    sm, tm = max(len(s) for s in sb), max(len(t) for t in tb)
    sp = torch.tensor([s + [PAD_IDX] * (sm - len(s)) for s in sb], dtype=torch.long)
    tp = torch.tensor([t + [PAD_IDX] * (tm - len(t)) for t in tb], dtype=torch.long)
    return sp, tp