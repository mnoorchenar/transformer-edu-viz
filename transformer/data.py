# transformer/data.py
import re, random
from collections import Counter
from torch.utils.data import Dataset
import torch

PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<pad>", "<sos>", "<eos>", "<unk>"
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
MAX_SEQ_LEN = 30

# ── 138 training pairs across 8 vocabulary categories ─────────────────────────
PAIRS = [
    # ── CATEGORY 1: Basic subject + verb ──────────────────────────────────────
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

    # ── CATEGORY 2: Extended actions with objects ──────────────────────────────
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

    # ── CATEGORY 3: Plural subjects ────────────────────────────────────────────
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

    # ── CATEGORY 4: Adjective descriptions ────────────────────────────────────
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
    ("the woman is beautiful .", "la femme est belle ."),
    ("the man is tall .", "l'homme est grand ."),
    ("the girl is short .", "la fille est petite ."),
    ("the house is clean .", "la maison est propre ."),
    ("the water is warm .", "l'eau est chaude ."),
    ("the bag is heavy .", "le sac est lourd ."),
    ("the book is light .", "le livre est léger ."),
    ("the sofa is soft .", "le canapé est doux ."),
    ("the sky is bright .", "le ciel est lumineux ."),

    # ── CATEGORY 5: First-person (i) sentences ─────────────────────────────────
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
    ("i am happy .", "je suis heureux ."),
    ("i am tired .", "je suis fatigué ."),
    ("i am young .", "je suis jeune ."),
    ("i walk to school .", "je marche à l'école ."),
    ("i cook dinner .", "je cuisine le dîner ."),
    ("i write a letter .", "j'écris une lettre ."),
    ("i paint a picture .", "je peins un tableau ."),

    # ── CATEGORY 6: Third-person he/she sentences ──────────────────────────────
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
    ("she is beautiful .", "elle est belle ."),
    ("he is tall .", "il est grand ."),
    ("she is cooking .", "elle cuisine ."),
    ("he is writing .", "il écrit ."),
    ("she is dancing .", "elle danse ."),
    ("he is jumping .", "il saute ."),
    ("she is happy .", "elle est heureuse ."),
    ("he is tired .", "il est fatigué ."),
    ("she is swimming .", "elle nage ."),
    ("he is climbing the tree .", "il grimpe à l'arbre ."),

    # ── CATEGORY 7: New action verbs ───────────────────────────────────────────
    ("the boy is jumping .", "le garçon saute ."),
    ("a girl is dancing .", "une fille danse ."),
    ("a woman is cooking .", "une femme cuisine ."),
    ("a man is writing .", "un homme écrit ."),
    ("the woman is painting .", "la femme peint ."),
    ("a boy is climbing the tree .", "un garçon grimpe à l'arbre ."),
    ("the girl is throwing the ball .", "la fille lance la balle ."),
    ("a child is catching the ball .", "un enfant attrape la balle ."),
    ("the man is opening the door .", "l'homme ouvre la porte ."),
    ("a woman is closing the window .", "une femme ferme la fenêtre ."),
    ("the children are swimming in the river .", "les enfants nagent dans la rivière ."),
    ("a dog is jumping .", "un chien saute ."),

    # ── CATEGORY 8: Adverbs + spatial prepositions ─────────────────────────────
    ("the man is running quickly .", "l'homme court vite ."),
    ("the woman is walking slowly .", "la femme marche lentement ."),
    ("the child is singing loudly .", "l'enfant chante fort ."),
    ("the cat is sleeping quietly .", "le chat dort tranquillement ."),
    ("the girl is drawing carefully .", "la fille dessine soigneusement ."),
    ("the horse is running quickly .", "le cheval court vite ."),
    ("the sun is shining .", "le soleil brille ."),
    ("the rain is falling .", "la pluie tombe ."),
    ("a bird is sitting on the tree .", "un oiseau est assis sur l'arbre ."),
    ("the cat is under the table .", "le chat est sous la table ."),
    ("the dog is behind the door .", "le chien est derrière la porte ."),
    ("a child is near the river .", "un enfant est près de la rivière ."),
    ("the woman is in the kitchen .", "la femme est dans la cuisine ."),
    ("the man is at the window .", "l'homme est à la fenêtre ."),
    ("a flower is in the garden .", "une fleur est dans le jardin ."),
    ("a man and a woman are walking .", "un homme et une femme marchent ."),
    ("the cat and the dog are sleeping .", "le chat et le chien dorment ."),
    ("she is cooking in the kitchen .", "elle cuisine dans la cuisine ."),
    ("he is reading near the window .", "il lit près de la fenêtre ."),
    ("the children are dancing near the river .", "les enfants dansent près de la rivière ."),
    ("a dog is sleeping under the table .", "un chien dort sous la table ."),
    ("the bird is sitting on the tree .", "l'oiseau est assis sur l'arbre ."),
]

# ── Held-out test sentences — NEVER added to PAIRS ────────────────────────────
# Each entry: (sentence, description, has_oov)
TEST_SENTENCES = [
    # All tokens present in training vocabulary
    ("the cat is running quickly .", "speed adverb + known subject", False),
    ("two children are swimming in the river .", "plural + new location combo", False),
    ("a woman is cooking in the kitchen .", "he/she + known room", False),
    ("he is opening the window .", "he + new action combo", False),
    ("the bird is sitting on the tree .", "preposition + new location", False),
    ("she is walking slowly in the garden .", "adverb + location combo", False),
    ("a dog is sleeping under the table .", "spatial preposition + furniture", False),
    ("the children are dancing near the river .", "group plural + location", False),
    ("he is painting a beautiful picture .", "he + adjective + object", False),
    ("i am tired and happy .", "first-person + conjunction combo", False),
    # Contains out-of-vocabulary tokens (⚠ OOV demonstration)
    ("the fox is running in the park .", "⚠ OOV: 'fox' not in training vocab", True),
    ("a student is reading a book .", "⚠ OOV: 'student' not in training vocab", True),
    ("the cat is always sleeping .", "⚠ OOV: 'always' not in training vocab", True),
    ("she is very beautiful .", "⚠ OOV: 'very' not in training vocab", True),
    ("an excited dog is jumping .", "⚠ OOV: 'an', 'excited' not in training vocab", True),
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