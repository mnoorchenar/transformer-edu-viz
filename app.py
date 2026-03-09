# app.py
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import json, threading, queue, math, random
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
import sacrebleu

from transformer.data import (PAIRS, TEST_SENTENCES, build_vocab, TranslationDataset, collate_fn,
                               encode, decode, preprocess_input,
                               PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX)
from transformer.model import Transformer, LabelSmoothingLoss, WarmupScheduler

app = Flask(__name__)
DEVICE = torch.device('cpu')

G = {
    'pairs': None, 'train_data': None, 'val_data': None,
    'src_vocab': None, 'tgt_vocab': None, 'src_inv': None, 'tgt_inv': None,
    'train_loader': None, 'val_loader': None,
    'model': None, 'cfg': None,
    'optimizer': None, 'scheduler': None, 'criterion': None,
    'train_losses': [], 'val_losses': [], 'lr_history': [],
    'current_epoch': 0, 'best_val': float('inf'),
    'bleu_greedy': 0.0, 'bleu_beam': 0.0, 'sample_translations': [],
    'ready': {'data': False, 'vocab': False, 'model': False, 'trained': False},
}
_lock = threading.Lock()
_q = queue.Queue()
_stop = threading.Event()
_thread = None

WATCH_EN = ["the cat is sleeping .", "i see a dog .", "she is reading a book ."]


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/init/data', methods=['POST'])
def init_data():
    body = request.json or {}
    aug = max(1, min(16, int(body.get('augmentation', 12))))
    max_pairs = max(20, min(len(PAIRS), int(body.get('max_pairs', len(PAIRS)))))
    pairs = PAIRS[:max_pairs]
    random.seed(42)
    augmented = pairs * aug
    random.shuffle(augmented)
    split = int(len(augmented) * 0.875)
    with _lock:
        G['pairs'] = pairs
        G['train_data'] = augmented[:split]
        G['val_data'] = augmented[split:]
        G['ready']['data'] = True
    return jsonify({
        'unique_pairs': len(pairs), 'total_available': len(PAIRS),
        'aug_factor': aug, 'total': len(augmented),
        'train_size': split, 'val_size': len(augmented) - split,
        'sample': [[en, fr] for en, fr in pairs[:8]],
    })


@app.route('/api/init/vocab', methods=['POST'])
def init_vocab():
    if not G['ready']['data']:
        return jsonify({'error': 'Initialize data first'}), 400
    sv, si = build_vocab([p[0] for p in G['pairs']])
    tv, ti = build_vocab([p[1] for p in G['pairs']])
    with _lock:
        G['src_vocab'] = sv; G['src_inv'] = si
        G['tgt_vocab'] = tv; G['tgt_inv'] = ti
        G['ready']['vocab'] = True
    return jsonify({
        'src_size': len(sv), 'tgt_size': len(tv),
        'src_sample': dict(list(sv.items())[4:14]),
        'tgt_sample': dict(list(tv.items())[4:14]),
    })


@app.route('/api/init/model', methods=['POST'])
def init_model():
    if not G['ready']['vocab']:
        return jsonify({'error': 'Build vocabulary first'}), 400
    c = request.json or {}
    d_model, num_heads = int(c.get('d_model', 128)), int(c.get('num_heads', 4))
    d_ff, num_layers = int(c.get('d_ff', 256)), int(c.get('num_layers', 2))
    dropout = float(c.get('dropout', 0.1))
    while d_model % num_heads != 0:
        num_heads -= 1
    mdl = Transformer(len(G['src_vocab']), len(G['tgt_vocab']),
                      d_model=d_model, num_heads=num_heads, d_ff=d_ff,
                      num_layers=num_layers, dropout=dropout).to(DEVICE)
    params = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    with _lock:
        G['model'] = mdl
        G['cfg'] = dict(d_model=d_model, num_heads=num_heads, d_ff=d_ff,
                        num_layers=num_layers, dropout=dropout)
        G['train_losses'] = []; G['val_losses'] = []; G['lr_history'] = []
        G['current_epoch'] = 0; G['best_val'] = float('inf')
        G['ready']['model'] = True; G['ready']['trained'] = False
    return jsonify({'params': params, 'cfg': G['cfg']})


@app.route('/api/train/start', methods=['POST'])
def train_start():
    global _thread, _stop
    if not G['ready']['model']:
        return jsonify({'error': 'Build model first'}), 400
    if _thread and _thread.is_alive():
        return jsonify({'error': 'Already training'}), 400
    c = request.json or {}
    epochs = int(c.get('epochs', 30))
    batch_size = int(c.get('batch_size', 32))
    warmup = int(c.get('warmup_steps', 100))
    patience = int(c.get('patience', 8))
    smoothing = float(c.get('label_smoothing', 0.05))
    _stop.clear()
    while not _q.empty():
        try: _q.get_nowait()
        except queue.Empty: break
    with _lock:
        G['train_losses'] = []; G['val_losses'] = []; G['lr_history'] = []
        G['current_epoch'] = 0; G['best_val'] = float('inf')
        G['ready']['trained'] = False
        G['train_loader'] = DataLoader(TranslationDataset(G['train_data'], G['src_vocab'], G['tgt_vocab']),
                                       batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        G['val_loader'] = DataLoader(TranslationDataset(G['val_data'], G['src_vocab'], G['tgt_vocab']),
                                     batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        G['criterion'] = LabelSmoothingLoss(len(G['tgt_vocab']), PAD_IDX, smoothing=smoothing)
        G['optimizer'] = torch.optim.Adam(G['model'].parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
        G['scheduler'] = WarmupScheduler(G['optimizer'], d_model=G['cfg']['d_model'], warmup_steps=warmup)
    _thread = threading.Thread(target=_train_worker, args=(epochs, patience), daemon=True)
    _thread.start()
    return jsonify({'status': 'started', 'epochs': epochs})


@app.route('/api/train/stream')
def train_stream():
    def generate():
        while True:
            try:
                msg = _q.get(timeout=20)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg['type'] in ('done', 'stopped', 'error'):
                    break
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'
    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/train/stop', methods=['POST'])
def train_stop():
    _stop.set()
    return jsonify({'status': 'stopping'})


@app.route('/api/test_sentences')
def get_test_sentences():
    return jsonify([
        {'sentence': s, 'description': desc, 'has_oov': oov}
        for s, desc, oov in TEST_SENTENCES
    ])


@app.route('/api/translate', methods=['POST'])
def translate():
    if G['model'] is None:
        return jsonify({'error': 'No model available'}), 400
    sentence = (request.json or {}).get('sentence', '').strip()
    if not sentence:
        return jsonify({'error': 'Empty input'}), 400
    sv, tv, si, ti = G['src_vocab'], G['tgt_vocab'], G['src_inv'], G['tgt_inv']
    pre = preprocess_input(sentence)
    toks = pre.split()
    unknown = [t for t in toks if t not in sv]
    g_out = _greedy(G['model'], sentence, sv, ti)
    b_out = _beam(G['model'], sentence, sv, ti)
    attn = _get_attn(G['model'], sentence, b_out, sv, tv, ti)
    return jsonify({'greedy': g_out, 'beam': b_out,
                    'src_tokens': toks,
                    'tgt_tokens': b_out.split() if b_out else [],
                    'unknown_tokens': unknown,
                    'attention': attn})


@app.route('/api/tokenize', methods=['POST'])
def tokenize_ep():
    sentence = (request.json or {}).get('sentence', '')
    if not G['src_vocab']:
        return jsonify({'error': 'Build vocabulary first'}), 400
    pre = preprocess_input(sentence)
    toks = pre.split()
    return jsonify({'preprocessed': pre, 'tokens': toks,
                    'encoded': encode(pre, G['src_vocab']),
                    'known': [t for t in toks if t in G['src_vocab']],
                    'unknown': [t for t in toks if t not in G['src_vocab']]})


@app.route('/api/state')
def get_state():
    return jsonify({
        'ready': G['ready'], 'train_losses': G['train_losses'],
        'val_losses': G['val_losses'], 'lr_history': G['lr_history'],
        'current_epoch': G['current_epoch'],
        'bleu_greedy': G['bleu_greedy'], 'bleu_beam': G['bleu_beam'],
        'sample_translations': G['sample_translations'], 'cfg': G['cfg'],
        'total_available_pairs': len(PAIRS),
        'params': sum(p.numel() for p in G['model'].parameters() if p.requires_grad) if G['model'] else 0,
    })


# ── Decode helpers ─────────────────────────────────────────────────────────────
def _greedy(model, sentence, sv, ti, max_len=30):
    model.eval()
    s = preprocess_input(sentence)
    with torch.no_grad():
        src = torch.tensor([encode(s, sv)], dtype=torch.long)
        sm = model.make_src_mask(src)
        enc = model.encoder(src, sm)
        out = [SOS_IDX]
        for _ in range(max_len):
            t = torch.tensor([out], dtype=torch.long)
            tm = model.make_tgt_mask(t)
            d = model.decoder(t, enc, sm, tm)
            nx = model.proj(d)[0, -1].argmax().item()
            out.append(nx)
            if nx == EOS_IDX: break
    return decode(out, ti)


def _beam(model, sentence, sv, ti, beam_size=4, max_len=30):
    model.eval()
    s = preprocess_input(sentence)
    with torch.no_grad():
        src = torch.tensor([encode(s, sv)], dtype=torch.long)
        sm = model.make_src_mask(src)
        enc = model.encoder(src, sm)
        beams, done = [(0.0, [SOS_IDX])], []
        for _ in range(max_len):
            cands = []
            for score, toks in beams:
                if toks[-1] == EOS_IDX: done.append((score, toks)); continue
                t = torch.tensor([toks], dtype=torch.long)
                tm = model.make_tgt_mask(t)
                d = model.decoder(t, enc, sm, tm)
                lp = F.log_softmax(model.proj(d)[0, -1], dim=-1)
                tk = lp.topk(beam_size)
                for lv, li in zip(tk.values, tk.indices):
                    cands.append((score + lv.item(), toks + [li.item()]))
            if not cands: break
            cands.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
            beams = cands[:beam_size]
            if all(b[1][-1] == EOS_IDX for b in beams): done.extend(beams); break
        done.extend(beams)
        done.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
    return decode(done[0][1], ti)


def _get_attn(model, sentence, translation, sv, tv, ti):
    if not translation: return []
    model.eval()
    s = preprocess_input(sentence)
    toks = [SOS_IDX] + [tv.get(t, UNK_IDX) for t in translation.split()]
    with torch.no_grad():
        src = torch.tensor([encode(s, sv)], dtype=torch.long)
        tgt = torch.tensor([toks], dtype=torch.long)
        sm = model.make_src_mask(src)
        tm = model.make_tgt_mask(tgt)
        enc = model.encoder(src, sm)
        _, attn = model.decoder(tgt, enc, sm, tm, return_attention=True)
        if attn is None: return []
        return attn[0].mean(0).cpu().tolist()


# ── Training worker ────────────────────────────────────────────────────────────
def _train_worker(epochs, patience):
    mdl = G['model']
    crit, opt, sched = G['criterion'], G['optimizer'], G['scheduler']
    tl, vl = G['train_loader'], G['val_loader']
    sv, ti = G['src_vocab'], G['tgt_inv']
    best_val, pat_ctr, best_state, lr = float('inf'), 0, None, 0.0
    try:
        for epoch in range(1, epochs + 1):
            if _stop.is_set():
                _q.put({'type': 'stopped', 'epoch': epoch}); return
            _q.put({'type': 'epoch_start', 'epoch': epoch, 'total': epochs})
            mdl.train()
            ep_l, ep_n = 0.0, 0
            for bi, (src, tgt) in enumerate(tl):
                if _stop.is_set(): break
                tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
                logits = mdl(src, tgt_in)
                B, S, V = logits.shape
                loss = crit(logits.view(B * S, V), tgt_out.contiguous().view(B * S))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                lr = sched.step(); opt.step()
                n = (tgt_out != PAD_IDX).sum().item()
                ep_l += loss.item() * n; ep_n += n
                _q.put({'type': 'batch', 'epoch': epoch, 'batch': bi + 1,
                        'total_batches': len(tl), 'loss': round(loss.item(), 4), 'lr': round(lr, 7)})
            train_loss = ep_l / ep_n if ep_n else 0
            mdl.eval(); vl_l, vl_n = 0.0, 0
            with torch.no_grad():
                for src, tgt in vl:
                    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
                    logits = mdl(src, tgt_in); B, S, V = logits.shape
                    loss = crit(logits.view(B * S, V), tgt_out.contiguous().view(B * S))
                    n = (tgt_out != PAD_IDX).sum().item()
                    vl_l += loss.item() * n; vl_n += n
            val_loss = vl_l / vl_n if vl_n else 0
            with _lock:
                G['train_losses'].append(round(train_loss, 4))
                G['val_losses'].append(round(val_loss, 4))
                G['lr_history'].append(round(lr, 7))
                G['current_epoch'] = epoch
            samples = [{'en': s, 'greedy': _greedy(mdl, s, sv, ti), 'beam': _beam(mdl, s, sv, ti)} for s in WATCH_EN]
            improved = val_loss < best_val
            if improved:
                best_val = val_loss; pat_ctr = 0
                best_state = {k: v.clone() for k, v in mdl.state_dict().items()}
            else:
                pat_ctr += 1
            _q.put({'type': 'epoch_end', 'epoch': epoch,
                    'train_loss': round(train_loss, 4), 'val_loss': round(val_loss, 4),
                    'train_ppl': round(math.exp(min(train_loss, 10)), 2),
                    'val_ppl': round(math.exp(min(val_loss, 10)), 2),
                    'lr': round(lr, 7), 'improved': improved,
                    'patience': pat_ctr, 'patience_max': patience, 'samples': samples})
            if pat_ctr >= patience:
                if best_state: mdl.load_state_dict(best_state)
                _q.put({'type': 'early_stop', 'epoch': epoch, 'best_val': round(best_val, 4)}); break
        if best_state: mdl.load_state_dict(best_state)
        hg, hb, refs = [], [], []
        for en, fr in G['pairs']:
            hg.append(_greedy(mdl, en, sv, ti)); hb.append(_beam(mdl, en, sv, ti)); refs.append(fr.lower())
        bg = sacrebleu.corpus_bleu(hg, [refs]).score
        bb = sacrebleu.corpus_bleu(hb, [refs]).score
        fsamp = [{'en': en, 'fr_ref': fr, 'greedy': _greedy(mdl, en, sv, ti), 'beam': _beam(mdl, en, sv, ti)}
                 for en, fr in random.sample(G['pairs'], min(10, len(G['pairs'])))]
        with _lock:
            G['bleu_greedy'] = round(bg, 2); G['bleu_beam'] = round(bb, 2)
            G['sample_translations'] = fsamp; G['ready']['trained'] = True
        _q.put({'type': 'done', 'bleu_greedy': round(bg, 2), 'bleu_beam': round(bb, 2), 'final_samples': fsamp})
    except Exception as e:
        _q.put({'type': 'error', 'message': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False, threaded=True)