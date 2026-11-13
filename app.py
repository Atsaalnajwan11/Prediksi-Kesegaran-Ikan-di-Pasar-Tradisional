import math, random
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Random Forest - Kesegaran Ikan", page_icon="🐟", layout="wide")

FITUR = ["Insang", "Mata", "Aroma", "Tekstur"]
DATA = [
    ("Merah","Jernih","Segar","Kenyal","Segar"), ("Merah","Jernih","Amis","Kenyal","Segar"),
    ("Merah","Keruh","Amis","Kenyal","Segar"), ("Merah","Jernih","Segar","Lembek","Segar"),
    ("Pucat","Jernih","Amis","Kenyal","Segar"), ("Pucat","Keruh","Amis","Lembek","Tidak Segar"),
    ("Pucat","Keruh","Busuk","Lembek","Tidak Segar"), ("Cokelat","Keruh","Busuk","Lembek","Tidak Segar"),
    ("Cokelat","Keruh","Amis","Lembek","Tidak Segar"), ("Cokelat","Jernih","Busuk","Kenyal","Tidak Segar"),
    ("Pucat","Jernih","Segar","Lembek","Segar"), ("Merah","Keruh","Busuk","Lembek","Tidak Segar"),
    ("Pucat","Keruh","Segar","Kenyal","Segar"), ("Merah","Jernih","Amis","Lembek","Segar"),
]
ROWS = [dict(No=i + 1, **dict(zip(FITUR, r[:4])), Kesegaran=r[4]) for i, r in enumerate(DATA)]
KELAS = ["Segar", "Tidak Segar"]

# Hasil yang sama persis dengan dokumen Word (seed 418)
DOKUMEN = [
    dict(idx=[6,6,2,5,3,5,11,9,4,5,13,1,3,7], ch=[["Mata","Aroma"],["Insang","Tekstur"]]),
    dict(idx=[5,11,5,11,7,9,2,1,5,2,10,8,6,2], ch=[["Aroma","Tekstur"],["Insang","Tekstur"]]),
    dict(idx=[5,3,2,2,12,13,11,5,12,4,4,13,3,7], ch=[["Mata","Tekstur"],["Insang","Mata"]]),
]

def f6(x, d=6): return f"{x:.{d}f}".replace(".", ",")
def hitung_ent(c):
    n = sum(c); return -sum(k / n * math.log2(k / n) for k in c if k)
def cnt(rs): return [sum(r["Kesegaran"] == k for r in rs) for k in KELAS]

def build(rs, avail, pilih, nama="1"):
    c = cnt(rs)
    n = dict(nama=nama, rows=rs, seg=c[0], ts=c[1], E=hitung_ent(c), avail=list(avail))
    if 0 in c or not avail:
        n["leaf"] = KELAS[0] if c[0] >= c[1] else KELAS[1]; return n
    ch = pilih(avail); n["chosen"] = ch; n["an"] = []
    for f in ch:
        g, si, vals = n["E"], 0.0, []
        for v in sorted({r[f] for r in rs}):
            s = [r for r in rs if r[f] == v]; k = cnt(s); w = len(s) / len(rs); e = hitung_ent(k)
            g -= w * e; si += -w * math.log2(w)
            vals.append(dict(v=v, n=len(s), seg=k[0], ts=k[1], e=e, si=-w * math.log2(w)))
        n["an"].append(dict(f=f, gain=g, split=si, gr=g / si if si > 1e-12 else 0, vals=vals))
    b = n["an"][0]
    for a in n["an"]:
        if a["gr"] > b["gr"]: b = a
    n["best"] = b["f"]; n["kids"] = {}
    for i, v in enumerate(b["vals"]):
        n["kids"][v["v"]] = build([r for r in rs if r[b["f"]] == v["v"]], [a for a in avail if a != b["f"]], pilih, f"{nama}.{i+1}")
    return n

def buat_forest(mode, seed):
    hasil = []
    rng = random.Random(seed)
    for t in range(3):
        if mode == "Sesuai dokumen Word":
            idx = DOKUMEN[t]["idx"]; q = [list(c) for c in DOKUMEN[t]["ch"]]
            pilih = lambda av, q=q: q.pop(0)
        else:
            idx = [rng.randrange(1, 15) for _ in range(14)]
            pilih = lambda av: sorted(rng.sample(av, min(2, len(av))), key=FITUR.index)
        rs = [ROWS[i - 1] for i in idx]
        hasil.append(dict(idx=idx, root=build(rs, FITUR, pilih)))
    return hasil

def semua(n, l=None):
    l = [] if l is None else l; l.append(n)
    for k in n.get("kids", {}).values(): semua(k, l)
    return l

def prediksi(n, x):
    jalur = [n]
    while "leaf" not in n:
        k = n["kids"].get(x[n["best"]])
        if k is None: return jalur, (KELAS[0] if n["seg"] >= n["ts"] else KELAS[1])
        n = k; jalur.append(n)
    return jalur, n["leaf"]

def dot(root, jalur=None):
    nm = {id(n) for n in jalur} if jalur else set()
    L = ['digraph G {', 'node [fontname="Helvetica"]; edge [fontname="Helvetica"];']
    for n in semua(root):
        o = id(n) in nm; pen = ',penwidth=3,color="#E07B1A"' if o else ""
        if "leaf" in n:
            col = "#7CC66B" if n["leaf"] == "Segar" else "#F4A582"
            L.append(f'"{n["nama"]}" [label="{n["leaf"]}\\n(n={len(n["rows"])})",shape=box,style="rounded,filled",fillcolor="{col}"{pen}];')
        else:
            L.append(f'"{n["nama"]}" [label="{n["nama"]}\\n{n["best"]}",shape=ellipse,style=filled,fillcolor="#92CF52"{pen}];')
            for v, k in n["kids"].items():
                e = id(n) in nm and id(k) in nm
                L.append(f'"{n["nama"]}" -> "{k["nama"]}" [label="{v}"' + (',color="#E07B1A",penwidth=3' if e else "") + "];")
    return "\n".join(L) + "}"

def tampil_node(n, no):
    st.markdown(f"#### Iterasi {no}: Node {n['nama']}" + (" (akar)" if n["nama"] == "1" else ""))
    st.caption(f"{len(n['rows'])} data (No: {', '.join(str(r['No']) for r in n['rows'])}) | {n['seg']} Segar, {n['ts']} Tidak Segar")
    st.markdown(f"**Step 1.** Fitur tersedia: {', '.join(n['avail'])}  \n**Fitur kandidat acak (m = 2): {' dan '.join(n['chosen'])}**")
    st.dataframe(pd.DataFrame([{**{"No": r["No"]}, **{f: r[f] for f in n["chosen"]}, "Kesegaran": r["Kesegaran"]} for r in n["rows"]]), hide_index=True)
    st.markdown("**Step 2.** Entropy, Gain, Split Info, Gain Ratio")
    c, N = [n["seg"], n["ts"]], len(n["rows"])
    st.latex(rf"\text{{Entropy}}(S)=-\frac{{{c[0]}}}{{{N}}}\log_2\frac{{{c[0]}}}{{{N}}}-\frac{{{c[1]}}}{{{N}}}\log_2\frac{{{c[1]}}}{{{N}}}={f6(n['E']).replace(',','{,}')}")
    kol = st.columns(2)
    for col, a in zip(kol, n["an"]):
        with col:
            st.markdown(f"**{a['f']}**")
            st.dataframe(pd.DataFrame([{"Nilai": v["v"], "Jumlah": v["n"], "Segar": v["seg"], "Tidak Segar": v["ts"], "Entropy": f6(v["e"]), "Split Info": f6(v["si"], 5)} for v in a["vals"]]), hide_index=True)
            suku = "+".join(rf"\frac{{{v['n']}}}{{{N}}}\times {f6(v['e']).replace(',','{,}')}" for v in a["vals"])
            st.latex(rf"\text{{Gain}}={f6(n['E']).replace(',','{,}')}-({suku})={f6(a['gain']).replace(',','{,}')}")
            st.latex(rf"\text{{SplitInfo}}={f6(a['split']).replace(',','{,}')}")
            st.latex(rf"\text{{GainRatio}}=\frac{{{f6(a['gain']).replace(',','{,}')}}}{{{f6(a['split']).replace(',','{,}')}}}={f6(a['gr'],5).replace(',','{,}')}")
    b = next(a for a in n["an"] if a["f"] == n["best"]); o = next(a for a in n["an"] if a["f"] != n["best"])
    st.success(f"**Step 3.** Gain Ratio {b['f']} ({f6(b['gr'],5)}) lebih besar dari {o['f']} ({f6(o['gr'],5)}), maka **{b['f']}** menjadi node. Cabang: {', '.join(n['kids'])}.")
    for v, k in n["kids"].items():
        if "leaf" in k:
            st.caption(f"Node {k['nama']} ({b['f']} = {v}): data No {', '.join(str(r['No']) for r in k['rows'])} semuanya {k['leaf']}, jadi menjadi daun.")

# ---------------- Sidebar ----------------
st.sidebar.title("Pengaturan")
mode = st.sidebar.radio("Mode pembentukan hutan", ["Sesuai dokumen Word", "Acak ulang"])
seed = 0
if mode == "Acak ulang":
    if "seed" not in st.session_state: st.session_state.seed = random.randrange(10**6)
    if st.sidebar.button("Bangun ulang Random Forest"): st.session_state.seed = random.randrange(10**6)
    seed = st.session_state.seed
st.sidebar.subheader("Data uji")
X = {f: st.sidebar.selectbox(f, sorted({r[f] for r in ROWS}), index=sorted({r[f] for r in ROWS}).index(v)) for f, v in
     dict(Insang="Merah", Mata="Jernih", Aroma="Amis", Tekstur="Lembek").items()}

POHON = buat_forest(mode, seed)
hasil = [prediksi(p["root"], X) for p in POHON]
suara = [h[1] for h in hasil]; ns = suara.count("Segar")
final = "Segar" if ns >= 2 else "Tidak Segar"

st.title("🐟 Random Forest: Prediksi Kesegaran Ikan di Pasar Tradisional")
st.caption("Data simulasi berdasarkan kriteria organoleptik ikan. 3 pohon, m = 2 fitur acak per node, kriteria Gain Ratio.")
(st.success if final == "Segar" else st.error)(f"Prediksi Random Forest: **{final}** ({ns} pohon Segar, {3 - ns} pohon Tidak Segar)")

tabs = st.tabs(["Dataset", "Pohon 1", "Pohon 2", "Pohon 3", "Hasil Voting", "Teori"])
with tabs[0]:
    st.subheader("Dataset (14 baris)")
    st.dataframe(pd.DataFrame(ROWS), hide_index=True, width="stretch")
    c1, c2, c3 = st.columns(3); c1.metric("Jumlah data", 14); c2.metric("Fitur", 4); c3.metric("Segar / Tidak Segar", "8 / 6")
    st.table(pd.DataFrame({"Fitur": FITUR + ["Kesegaran (target)"], "Nilai": ["Merah, Pucat, Cokelat", "Jernih, Keruh", "Segar, Amis, Busuk", "Kenyal, Lembek", "Segar, Tidak Segar"]}))
for i in range(3):
    with tabs[i + 1]:
        p = POHON[i]; root = p["root"]
        st.subheader(f"Pohon {i+1}")
        st.markdown("**Bootstrap sampling** (Row ID, dengan pengembalian): " + ", ".join(map(str, p["idx"])))
        st.caption("Out-of-bag: " + ", ".join(str(r["No"]) for r in ROWS if r["No"] not in p["idx"]))
        with st.expander("Tabel data hasil bootstrap"):
            st.dataframe(pd.DataFrame(root["rows"]), hide_index=True)
        for k, n in enumerate([n for n in semua(root) if "kids" in n], 1): tampil_node(n, k); st.divider()
        st.markdown("#### Diagram pohon (garis oranye = jalur data uji)")
        st.graphviz_chart(dot(root, hasil[i][0]))
        st.info("Jalur data uji: " + " → ".join(f"{n['best']} = {X[n['best']]}" for n in hasil[i][0] if "kids" in n) + f" → **{hasil[i][1]}**")
with tabs[4]:
    st.subheader("Mayoritas vote")
    st.table(pd.DataFrame([{"Pohon": i + 1, "Fitur akar": POHON[i]["root"]["best"], "Prediksi": hasil[i][1]} for i in range(3)]))
    st.latex(r"\hat{y}=\operatorname{mode}\{h_1(x),h_2(x),h_3(x)\}=\text{" + final + "}")
    st.markdown(f"Dengan data uji **{', '.join(f'{f}: {v}' for f, v in X.items())}**, hasil voting adalah **{final}**.")
with tabs[5]:
    st.subheader("Rumus")
    st.latex(r"\text{Entropy}(S)=\sum_{i=1}^{n}-p_i\log_2 p_i")
    st.latex(r"\text{Gain}(S,A)=\text{Entropy}(S)-\sum_{i=1}^{n}\frac{|S_i|}{|S|}\text{Entropy}(S_i)")
    st.latex(r"\text{SplitInfo}(S,A)=-\sum_{i=1}^{c}\frac{|S_i|}{|S|}\log_2\frac{|S_i|}{|S|}")
    st.latex(r"\text{GainRatio}(S,A)=\frac{\text{Gain}(S,A)}{\text{SplitInfo}(S,A)}")
    st.markdown("1. Bootstrap sampling 14 data (dengan pengembalian).\n2. Di setiap node pilih m = 2 fitur acak.\n3. Fitur dengan Gain Ratio tertinggi menjadi node.\n4. Ulangi sampai data di cabang satu kelas.\n5. Ulangi untuk 3 pohon, lalu voting mayoritas.")
