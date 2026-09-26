"""
30x50格子上のA・B・S相分離シミュレーション(Gillespie法 / Kinetic Monte Carlo版)

鳥谷部先生の相分離コード(格子ガス+Gillespieの直接法)の枠組みをそのまま使い、
A(固定・濃度勾配・活性/不活性)とB, S(移動・相互作用で駆動)の3成分系に拡張したもの。

- 格子: 30x50(x方向30マス x y方向50マス)。各格子点は「空」または粒子1個(A/B/S)が占有(重なり不可)。
- A: 固定(移動しない)。上側(y大)に多く、下側(y小)に少ない濃度勾配で初期配置。
     Bが隣接している間、そのAは速度定数 K_OFF x (隣接Bの数) で確率的に不活性化して
     格子から消える(=A-B結合の寿命は平均 1/K_OFF x 隣接数)。接触した瞬間に即消費
     されるわけではないので、Bが凝集体から一時的に離れてもAはまだ活性なままのことが
     あり、戻ってきて再び結合できる。
- B, S: 隣接する空きサイトへホップして移動する(個々の粒子が単独でホップし、
     凝集体からのはみ出し・分裂も自由に起こる)。各移動の起こりやすさ(速度)は
     w = exp(-ΔE / 2)   (ΔE = 移動後の結合エネルギー - 移動前の結合エネルギー)
  で決まり、エネルギーが下がる(=より安定な配置になる)移動ほど起こりやすい。
- B, S 同士の位置交換(スワップ): 隣接する2つの可動粒子(B/S)は、空きサイトが
     なくても互いの位置を交換できる。交換前後で2粒子間の結合(交換しても切れない)
     は打ち消し合うため、ΔEは交換で変わる「他の隣接相手との結合」だけで決まる。
     B同士の交換はΔE=0で常に起こりやすく(区別できない粒子の入れ替えなので密な
     凝集体内部でもすり抜けるように自由に混ざる)、B-S交換はEintの差に応じた
     速度になる。これにより、空きサイトが乏しい凝集体の内部でも粒子の入れ替え
     (Sが凝集体の中を移動することも含む)が可能になる。
- クラスタ全体の剛体並進移動: 隣接するB/S粒子は連結成分(クラスタ)とみなし、
     クラスタ全体を1格子分だけ平行移動する候補も反応リストに加える。内部の
     B-B/B-S結合は並進で変化しないため、ΔEは移動でクラスタ境界のBが接する
     活性Aの数がどう変わるかだけで決まる(移動先が他の粒子で塞がれていれば
     不可)。速度は CLUSTER_MOVE_RATE0 x N^-CLUSTER_SIZE_EXPONENT x exp(-ΔE/2)
     (Nはクラスタサイズ)で、大きいクラスタほど動きにくくする。孤立粒子
     (サイズ1)の移動は通常のホップと同一の遷移になり二重計上してしまうため、
     CLUSTER_MIN_SIZE 未満のクラスタは対象外とする。単体粒子のホップ/スワップ
     だけでは表現できない「凝集体そのものの拡散(ブラウン運動)」を可能にする。
  Gillespieの直接法で「次にどのイベント(ホップ/スワップ/クラスタ移動/Aの
  不活性化)が、いつ起こるか」を、すべての速度(レート)を1つの反応リストとして
  扱い毎ステップ選ぶ。
- 結合エネルギー Eint: B-B, B-S, B-A(活性なAのみ)を定義。S-S, S-Aは0(相互作用なし)。
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ============================================================
# パラメータ
# ============================================================
LX = 30         # 格子サイズ(x方向)
LY = 70         # 格子サイズ(y方向)
N_A = 70        # A(固定分子)の個数(旧200の約1/4)
N_B = 40        # B(移動分子)の個数
N_S = 1         # S(移動分子)の個数

E_BB = -4.0     # B-B 結合エネルギー(負が大きいほど強い引力)
E_BS = -6.0     # B-S 結合エネルギー(B-Bより強め = Sは凝集体に強く保持される)
E_BA = -5.0     # B-A(活性) 結合エネルギー(Aの多い方へ凝集体を引き寄せる)
# S-S, S-A, A-A は相互作用なし(0のまま)

K_OFF = 0.001     # A-B結合が切れてAが不活性化する速度定数(隣接Bの数に比例)
                # 平均結合寿命 = 1 / (K_OFF x 隣接Bの数)。小さいほど結合が長持ちする。

CLUSTER_MOVE_RATE0 = 1.0       # クラスタ全体並進移動の速度プリファクタ
CLUSTER_SIZE_EXPONENT = 1.0    # クラスタサイズNによる減速指数(rate ∝ N^-この値)
CLUSTER_MIN_SIZE = 3           # これ未満(=孤立粒子)は通常のホップと同一遷移になるため対象外

A_TYPE, B_TYPE, S_TYPE = 0, 1, 2

Eint = np.zeros((3, 3))
Eint[B_TYPE, B_TYPE] = E_BB
Eint[B_TYPE, S_TYPE] = Eint[S_TYPE, B_TYPE] = E_BS
Eint[B_TYPE, A_TYPE] = Eint[A_TYPE, B_TYPE] = E_BA

mx = [1, 0, -1, 0]
my = [0, 1, 0, -1]


# ============================================================
# 初期化
# ============================================================
def init_state(seed=-1):
    if seed >= 0:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    occ_type = np.full((LX, LY), -1, dtype=int)   # -1=空, 0=A, 1=B, 2=S
    occ_idx = np.full((LX, LY), -1, dtype=int)     # B/Sのみ: 動く粒子配列内のindex

    # --- A: 濃度勾配(上ほど密度が高い)で配置 ---
    all_cells = np.arange(LX * LY)
    ys_of_cell = all_cells % LY
    weights = (np.exp(0.1 * ys_of_cell) + 1).astype(float)      # y(上)が大きいほど選ばれやすい
    weights /= weights.sum()
    a_cells = rng.choice(all_cells, size=N_A, replace=False, p=weights)
    pos_A = np.stack([a_cells // LY, a_cells % LY], axis=1)
    for px, py in pos_A:
        occ_type[px, py] = A_TYPE
    A_present = np.ones(N_A, dtype=bool)
    A_pos_to_idx = {(int(px), int(py)): i for i, (px, py) in enumerate(pos_A)}

    # --- B, S: 中央付近の空きセルにまとめて配置(初期から凝集体を作る) ---
    cx, cy = LX // 2, LY // 2
    n_mobile = N_B + N_S
    seen, candidates = set(), []
    r = 0
    while len(candidates) < n_mobile:
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                px, py = (cx + dx) % LX, (cy + dy) % LY
                if (px, py) not in seen:
                    seen.add((px, py))
                    if occ_type[px, py] == -1:
                        candidates.append((px, py))
        r += 1

    placed = candidates[:n_mobile]
    xm = np.array([p[0] for p in placed])
    ym = np.array([p[1] for p in placed])
    # candidates は中心(cx,cy)から近い順に並んでいるため、Sを先頭にして
    # 凝集体の真ん中に配置し、Bをその周囲に配置する。
    spm = np.array([S_TYPE] * N_S + [B_TYPE] * N_B)

    for i in range(n_mobile):
        occ_type[xm[i], ym[i]] = spm[i]
        occ_idx[xm[i], ym[i]] = i

    w = init_rates(xm, ym, spm, occ_type)
    w_swap = init_swap_rates(xm, ym, spm, occ_type, occ_idx)
    wA = init_A_rates(pos_A, occ_type)

    return dict(occ_type=occ_type, occ_idx=occ_idx, xm=xm, ym=ym, spm=spm, w=w, w_swap=w_swap,
                pos_A=pos_A, A_present=A_present, A_pos_to_idx=A_pos_to_idx, wA=wA,
                rng=rng, t=0.0)


# ============================================================
# 結合エネルギーとホップ速度(レート)
# ============================================================
def bond_energy(px, py, species, occ_type):
    e = 0.0
    for m in range(4):
        qx, qy = (px + mx[m]) % LX, (py + my[m]) % LY
        neighbor_type = occ_type[qx, qy]
        if neighbor_type >= 0:
            e += Eint[species, neighbor_type]
    return e


def compute_rates_for(i, xm, ym, spm, occ_type, w):
    ox, oy = xm[i], ym[i]
    e_here = bond_energy(ox, oy, spm[i], occ_type)
    occ_type[ox, oy] = -1          # 自己相互作用を除外するため元位置を一時的に空にする
    for m in range(4):
        nx, ny = (ox + mx[m]) % LX, (oy + my[m]) % LY
        if occ_type[nx, ny] != -1:
            w[i, m] = 0.0
        else:
            e_next = bond_energy(nx, ny, spm[i], occ_type)
            de = e_next - e_here
            w[i, m] = np.exp(-0.5 * de)
    occ_type[ox, oy] = spm[i]      # 元に戻す
    return w


def init_rates(xm, ym, spm, occ_type):
    n = len(xm)
    w = np.ones((n, 4))
    for i in range(n):
        w = compute_rates_for(i, xm, ym, spm, occ_type, w)
    return w


def compute_swap_rate_for(i, xm, ym, spm, occ_type, occ_idx, w_swap):
    """粒子iと、その東(m=0)・北(m=1)隣にいる可動粒子(B/S)との位置交換レートを計算する。
    西(m=2)・南(m=3)方向は使わない(常に0のまま): 隣接ペアは東/北側からのみ記録することで、
    同じペアを両側から二重に反応リストへ載せてしまうのを防いでいる。
    交換しても2粒子間の結合自体は保たれる(隣接のまま)ので、ΔEは互いの「他の隣人」との
    結合の変化分だけで決まる。"""
    ix, iy = xm[i], ym[i]
    for m in (0, 1):
        jx, jy = (ix + mx[m]) % LX, (iy + my[m]) % LY
        jtype = occ_type[jx, jy]
        if jtype != B_TYPE and jtype != S_TYPE:
            w_swap[i, m] = 0.0
            continue
        j = occ_idx[jx, jy]
        e_before = bond_energy(ix, iy, spm[i], occ_type) + bond_energy(jx, jy, spm[j], occ_type)
        occ_type[ix, iy], occ_type[jx, jy] = spm[j], spm[i]   # 仮に交換して評価
        e_after = bond_energy(ix, iy, spm[j], occ_type) + bond_energy(jx, jy, spm[i], occ_type)
        occ_type[ix, iy], occ_type[jx, jy] = spm[i], spm[j]   # 元に戻す
        de = e_after - e_before
        w_swap[i, m] = np.exp(-0.5 * de)
    return w_swap


def init_swap_rates(xm, ym, spm, occ_type, occ_idx):
    n = len(xm)
    w_swap = np.zeros((n, 4))
    for i in range(n):
        w_swap = compute_swap_rate_for(i, xm, ym, spm, occ_type, occ_idx, w_swap)
    return w_swap


def find_clusters(xm, ym, occ_type, occ_idx):
    """隣接するB/S粒子どうしを連結成分(クラスタ)としてまとめる。
    戻り値: 各クラスタを粒子indexのリストとして持つリスト。
    可動粒子数は少ない(N_B+N_S程度)ので、毎ステップこの関数を呼んで
    フルスクラッチで再構築する(差分更新はしない)方針にしている。"""
    n = len(xm)
    visited = np.zeros(n, dtype=bool)
    clusters = []
    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        stack = [start]
        members = []
        while stack:
            p = stack.pop()
            members.append(p)
            px, py = int(xm[p]), int(ym[p])
            for m in range(4):
                qx, qy = (px + mx[m]) % LX, (py + my[m]) % LY
                qtype = occ_type[qx, qy]
                if qtype == B_TYPE or qtype == S_TYPE:
                    q = occ_idx[qx, qy]
                    if not visited[q]:
                        visited[q] = True
                        stack.append(q)
        clusters.append(members)
    return clusters


def A_contact_energy(px, py, occ_type):
    """あるセルの4近傍にいる活性Aとの結合エネルギー(E_BA x 隣接する活性Aの数)。
    Aは固定粒子でありクラスタのメンバーには絶対にならないので、
    compute_rates_for のような自己相互作用除外の仮置き換えは不要。"""
    e = 0.0
    for m in range(4):
        qx, qy = (px + mx[m]) % LX, (py + my[m]) % LY
        if occ_type[qx, qy] == A_TYPE:
            e += E_BA
    return e


def compute_cluster_move_rates(clusters, xm, ym, spm, occ_type):
    """各クラスタ・各方向(4方向)の剛体並進移動レートを計算する。
    内部のB-B/B-S結合は並進で変化しないため、ΔEは境界のBが接する活性A
    との結合エネルギーの変化分だけで決まる(Sは無関係)。移動先が(クラスタ
    自身を除いて)空でなければその方向は不可(レート0のまま)。"""
    n_c = len(clusters)
    w_cluster = np.zeros((n_c, 4))
    for ci, members in enumerate(clusters):
        N = len(members)
        if N < CLUSTER_MIN_SIZE:
            continue
        cells = set((int(xm[p]), int(ym[p])) for p in members)
        for m in range(4):
            dx, dy = mx[m], my[m]
            valid = True
            de = 0.0
            for p in members:
                px, py = int(xm[p]), int(ym[p])
                nx, ny = (px + dx) % LX, (py + dy) % LY
                if (nx, ny) not in cells and occ_type[nx, ny] != -1:
                    valid = False
                    break
                if spm[p] == B_TYPE:
                    de += A_contact_energy(nx, ny, occ_type) - A_contact_energy(px, py, occ_type)
            if valid:
                w_cluster[ci, m] = CLUSTER_MOVE_RATE0 * (N ** (-CLUSTER_SIZE_EXPONENT)) * np.exp(-0.5 * de)
    return w_cluster


def count_B_neighbors(px, py, occ_type):
    n = 0
    for m in range(4):
        qx, qy = (px + mx[m]) % LX, (py + my[m]) % LY
        if occ_type[qx, qy] == B_TYPE:
            n += 1
    return n


def compute_A_rate(px, py, occ_type):
    """Aの不活性化(=結合が切れて消える)速度: K_OFF x 隣接するBの数。"""
    return K_OFF * count_B_neighbors(px, py, occ_type)


def init_A_rates(pos_A, occ_type):
    wA = np.zeros(len(pos_A))
    for i, (px, py) in enumerate(pos_A):
        wA[i] = compute_A_rate(px, py, occ_type)
    return wA


def update_neighbors_rates(px, py, occ_type, occ_idx, xm, ym, spm, w, w_swap, wA, A_pos_to_idx):
    """(px,py)の環境が変わったとき、その4近傍にいる動ける粒子(B,S)のホップ速度・
    スワップ速度と、近傍にいるA(活性なもの)の不活性化速度を更新する。"""
    for m in range(4):
        qx, qy = (px + mx[m]) % LX, (py + my[m]) % LY
        qtype = occ_type[qx, qy]
        if qtype == B_TYPE or qtype == S_TYPE:
            qi = occ_idx[qx, qy]
            w = compute_rates_for(qi, xm, ym, spm, occ_type, w)
            w_swap = compute_swap_rate_for(qi, xm, ym, spm, occ_type, occ_idx, w_swap)
        elif qtype == A_TYPE:
            a_idx = A_pos_to_idx[(qx, qy)]
            wA[a_idx] = compute_A_rate(qx, qy, occ_type)
    return w, w_swap, wA


# ============================================================
# Gillespieの直接法による1ステップ
# ============================================================
def step(state):
    occ_type, occ_idx = state['occ_type'], state['occ_idx']
    xm, ym, spm, w, w_swap = state['xm'], state['ym'], state['spm'], state['w'], state['w_swap']
    pos_A, A_present, A_pos_to_idx = state['pos_A'], state['A_present'], state['A_pos_to_idx']
    wA = state['wA']
    rng = state['rng']

    # クラスタ(連結成分)は可動粒子数が少ないので毎ステップ全再構築する
    clusters = find_clusters(xm, ym, occ_type, occ_idx)
    w_cluster = compute_cluster_move_rates(clusters, xm, ym, spm, occ_type)

    # 反応リスト = [B/Sのホップ] + [B/Sのスワップ] + [クラスタの剛体並進移動] + [Aの不活性化]
    wflat = w.reshape(-1)
    wswap_flat = w_swap.reshape(-1)
    wcluster_flat = w_cluster.reshape(-1)
    n_hop = wflat.size
    n_swap = wswap_flat.size
    n_cluster = wcluster_flat.size
    combined = np.concatenate([wflat, wswap_flat, wcluster_flat, wA])
    wsum = combined.sum()
    wcum = np.cumsum(combined / wsum)

    r = rng.random()
    k = np.argmax(wcum > r)

    if k < n_hop:
        # --- B/Sのホップ(空きサイトへの移動) ---
        p, m = k // 4, k % 4

        old_x, old_y = int(xm[p]), int(ym[p])
        occ_type[old_x, old_y] = -1
        occ_idx[old_x, old_y] = -1

        new_x, new_y = (old_x + mx[m]) % LX, (old_y + my[m]) % LY
        xm[p], ym[p] = new_x, new_y
        occ_type[new_x, new_y] = spm[p]
        occ_idx[new_x, new_y] = p

        # レート再計算: 動いた粒子自身、旧位置・新位置それぞれの隣接粒子(B/Sのホップ・スワップ速度とAの不活性化速度)
        w = compute_rates_for(p, xm, ym, spm, occ_type, w)
        w_swap = compute_swap_rate_for(p, xm, ym, spm, occ_type, occ_idx, w_swap)
        w, w_swap, wA = update_neighbors_rates(old_x, old_y, occ_type, occ_idx, xm, ym, spm, w, w_swap, wA, A_pos_to_idx)
        w, w_swap, wA = update_neighbors_rates(new_x, new_y, occ_type, occ_idx, xm, ym, spm, w, w_swap, wA, A_pos_to_idx)

    elif k < n_hop + n_swap:
        # --- B/Sどうしの位置交換(空きサイトがなくても起こる) ---
        kk = k - n_hop
        p, m = kk // 4, kk % 4

        ix, iy = int(xm[p]), int(ym[p])
        jx, jy = (ix + mx[m]) % LX, (iy + my[m]) % LY
        q = int(occ_idx[jx, jy])

        occ_type[ix, iy], occ_type[jx, jy] = spm[q], spm[p]
        occ_idx[ix, iy], occ_idx[jx, jy] = q, p
        xm[p], ym[p] = jx, jy
        xm[q], ym[q] = ix, iy

        # レート再計算: 交換した2粒子自身と、両方の位置の周囲(B/Sのホップ・スワップ速度とAの不活性化速度)
        w = compute_rates_for(p, xm, ym, spm, occ_type, w)
        w = compute_rates_for(q, xm, ym, spm, occ_type, w)
        w_swap = compute_swap_rate_for(p, xm, ym, spm, occ_type, occ_idx, w_swap)
        w_swap = compute_swap_rate_for(q, xm, ym, spm, occ_type, occ_idx, w_swap)
        w, w_swap, wA = update_neighbors_rates(ix, iy, occ_type, occ_idx, xm, ym, spm, w, w_swap, wA, A_pos_to_idx)
        w, w_swap, wA = update_neighbors_rates(jx, jy, occ_type, occ_idx, xm, ym, spm, w, w_swap, wA, A_pos_to_idx)

    elif k < n_hop + n_swap + n_cluster:
        # --- クラスタ全体の剛体並進移動(空きサイトがなくても、境界のB-Aコンタクト変化だけで決まる) ---
        kk = k - n_hop - n_swap
        ci, m = kk // 4, kk % 4
        members = clusters[ci]
        dx, dy = mx[m], my[m]

        old_cells = [(int(xm[p]), int(ym[p])) for p in members]
        new_cells = [((ox + dx) % LX, (oy + dy) % LY) for (ox, oy) in old_cells]

        # 2段階更新: 先にクラスタの全旧セルを空にしてから、全新セルへ配置する
        # (1粒子ずつ処理すると、まだ動いていない自分自身の旧位置を誤って
        #  「空いている」と判定してしまう恐れがあるため)
        for (ox, oy) in old_cells:
            occ_type[ox, oy] = -1
            occ_idx[ox, oy] = -1
        for p, (nx, ny) in zip(members, new_cells):
            xm[p], ym[p] = nx, ny
            occ_type[nx, ny] = spm[p]
            occ_idx[nx, ny] = p

        # レート再計算: 動いた各粒子自身と、旧位置・新位置それぞれの周囲
        # (クラスタ外の粒子のホップ・スワップ速度、隣接するAの不活性化速度)
        for p in members:
            w = compute_rates_for(p, xm, ym, spm, occ_type, w)
            w_swap = compute_swap_rate_for(p, xm, ym, spm, occ_type, occ_idx, w_swap)
        for (ox, oy) in old_cells:
            w, w_swap, wA = update_neighbors_rates(ox, oy, occ_type, occ_idx, xm, ym, spm, w, w_swap, wA, A_pos_to_idx)
        for (nx, ny) in new_cells:
            w, w_swap, wA = update_neighbors_rates(nx, ny, occ_type, occ_idx, xm, ym, spm, w, w_swap, wA, A_pos_to_idx)

    else:
        # --- Aの不活性化(A-B結合が確率的に切れて消える) ---
        a_idx = k - n_hop - n_swap - n_cluster
        ax, ay = int(pos_A[a_idx, 0]), int(pos_A[a_idx, 1])
        occ_type[ax, ay] = -1
        A_present[a_idx] = False
        wA[a_idx] = 0.0
        w, w_swap, wA = update_neighbors_rates(ax, ay, occ_type, occ_idx, xm, ym, spm, w, w_swap, wA, A_pos_to_idx)

    state['w'] = w
    state['w_swap'] = w_swap
    state['wA'] = wA
    tau = -np.log(rng.random()) / wsum
    state['t'] += tau
    return tau


# ============================================================
# 可視化
# ============================================================
def get_positions(state):
    pos_A, A_present = state['pos_A'], state['A_present']
    xm, ym, spm = state['xm'], state['ym'], state['spm']
    pos_B = np.stack([xm[spm == B_TYPE], ym[spm == B_TYPE]], axis=1)
    pos_S = np.stack([xm[spm == S_TYPE], ym[spm == S_TYPE]], axis=1)
    return pos_A[A_present], pos_A[~A_present], pos_B, pos_S


def save_progress_snapshots(snapshots, filename="progress_snapshots_gillespie.png"):
    """run_animation が記録した snapshots [(step, t, act, inact, pos_B, pos_S), ...] を並べて保存する。"""
    ncols = min(4, len(snapshots))
    nrows = int(np.ceil(len(snapshots) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (step_i, t, act, inact, pos_B, pos_S) in zip(axes, snapshots):
        ax.set_xlim(-0.5, LX - 0.5)
        ax.set_ylim(-0.5, LY - 0.5)
        ax.set_aspect('equal')
        if len(act):
            ax.scatter(act[:, 0], act[:, 1], s=12, c='dimgray')
        if len(inact):
            ax.scatter(inact[:, 0], inact[:, 1], s=12, c='lightgray')
        ax.scatter(pos_B[:, 0], pos_B[:, 1], s=25, c='royalblue')
        ax.scatter(pos_S[:, 0], pos_S[:, 1], s=25, c='orange')
        ax.set_title(f"step {step_i} (t={t:.1f})")

    for ax in axes[len(snapshots):]:
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"saved: {filename}")


def run_animation(state, n_frames=400, steps_per_frame=300, save_path="simulation_gillespie.gif",
                  n_snapshots=8, snapshot_path=None):
    """フレーム0が初期状態、最終フレームが (n_frames-1)*steps_per_frame ステップ後。
    snapshot_path を指定すると、同じ実行の途中経過を n_snapshots 枚(先頭・末尾フレームを含む)
    記録して保存するので、GIF とスナップショットの時間範囲が必ず一致する。"""
    snapshot_frames = set(np.linspace(0, n_frames - 1, n_snapshots, dtype=int).tolist()) if snapshot_path else set()
    snapshots = []

    fig, ax = plt.subplots(figsize=(6 * LX / LY + 1.5, 6))  # 格子の縦横比(LX:LY)に合わせ、右に凡例の幅を足す
    ax.set_xlim(-0.5, LX - 0.5)
    ax.set_ylim(-0.5, LY - 0.5)
    ax.set_aspect('equal')
    ax.set_title("step 0 (t=0.0)")

    scat_A_active = ax.scatter([], [], s=25, c='dimgray', label='A (active)')
    scat_A_inactive = ax.scatter([], [], s=25, c='lightgray', label='A (inactive)')
    scat_B = ax.scatter([], [], s=35, c='royalblue', label='B')
    scat_S = ax.scatter([], [], s=35, c='orange', label='S')
    ax.legend(loc='lower left', bbox_to_anchor=(1.02, 0), fontsize=8)  # 図の外・右下
    fig.tight_layout()

    def init():
        return ()  # 初期描画で update を余計に呼ばせない(呼ぶと steps_per_frame 分進んでしまう)

    def update(frame):
        if frame > 0:
            for _ in range(steps_per_frame):
                step(state)
        act, inact, pos_B, pos_S = get_positions(state)
        if frame in snapshot_frames and not any(s[0] == frame * steps_per_frame for s in snapshots):
            snapshots.append((frame * steps_per_frame, state['t'], act, inact, pos_B, pos_S))
        scat_A_active.set_offsets(act if len(act) else np.empty((0, 2)))
        scat_A_inactive.set_offsets(inact if len(inact) else np.empty((0, 2)))
        scat_B.set_offsets(pos_B)
        scat_S.set_offsets(pos_S)
        ax.set_title(f"step {frame * steps_per_frame} (t={state['t']:.1f})")
        return scat_A_active, scat_A_inactive, scat_B, scat_S

    ani = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init,
                                  interval=30, blit=False)

    if save_path:
        writer = 'ffmpeg' if save_path.lower().endswith('.mp4') else 'pillow'
        ani.save(save_path, writer=writer, fps=30)
        print(f"saved: {save_path}")
        plt.close(fig)  # 複数回実行したときに図が溜まらないようにする
    else:
        plt.show()

    if snapshot_path and snapshots:
        save_progress_snapshots(snapshots, filename=snapshot_path)

    return ani


if __name__ == "__main__":
    matplotlib.use("Agg")  # 画面表示せず、ファイルとして保存する

    N_RUNS = 3        # 同一パラメータでの実行回数
    BASE_SEED = -1    # >=0 なら run 番号 i に seed = BASE_SEED + i を使う(再現可能)。-1 なら毎回ランダム

    for run in range(1, N_RUNS + 1):
        print(f"=== run {run:03d} / {N_RUNS:03d} ===")
        sim_state = init_state(seed=BASE_SEED + run if BASE_SEED >= 0 else -1)

        # 1回の実行で GIF とスナップショットを同時に作る(総ステップ数 = (n_frames-1) * steps_per_frame)
        run_animation(sim_state, n_frames=401, steps_per_frame=1000,
                      save_path=f"simulation_lattice_gas_cluster_{run:03d}.gif",
                      n_snapshots=8, snapshot_path=f"progress_snapshots_lattice_gas_cluster_{run:03d}.png")
