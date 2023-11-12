# 競プロで使えそうなライブラリ

自分用に使いそうなものをメモしておきます

## UnionFind木

頂点の連結判定や，木の結合，閉路検知を$O(α(N))$(ただし，$α(N)$は逆アッカーマン関数)で行えるデータ構造

UnionFind uf(N); 頂点数Nで宣言

uf.root(x); 頂点xの親ノードを返す

uf.same(x,y); xとyが同じ木の中にあるかをboolで返す

uf.unite(x,y); xの木とyの木を結合させる

uf.size(); 連結成分の数を返す

uf.v_size(x): xの木の頂点数を返す

uf.Ancestor(); それぞれの木の親を返す

```cpp=
class UnionFind {
  private:
    int N;
    int num;
    vector<int> par;
    vector<int> rank;
    vector<int> depth;
    unordered_set<int> ancestor;

  public:
    UnionFind(int n) {
        num = n;
        par.resize(n);
        rank.resize(n, 0);
        depth.resize(n, 1);
        for (int i = 0; i < n; i++) {
            par[i] = i;
            ancestor.insert(i);
        }
    }

    int root(int x) {
        return par[x] == x ? x : par[x] = root(par[x]);
    }

    bool same(int x, int y) {
        return root(x) == root(y);
    }

    bool unite(int x, int y) {
        x = root(x);
        y = root(y);
        if (x == y)
            return false;
        if (rank[x] < rank[y]) {
            par[x] = y;
            depth[y] += depth[x];
            ancestor.erase(x);
        }
        else {
            if (rank[x] == rank[y]) {
                rank[x]++;
            }
            par[y] = x;
            depth[x] += depth[y];
            ancestor.erase(y);
        }
        num--;
        return true;
    }

    int size() {
        return num;
    }

    int v_size(int x) {
        return depth[x];
    }

    unordered_set<int> Ancestor() {
        return ancestor;
    }
};
```

## ランレングス圧縮

文字列Sを，文字+何個その文字が連続するかという情報にするアルゴリズム

計算量は$O(N)$

例

aaabbabcc→a3b2a1b1c2

```cpp=
vector<pair<int, char>> RunLength(string S)
{
    int N = S.size();
    vector<pair<int, char>> memo;
 
    if (N == 1)
    {
        memo.push_back(make_pair(1, S.at(0)));
        return memo;
    }
 
    int tempo = 1;
    for (int i = 1; i < N; i++)
    {
        if (i != N - 1)
        {
            if (S.at(i) == S.at(i - 1))
                tempo++;
            else
            {
                memo.push_back(make_pair(tempo, S.at(i - 1)));
                tempo = 1;
            }
        }
        else
        {
            if (S.at(i) == S.at(i - 1))
            {
                tempo++;
                memo.push_back(make_pair(tempo, S.at(i - 1)));
            }
            else
            {
                memo.push_back(make_pair(tempo, S.at(i - 1)));
                memo.push_back(make_pair(1, S.at(i)));
            }
        }
    }
 
    return memo;
}
```

## エラトステネスの篩

素数かどうかの情報が格納された配列を$O(\log\log N)$で返すアルゴリズム

```cpp=
vector<bool> Eratosthenes(int N){
    vector<bool> ans(N+1,true);
    ans[0]=false;
    ans[1]=false;
    for(int i=2;i*i<=N;i++){
        if(ans[i]){
            for(int j=i*i;j<=N;j+=i){
                ans[j]=false;
            }
        }
    }
    return ans;
}
```

## 素因数分解

素因数分解を行うアルゴリズム

高速素因数分解とは違い，非常に大きい値でも行える．

計算量は$O(\sqrt{N})$

```cpp=
//キーに素因数，値に何乗かを格納した連想配列を返すタイプ
template<class T>
map<T,T>prime_factorization(T N){
    map<T,T> ans;
    T X=N;
    for(T i=2;i*i<=N;i++){
        if(X%i!=0)continue;
        if(X==1)break;
        while(X%i==0){
            ans[i]++;
            X/=i;
        }
    }
    if(X!=1)ans[X]++;
    return ans;
}

//素因数を配列に詰めたタイプ
template<class T>
vector<T>prime_factorization2(T N){
    vector<T> ans;
    T X=N;
    for(T i=2;i*i<=N;i++){
        if(X%i!=0)continue;
        if(X==1)break;
        while(X%i==0){
            ans.push_back(i);
            X/=i;
        }
    }
    if(X!=1)ans.push_back(X);
    return ans;
}
```

## 高速素因数分解

素因数分解を高速で行えるアルゴリズム

前計算で各値の最小の素因数を求めるため，非常に大きい値を素因数分解しようとするとメモリがとんでもないことになる

計算量は前計算が$O(N\log N)$，高速素因数分解が$O(\log N)$

```cpp=
//最初にSPFを実行
vector<int> SPF(int N){
    vector<int> ans(N+1);
    for(int i=0;i<=N;i++){
        ans[i]=i;
    }
    for(int i=2;i*i<=N;i++){
        if(ans[i]==i){
            for(int j=i*i;j<=N;j+=i){
                ans[j]=i;
            }
        }
    }
    return ans;
}

map<int,int> FPF(int X,vector<int> &spf){
    map<int,int> ans;
    while(X!=1){
        ans[spf[X]]++;
        X/=spf[X];
    }
    return ans;
}

//使用例
int main(){
    auto spf=SPF(200000);//入力の最大値まででSPF配列を作成
    int N;
    cin>>N;
    auto ans=FPF(N,spf);
    auto itr=ans.begin();
    for(int i=0;i<ans.size();i++,itr++){
        cout<<(*itr).first<<"^"<<(*itr).second<<endl;
    }
    return 0;
}
```

## ダイクストラ法

グラフの最短経路を求めるのに使える

計算量は$O(E\log V)$ (Eは辺の数，Vは頂点の数)

```cpp=
//頂点のクラス
class node
{
public:
    int cost = 2147483647;
    vector<pair<int, int>> to_v;
    bool dis = false;
};

// 始点から各頂点へ移動した場合の最小コストが格納された配列を返す
// mode:0 無向辺 , mode:1 有向辺
// V:頂点数 , a,b:a から b へつながる辺 , c:cost , S:始点の頂点番号
template <typename T>
vector<node> dijkstra(T V, vector<T> &a, vector<T> &b, vector<T> &c, T S, int mode)
{
    vector<node> Graph(V);
    for (ll i = 0; i < a.size(); i++)
    {
        Graph[a[i]].to_v.push_back(make_pair(b[i], c[i]));
        if (mode == 0)
        {
            Graph[b[i]].to_v.push_back(make_pair(a[i], c[i]));
        }
    }
    Graph[S].cost = 0;
    Graph[S].dis = true;
    priority_queue<pair<T, T>, vector<pair<T, T>>, greater<pair<T, T>>> q;
    q.push(make_pair(0, S));
    while (!q.empty())
    {
        Graph[q.top().second].dis = true;
        for (pair<int, int> x : Graph[q.top().second].to_v)
        {
            if (!Graph[x.first].dis)
            {
                Graph[x.first].cost = min(Graph[x.first].cost, q.top().first + x.second);
                q.push(make_pair(Graph[x.first].cost, x.first));
            }
        }
        q.pop();
    }
    return Graph;
}
```

## 遅延伝搬セグメントツリー

[セグメント木を徹底解説！0から遅延評価やモノイドまで](https://algo-logic.info/segment-tree/)からコードを引っ張ってきた

ちなみにこれはRMQ(区間内の最小の値を取得するクエリ)のセグ木である

```cpp=
/* RMQ：[0,n-1] について、区間ごとの最小値を管理する構造体
    set(int i, T x), build(): i番目の要素をxにセット。まとめてセグ木を構築する。O(n)
    update(i,x): i 番目の要素を x に更新。O(log(n))
    query(a,b): [a,b) での最小の要素を取得。O(log(n))
    find_rightest(a,b,x): [a,b) で x以下の要素を持つ最右位置を求める。O(log(n))
    find_leftest(a,b,x): [a,b) で x以下の要素を持つ最左位置を求める。O(log(n))
*/
template <typename T>
struct RMQ {
    const T e = numeric_limits<T>::max();
    function<T(T, T)> fx = [](T x1, T x2) -> T { return min(x1, x2); };
    int n;
    vector<T> dat;
    RMQ(int n_) : n(), dat(n_ * 4, e) {
        int x = 1;
        while (n_ > x) {
            x *= 2;
        }
        n = x;
    }

    void set(int i, T x) { dat[i + n - 1] = x; }
    void build() {
        for (int k = n - 2; k >= 0; k--) dat[k] = fx(dat[2 * k + 1], dat[2 * k + 2]);
    }

    void update(int i, T x) {
        i += n - 1;
        dat[i] = x;
        while (i > 0) {
            i = (i - 1) / 2;  // parent
            dat[i] = fx(dat[i * 2 + 1], dat[i * 2 + 2]);
        }
    }

    // the minimum element of [a,b)
    T query(int a, int b) { return query_sub(a, b, 0, 0, n); }
    T query_sub(int a, int b, int k, int l, int r) {
        if (r <= a || b <= l) {
            return e;
        } else if (a <= l && r <= b) {
            return dat[k];
        } else {
            T vl = query_sub(a, b, k * 2 + 1, l, (l + r) / 2);
            T vr = query_sub(a, b, k * 2 + 2, (l + r) / 2, r);
            return fx(vl, vr);
        }
    }

    int find_rightest(int a, int b, T x) { return find_rightest_sub(a, b, x, 0, 0, n); }
    int find_leftest(int a, int b, T x) { return find_leftest_sub(a, b, x, 0, 0, n); }
    int find_rightest_sub(int a, int b, T x, int k, int l, int r) {
        if (dat[k] > x || r <= a || b <= l) {  // 自分の値がxより大きい or [a,b)が[l,r)の範囲外ならreturn a-1
            return a - 1;
        } else if (k >= n - 1) {  // 自分が葉ならその位置をreturn
            return (k - (n - 1));
        } else {
            int vr = find_rightest_sub(a, b, x, 2 * k + 2, (l + r) / 2, r);
            if (vr != a - 1) {  // 右の部分木を見て a-1 以外ならreturn
                return vr;
            } else {  // 左の部分木を見て値をreturn
                return find_rightest_sub(a, b, x, 2 * k + 1, l, (l + r) / 2);
            }
        }
    }
    int find_leftest_sub(int a, int b, T x, int k, int l, int r) {
        if (dat[k] > x || r <= a || b <= l) {  // 自分の値がxより大きい or [a,b)が[l,r)の範囲外ならreturn b
            return b;
        } else if (k >= n - 1) {  // 自分が葉ならその位置をreturn
            return (k - (n - 1));
        } else {
            int vl = find_leftest_sub(a, b, x, 2 * k + 1, l, (l + r) / 2);
            if (vl != b) {  // 左の部分木を見て b 以外ならreturn
                return vl;
            } else {  // 右の部分木を見て値をreturn
                return find_leftest_sub(a, b, x, 2 * k + 2, (l + r) / 2, r);
            }
        }
    }
};
```

## フェニック木(BIT)

[Binary Indexed Tree(フェニック木)](https://take44444.github.io/Algorithm-Book/range/bit/main.html)から引っ張ってきた

```cpp=
template <typename T>
struct BinaryIndexedTree {
  int n;
  vector<T> data;

  BinaryIndexedTree(int size) {
    n = ++size;
    data.assign(n, 0);
  }

  // get sum of [0,k]
  T sum(int k) const {
    if (k < 0) return 0;
    T ret = 0;
    for (++k; k > 0; k -= k&(-k)) ret += data[k];
    return ret;
  }

  // getsum of [l,r]
  inline T sum(int l, int r) const { return sum(r) - sum(l-1); }

  // data[k] += x
  void add(int k, T x) {
    for (++k; k < n; k += k&(-k)) data[k] += x;
  }
```

## 繰り返し二乗法

$A^N$を$O(\log N)$で求められるアルゴリズム

実数の累乗だとstd::powlでいいが，行列累乗や多項式の累乗をするときに本領を発揮する

```cpp=
ll Pow(ll X, ll N)
{
    ll ans = 1;
    while (N)
    {
        if (N & 1)
        {
            ans *= X;
        }
        X *= X;
        N >>= 1;
    }
    return ans;
}
```

## 一般化された繰り返し二乗法



```cpp=
template <class T> T pow_t(T X, T (*op)(T, T), T (*e)(), ll N) {
    T ans = e();
    while(N) {
        if(N & 1) {
            ans = op(ans, X);
        }
        X = op(X, X);
        N >>= 1;
    }
    return ans;
}
```

使用例(行列のN乗)

```cpp=
template <class T> T pow_t(T X, T (*op)(T, T), T (*e)(), ll N) {
    T ans = e();
    while(N) {
        if(N & 1) {
            ans = op(ans, X);
        }
        X = op(X, X);
        N >>= 1;
    }
    return ans;
}

vector<vector<ll>> op(vector<vector<ll>> A, vector<vector<ll>> B) {
    ll H, W;
    H = A.size();
    W = B.size();
    vector<vector<ll>> C(H, vector<ll>(W, 0));
    for(ll i = 0; i < H; i++) {
        for(ll j = 0; j < W; j++) {
            for(ll k = 0; k < W; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

ll sz = 2;

vector<vector<ll>> e() {
    vector<vector<ll>> e(sz, vector<ll>(sz, 0));
    rep(i, sz) { e[i][i] = 1; }
    return e;
}

int main() {
    ll N;
    cin >> N;
    sz = N;
    vector<vector<ll>> A(sz, vector<ll>(sz, 0));
    rep(i, N) {
        rep(j, N) { cin >> A[i][j]; }
    }
    ll B;
    cin >> B;
    A = pow_t(A, op, e, B);
    rep(i, N) {
        rep(j, N) { cout << A[i][j] << " "; }
        cout << endl;
    }
    return 0;
}
```

## log2

$\lceil \log_2 N \rceil$ を$O(\log N)$で求める

ダブリング関連でよく見るので一応掲載

```cpp=
int flog2(ll N){ 
    int K=1;
    while((1<<K)<N)K++;
    return K;
}
```

<!-- ## 正方行列のN乗

$N \times N$の正方行列$A$を$b$乗したものを$O(N^3 \log b)$で求める

```cpp=
// N*Nの正方行列Aのb乗を返す
// O(N^3logb)
std::vector<std::vector<long long>> MatrrixPow(long long N, std::vector<std::vector<long long>> &A, long long b) {
    long long bsz = 1;
    while((1LL << bsz) < b) bsz++;
    std::vector<std::vector<std::vector<long long>>> B(bsz, std::vector<std::vector<long long>>(N, std::vector<long long>(N, 0)));
    std::vector<std::vector<long long>> C(N, std::vector<long long>(N, 0));
    for(long long i = 0; i < N; i++) {
        for(long long j = 0; j < N; j++) {
            B[0][i][j] = A[i][j];
        }
    }
    for(long long i = 1; i < bsz; i++) {
        for(auto &x : C) {
            for(auto &y : x) {
                y = 0;
            }
        }
        for(long long j = 0; j < N; j++) {
            for(long long k = 0; k < N; k++) {
                for(long long l = 0; l < N; l++) {
                    C[j][k] += B[i - 1][j][l] * B[i - 1][l][k];
                }
            }
        }
        for(long long j = 0; j < N; j++) {
            for(long long k = 0; k < N; k++) {
                B[i][j][k] = C[j][k];
            }
        }
    }
    std::vector<std::vector<long long>> ans(N, std::vector<long long>(N, 0));
    for(long long i = 0; i < N; i++)
        ans[i][i] = 1;
    for(long long i = 0; i < bsz; i++) {
        if((1LL << i) & b) {
            for(auto &x : C) {
                for(auto &y : x) {
                    y = 0;
                }
            }
            for(long long j = 0; j < N; j++) {
                for(long long k = 0; k < N; k++) {
                    for(long long l = 0; l < N; l++) {
                        C[j][k] += ans[j][l] * B[i][l][k];
                    }
                }
            }
            for(long long j = 0; j < N; j++) {
                for(long long k = 0; k < N; k++) {
                    ans[j][k] = C[j][k];
                }
            }
        }
    }
    return ans;
}
```

## 正方行列のN乗の余り

```cpp=
// N*Nの正方行列Aのb乗の各要素をModで割った余りを返す
// O(N^3logb)
std::vector<std::vector<long long>>
MatrrixPowMod(long long N, std::vector<std::vector<long long>> &A, long long b, long long Mod) {
    long long bsz = 1;
    while((1LL << bsz) < b) bsz++;
    std::vector<std::vector<std::vector<long long>>> B(bsz, std::vector<std::vector<long long>>(N, std::vector<long long>(N, 0)));
    std::vector<std::vector<long long>> C(N, std::vector<long long>(N, 0));
    for(long long i = 0; i < N; i++) {
        for(long long j = 0; j < N; j++) {
            B[0][i][j] = A[i][j] % Mod;
        }
    }
    for(long long i = 1; i < bsz; i++) {
        for(auto &x : C) {
            for(auto &y : x) {
                y = 0;
            }
        }
        for(long long j = 0; j < N; j++) {
            for(long long k = 0; k < N; k++) {
                for(long long l = 0; l < N; l++) {
                    C[j][k] += B[i - 1][j][l] * B[i - 1][l][k];
                    C[j][k] %= Mod;
                }
            }
        }
        for(long long j = 0; j < N; j++) {
            for(long long k = 0; k < N; k++) {
                B[i][j][k] = C[j][k];
            }
        }
    }
    std::vector<std::vector<long long>> ans(N, std::vector<long long>(N, 0));
    for(long long i = 0; i < N; i++)
        ans[i][i] = 1;
    for(long long i = 0; i < bsz; i++) {
        if((1LL << i) & b) {
            for(auto &x : C) {
                for(auto &y : x) {
                    y = 0;
                }
            }
            for(long long j = 0; j < N; j++) {
                for(long long k = 0; k < N; k++) {
                    for(long long l = 0; l < N; l++) {
                        C[j][k] += ans[j][l] * B[i][l][k];
                        C[j][k] %= Mod;
                    }
                }
            }
            for(long long j = 0; j < N; j++) {
                for(long long k = 0; k < N; k++) {
                    ans[j][k] = C[j][k];
                }
            }
        }
    }
    return ans;
}
``` -->

## 最小共通祖先(LCA)

[ダブリングによる木の最近共通祖先（LCA：Lowest Common Ancestor）を求めるアルゴリズム](https://algo-logic.info/lca/)から引っ張ってきた

```cpp=
struct Edge {
    long long to;
};
using Graph = vector<vector<Edge>>;

/* LCA(G, root): 木 G に対する根を root として Lowest Common Ancestor を求める構造体
    query(u,v): u と v の LCA を求める。計算量 O(logn)
    前処理: O(nlogn)時間, O(nlogn)空間
*/
struct LCA {
    vector<vector<int>> parent;  // parent[k][u]:= u の 2^k 先の親
    vector<int> dist;            // root からの距離
    LCA(const Graph &G, int root = 0) { init(G, root); }

    // 初期化
    void init(const Graph &G, int root = 0) {
        int V = G.size();
        int K = 1;
        while ((1 << K) < V) K++;
        parent.assign(K, vector<int>(V, -1));
        dist.assign(V, -1);
        dfs(G, root, -1, 0);
        for (int k = 0; k + 1 < K; k++) {
            for (int v = 0; v < V; v++) {
                if (parent[k][v] < 0) {
                    parent[k + 1][v] = -1;
                } else {
                    parent[k + 1][v] = parent[k][parent[k][v]];
                }
            }
        }
    }

    // 根からの距離と1つ先の頂点を求める
    void dfs(const Graph &G, int v, int p, int d) {
        parent[0][v] = p;
        dist[v] = d;
        for (auto e : G[v]) {
            if (e.to != p) dfs(G, e.to, v, d + 1);
        }
    }

    int query(int u, int v) {
        if (dist[u] < dist[v]) swap(u, v);  // u の方が深いとする
        int K = parent.size();
        // LCA までの距離を同じにする
        for (int k = 0; k < K; k++) {
            if ((dist[u] - dist[v]) >> k & 1) {
                u = parent[k][u];
            }
        }
        // 二分探索で LCA を求める
        if (u == v) return u;
        for (int k = K - 1; k >= 0; k--) {
            if (parent[k][u] != parent[k][v]) {
                u = parent[k][u];
                v = parent[k][v];
            }
        }
        return parent[0][u];
    }
};
```

## 畳み込み

FFTの一種である数論変換(NTT)による畳み込み

```cpp=
//mintはModintである。
//畳み込みをする前にsetup()を実行する。
typedef std::vector<mint> vectorM;//NTT用のmintのベクター型
const int DIVIDE_LIMIT = 23;//99...の有名素数は23回分割統治できる。
mint ROOT[DIVIDE_LIMIT + 1];//[i]は2^i乗根　99...の有名素数の原始根は3で、そこから2^22乗根, 2^21...などをsetup()で計算する。
mint inv_ROOT[DIVIDE_LIMIT + 1];//[i]は2^i乗根の逆数　setup()で計算する。
mint PRIMITIVE_ROOT = 3;

void setup() {
    ROOT[DIVIDE_LIMIT] = modpow(PRIMITIVE_ROOT, (MOD - 1) / modpow(2, 23).val);//99..なら119乗
    inv_ROOT[DIVIDE_LIMIT] = 1 / ROOT[DIVIDE_LIMIT];
    for (int i = DIVIDE_LIMIT - 1; i >= 0; i--) {
        ROOT[i] = ROOT[i + 1] * ROOT[i + 1];
        inv_ROOT[i] = inv_ROOT[i + 1] * inv_ROOT[i + 1];
    }
}

vectorM ntt(const vectorM& f, const int inverse, const int log2_f, const int divide_cnt = DIVIDE_LIMIT) {
    vectorM ret;
    if (f.size() == 1 || divide_cnt == 0) {
        ret.resize(f.size());
        mint zeta = 1;
        for (int i = 0; i < ret.size(); i++) {
            mint now = zeta;
            for (int j = 0; j < f.size(); j++) {
                ret[i] += f[j] * now;
                now *= zeta;
            }
            zeta *= ((inverse == 1) ? ROOT[0] : inv_ROOT[0]);
        }
        return ret;
    }

    vectorM f1(f.size() / 2), f2(f.size() / 2);
    //f1とf2を作る。
    for (int i = 0; i < f.size() / 2; i++) {
        f1[i] = f[i * 2];
        f2[i] = f[i * 2 + 1];
    }

    vectorM f1_dft = ntt(f1, inverse, log2_f - 1, divide_cnt  -1), f2_dft = ntt(f2, inverse, log2_f - 1, divide_cnt - 1);
    ret.resize(f.size());
    mint now = 1;

    for (int i = 0; i < f.size(); i++) {
        ret[i] = f1_dft[i % f1_dft.size()] + now * f2_dft[i % f2_dft.size()];
        now *= ((inverse == 1) ? ROOT[log2_f] : inv_ROOT[log2_f]);
    }
    return ret;
}

//eraseHigh0は高次項が係数ゼロ、vectorから排除するかどうか
vectorM mulp(const vectorM& _f, const vectorM& _g) {
    vectorM f = _f, g = _g;

    //fとgの次数の和以上の最小の2冪-1を次数とする。
    int max_dim = 1, log2_max_dim = 0;
    while (f.size() + g.size() > max_dim) max_dim <<= 1, log2_max_dim++;
    f.resize(max_dim), g.resize(max_dim);
    //多項式fとgのDFT結果を求める。 O(n log n)
    vectorM f_dft = ntt(f, 1, log2_max_dim), g_dft = ntt(g, 1, log2_max_dim);

    //f*gのDFT結果は各f_dftとg_ftの係数の積。O(n)
    vectorM fg_dft(max_dim);
    for (int i = 0; i < max_dim; i++) {
        fg_dft[i] = f_dft[i] * g_dft[i];
    }

    //fg_dftをDFT
    vectorM fg = ntt(fg_dft, -1, log2_max_dim);

    //最後にmax_dimで割る
    for (int i = 0; i < fg.size(); i++) {
        fg[i] = fg[i] / max_dim;
    }
    return fg;
}
```

## string_mod

巨大な数字のmodを取るのに使える

$O(N)$(Nは桁数)

```cpp=
ll string_mod(string s, ll mod) {
    ll rest = 0;
    for(char c : s) {
        ll v = c - '0';
        rest = (rest * 10 + v) % mod;
    }
    return rest;
}
```

## rolling_hash

この人の解答から持ってきた

https://kanpurin.hatenablog.com/entry/2023/01/12/144745

```cpp=
struct RollingHash {
  private:
    using ull = unsigned long long;
    static const ull _mod = 0x1fffffffffffffff;
    static ull _base;
    vector<ull> _hashed, _power;

    inline ull _mul(ull a, ull b) const {
        ull au = a >> 31;
        ull ad = a & ((1UL << 31) - 1);
        ull bu = b >> 31;
        ull bd = b & ((1UL << 31) - 1);
        ull mid = ad * bu + au * bd;
        ull midu = mid >> 30;
        ull midd = mid & ((1UL << 30) - 1);
        ull ans = au * bu * 2 + midu + (midd << 31) + ad * bd;

        ans = (ans >> 61) + (ans & _mod);
        if(ans >= _mod)
            ans -= _mod;
        return ans;
    }

  public:
    RollingHash(const string &s) {
        ll n = s.size();
        _hashed.assign(n + 1, 0);
        _power.assign(n + 1, 0);
        _power[0] = 1;
        for(ll i = 0; i < n; i++) {
            _power[i + 1] = _mul(_power[i], _base);
            _hashed[i + 1] = _mul(_hashed[i], _base) + s[i];
            if(_hashed[i + 1] >= _mod)
                _hashed[i + 1] -= _mod;
        }
    }

    ull get(ll l, ll r) const {
        ull ret = _hashed[r] + _mod - _mul(_hashed[l], _power[r - l]);
        if(ret >= _mod)
            ret -= _mod;
        return ret;
    }

    ull connect(ull h1, ull h2, ll h2len) const {
        ull ret = _mul(h1, _power[h2len]) + h2;
        if(ret >= _mod)
            ret -= _mod;
        return ret;
    }

    void connect(const string &s) {
        ll n = _hashed.size() - 1, m = s.size();
        _hashed.resize(n + m + 1);
        _power.resize(n + m + 1);
        for(ll i = n; i < n + m; i++) {
            _power[i + 1] = _mul(_power[i], _base);
            _hashed[i + 1] = _mul(_hashed[i], _base) + s[i - n];
            if(_hashed[i + 1] >= _mod)
                _hashed[i + 1] -= _mod;
        }
    }

    ll LCP(const RollingHash &b, ll l1, ll r1, ll l2, ll r2) const {
        ll len = min(r1 - l1, r2 - l2);
        ll low = -1, high = len + 1;
        while(high - low > 1) {
            ll mid = (low + high) / 2;
            if(get(l1, l1 + mid) == b.get(l2, l2 + mid))
                low = mid;
            else
                high = mid;
        }
        return low;
    }
};

mt19937_64 mt{(unsigned int)time(NULL)};
RollingHash::ull RollingHash::_base = mt() % RollingHash::_mod;
```
使い方
```cpp=
int main(){
    string S,T;
    cin>>S>>T;
    RollingHash Sr(S),Tr(T);
    if(Sr.get(0,N)==Tr(0,N))cout<<"Same"<<endl;//get(l,r)=[l,r)のハッシュを取得
    else cout<<"diff"<<endl;
    return 0;
}
```

## 仮分数

```cpp=
template <class T> class frac {
    T bunsi, bunbo;
    constexpr void setting() noexcept {
        T g = __gcd(bunsi, bunbo);
        bunsi /= g;
        bunbo /= g;
        if(bunbo < 0) {
            bunsi = -bunsi;
            bunbo = -bunbo;
        }
    }

  public:
    constexpr frac(T Bunsi = 0, T Bunbo = 1) noexcept {
        bunsi = Bunsi;
        bunbo = Bunbo;
        setting();
    }
    constexpr T &Bunsi() noexcept { return bunsi; }
    constexpr const T &Bunsi() const noexcept { return bunsi; }
    constexpr T &Bunbo() noexcept { return bunbo; }
    constexpr const T &Bunbo() const noexcept { return bunbo; }
    constexpr frac<T> &operator+=(const frac<T> &rhs) noexcept {
        bunsi = bunsi * rhs.bunbo + bunbo * rhs.bunsi;
        bunbo *= rhs.bunbo;
        setting();
        return *this;
    }
    constexpr frac<T> &operator-=(const frac<T> &rhs) noexcept {
        bunsi = bunsi * rhs.bunbo - bunbo * rhs.bunsi;
        bunbo *= rhs.bunbo;
        setting();
        return *this;
    }
    constexpr frac<T> &operator*=(const frac<T> &rhs) noexcept {
        bunbo *= rhs.bunbo;
        bunsi *= rhs.bunsi;
        setting();
        return *this;
    }
    constexpr frac<T> &operator/=(const frac<T> &rhs) noexcept {
        bunbo *= rhs.bunsi;
        bunsi *= rhs.bunbo;
        setting();
        return *this;
    }
    constexpr frac<T> operator+(const frac<T> &rhs) const noexcept {
        return frac(*this) += rhs;
    }
    constexpr frac<T> operator-(const frac<T> &rhs) const noexcept {
        return frac(*this) -= rhs;
    }
    constexpr frac<T> operator*(const frac<T> &rhs) const noexcept {
        return frac(*this) *= rhs;
    }
    constexpr frac<T> operator/(const frac<T> &rhs) const noexcept {
        return frac(*this) /= rhs;
    }
    constexpr bool operator<(const frac<T> &rhs) const noexcept {
        return bunsi * rhs.bunbo < bunbo * rhs.bunsi;
    }
    constexpr bool operator>(const frac<T> &rhs) const noexcept {
        return bunsi * rhs.bunbo > bunbo * rhs.bunsi;
    }
    constexpr bool operator>=(const frac<T> &rhs) const noexcept {
        return bunsi * rhs.bunbo >= bunbo * rhs.bunsi;
    }
    constexpr bool operator<=(const frac<T> &rhs) const noexcept {
        return bunsi * rhs.bunbo <= bunbo * rhs.bunsi;
    }
    constexpr bool operator==(const frac<T> &rhs) const noexcept {
        return bunsi * rhs.bunbo == bunbo * rhs.bunsi;
    }
    constexpr bool operator!=(const frac<T> &rhs) const noexcept {
        return bunsi * rhs.bunbo != bunbo * rhs.bunsi;
    }
};
```

## 形式的冪級数

依存ライブラリ <atcoder/modint>,<atcoder/convolution>

このライブラリを使用した関数一覧

https://hackmd.io/@dekavit/BJCHEZAG6

```cpp=
// 形式的冪級数

#define rep(i, r) for(int i = 0; (int)(i) < (int)(r); i++)
#define rep2(i, m, n) for(int i = (m); i < (n); ++i)
#define drep2(i, m, n) for(int i = (m)-1; i >= (n); --i)
#define drep(i, n) drep2(i, n, 0)

template <class T> struct FormalPowerSeries : vector<T> {
    using vector<T>::vector;
    using vector<T>::operator=;
    using F = FormalPowerSeries;

    F operator-() const {
        F res(*this);
        for(auto &e : res)
            e = -e;
        return res;
    }
    F &operator*=(const T &g) {
        for(auto &e : *this)
            e *= g;
        return *this;
    }
    F &operator/=(const T &g) {
        assert(g != T(0));
        *this *= g.inv();
        return *this;
    }
    F &operator+=(const F &g) {
        int n = (*this).size(), m = g.size();
        rep(i, min(n, m))(*this)[i] += g[i];
        return *this;
    }
    F &operator-=(const F &g) {
        int n = (*this).size(), m = g.size();
        rep(i, min(n, m))(*this)[i] -= g[i];
        return *this;
    }
    F &operator<<=(const int d) {
        int n = (*this).size();
        (*this).insert((*this).begin(), d, 0);
        (*this).resize(n);
        return *this;
    }
    F &operator>>=(const int d) {
        int n = (*this).size();
        (*this).erase((*this).begin(), (*this).begin() + min(n, d));
        (*this).resize(n);
        return *this;
    }
    F inv(int d = -1) const {
        int n = (*this).size();
        assert(n != 0 && (*this)[0] != 0);
        if(d == -1)
            d = n;
        assert(d > 0);
        F res{(*this)[0].inv()};
        while(res.size() < d) {
            int m = size(res);
            F f(begin(*this), begin(*this) + min(n, 2 * m));
            F r(res);
            f.resize(2 * m), internal::butterfly(f);
            r.resize(2 * m), internal::butterfly(r);
            rep(i, 2 * m) f[i] *= r[i];
            internal::butterfly_inv(f);
            f.erase(f.begin(), f.begin() + m);
            f.resize(2 * m), internal::butterfly(f);
            rep(i, 2 * m) f[i] *= r[i];
            internal::butterfly_inv(f);
            T iz = T(2 * m).inv();
            iz *= -iz;
            rep(i, m) f[i] *= iz;
            res.insert(res.end(), f.begin(), f.begin() + m);
        }
        return {res.begin(), res.begin() + d};
    }

    // fast: FMT-friendly modulus only
    F &operator*=(const F &g) {
        int n = (*this).size();
        *this = convolution(*this, g);
        (*this).resize(n);
        return *this;
    }
    F &operator/=(const F &g) {
        int n = (*this).size();
        *this = convolution(*this, g.inv(n));
        (*this).resize(n);
        return *this;
    }

    // // naive
    // F &operator*=(const F &g) {
    //   int n = (*this).size(), m = g.size();
    //   drep(i, n) {
    //     (*this)[i] *= g[0];
    //     rep2(j, 1, min(i+1, m)) (*this)[i] += (*this)[i-j] * g[j];
    //   }
    //   return *this;
    // }
    // F &operator/=(const F &g) {
    //   assert(g[0] != T(0));
    //   T ig0 = g[0].inv();
    //   int n = (*this).size(), m = g.size();
    //   rep(i, n) {
    //     rep2(j, 1, min(i+1, m)) (*this)[i] -= (*this)[i-j] * g[j];
    //     (*this)[i] *= ig0;
    //   }
    //   return *this;
    // }

    // sparse
    F &operator*=(vector<pair<int, T>> g) {
        int n = (*this).size();
        auto [d, c] = g.front();
        if(d == 0)
            g.erase(g.begin());
        else
            c = 0;
        drep(i, n) {
            (*this)[i] *= c;
            for(auto &[j, b] : g) {
                if(j > i)
                    break;
                (*this)[i] += (*this)[i - j] * b;
            }
        }
        return *this;
    }
    F &operator/=(vector<pair<int, T>> g) {
        int n = (*this).size();
        auto [d, c] = g.front();
        assert(d == 0 && c != T(0));
        T ic = c.inv();
        g.erase(g.begin());
        rep(i, n) {
            for(auto &[j, b] : g) {
                if(j > i)
                    break;
                (*this)[i] -= (*this)[i - j] * b;
            }
            (*this)[i] *= ic;
        }
        return *this;
    }

    // multiply and divide (1 + cz^d)
    void multiply(const int d, const T c) {
        int n = (*this).size();
        if(c == T(1))
            drep(i, n - d)(*this)[i + d] += (*this)[i];
        else if(c == T(-1))
            drep(i, n - d)(*this)[i + d] -= (*this)[i];
        else
            drep(i, n - d)(*this)[i + d] += (*this)[i] * c;
    }
    void divide(const int d, const T c) {
        int n = (*this).size();
        if(c == T(1))
            rep(i, n - d)(*this)[i + d] -= (*this)[i];
        else if(c == T(-1))
            rep(i, n - d)(*this)[i + d] += (*this)[i];
        else
            rep(i, n - d)(*this)[i + d] -= (*this)[i] * c;
    }

    T eval(const T &a) const {
        T x(1), res(0);
        for(auto e : *this)
            res += e * x, x *= a;
        return res;
    }

    F operator*(const T &g) const { return F(*this) *= g; }
    F operator/(const T &g) const { return F(*this) /= g; }
    F operator+(const F &g) const { return F(*this) += g; }
    F operator-(const F &g) const { return F(*this) -= g; }
    F operator<<(const int d) const { return F(*this) <<= d; }
    F operator>>(const int d) const { return F(*this) >>= d; }
    F operator*(const F &g) const { return F(*this) *= g; }
    F operator/(const F &g) const { return F(*this) /= g; }
    F operator*(vector<pair<int, T>> g) const { return F(*this) *= g; }
    F operator/(vector<pair<int, T>> g) const { return F(*this) /= g; }
};

using mint = modint998244353;
using fps = FormalPowerSeries<mint>;
using sfps = vector<pair<int, mint>>;
```

使い方(部分和問題での例)

```cpp=
int main() {
    ll N, K;
    cin >> N >> K;
    fps f = {1};
    f.resize(K + 1);
    ll A;
    rep(i, N) {
        cin >> A;
        if(A <= K) {
            sfps g = {{0, 1}, {A, 1}};
            f *= g;
        }
    }
    cout << f[K].val() << endl;
    return 0;
}
```

## 多倍長数

```cpp=
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
using namespace boost::multiprecision;
using Bint = cpp_int;//任意長整数
using Real32 = number<cpp_dec_float<32>>;//仮数部が32桁の浮動小数
```

## 三分探索

flag : true→最大値,false→最小値

accuracy : 精度

色々未検証

(https://atcoder.jp/contests/abc292/submissions/39441587)(最大値)や、(https://atcoder.jp/contests/abc279/submissions/36826220)(最小値)を参考にしてね

(https://qiita.com/ganyariya/items/1553ff2bf8d6d7789127)三分探索の記事

```cpp=
double ternary_search(double Min, double Max, double accuracy, bool flag,
                      double (*func)(double x)) {
    double left = Min, right = Max;
    double C1, C2;
    double ans1, ans2, ans3, ans4;
    int sign = (flag ? 1 : -1);
    while((right - left) > accuracy) {
        C1 = (left + ((right - left) / 3.0));
        C2 = right - ((right - left) / 3.0);
        ans1 = func(C1);
        ans2 = func(C2);
        ans3 = func(left);
        ans4 = func(right);
        if(ans1 * sign >= ans2 * sign && ans2 * sign >= ans4 * sign) {
            right = C2;
        } else if(ans3 * sign <= ans1 * sign && ans1 * sign <= ans2 * sign) {
            left = C1;
        } else {
            left = C1;
            right = C2;
        }
    }
    ans1 = func(C1);
    ans2 = func(C2);
    ans3 = func(left);
    ans4 = func(right);
    if(flag) {
        return max(max(ans1, ans2), max(ans3, ans4));
    }
    return min(min(ans1, ans2), min(ans3, ans4));
}

ll A, B;

double solve(double x) { return min(A / cosl(M_PI / 6 - x), B / cosl(x)); }

int main() {
    //ABC292-Fでの例
    cin >> A >> B;
    printf("%.12lf\n", ternary_search(0, M_PI / 6, 1e-10, true, solve));
    return 0;
}
```

## 逆元

https://qiita.com/drken/items/3b4fdf0a78e7a138cd9a より

$ax≡1modM$ となるxを返す。ただし、$GCD(a,M)=1$のときのみ使用可能

$GCD(a,M)\neq 1$のときもなにやらうまいことをするとうまく行くらしい

https://atcoder.jp/contests/abc293/editorial/5966

$O(logM)$


```cpp=
// mod. m での a の逆元 a^{-1} を計算する
long long modinv(long long a, long long m) {
    long long b = m, u = 1, v = 0;
    while(b) {
        long long t = a / b;
        a -= t * b;
        swap(a, b);
        u -= t * v;
        swap(u, v);
    }
    u %= m;
    if(u < 0)
        u += m;
    return u;
}
```

## Mo's argorithm

区間$[l,r)$の答えから区間$[l-1,r+1)$の答えが高速にわかるときかつ、クエリの順番を入れ替えても問題がないときに、区間$[l,r)$に対するクエリを高速で処理できるデータ構造

使い方としては、

1. add(区間を$[l,r)$から$[l,r+1)$にするときにする操作)とdel(区間を$[l,r)$から$[l-1,r)$にするときにする操作)をオーバーロードする、このとき、答えはグローバル変数resに入れることに注意
3. Mo mo(配列の長さ)と、長さQの配列ansを宣言
4. mo.insert(l,r)でクエリを渡す
5. すべてのクエリを渡したあとにmo.build()を実行
6. mo.process()で何番目のクエリを処理したかを取得
7. ans[mo.process()]にresを代入
8. ansを0からQ-1まで出力

https://ei1333.github.io/algorithm/mo.html

このサイトから拝借した

```cpp=
struct Mo {
    vector<int> left, right, order;
    vector<bool> v;
    int width;
    int nl, nr, ptr;

    Mo(int n) : width((int)sqrt((double)n)), nl(0), nr(0), ptr(0), v(n) {}

    void insert(int l, int r) /* [l, r) */
    {
        left.push_back(l);
        right.push_back(r);
    }

    /* ソート */
    void build() {
        order.resize(left.size());
        iota(begin(order), end(order), 0);
        sort(begin(order), end(order), [&](int a, int b) {
            if(left[a] / width != left[b] / width)
                return left[a] < left[b];
            return right[a] < right[b];
        });
    }

    /* クエリを 1 つぶんすすめて, クエリのidを返す */
    int process() {
        if(ptr == order.size())
            return (-1);
        const auto id = order[ptr];
        while(nl > left[id])
            distribute(--nl);
        while(nr < right[id])
            distribute(nr++);
        while(nl < left[id])
            distribute(nl++);
        while(nr > right[id])
            distribute(--nr);
        return (order[ptr++]);
    }

    inline void distribute(int idx) {
        v[idx].flip();
        if(v[idx])
            add(idx);
        else
            del(idx);
    }

    void add(int idx);

    void del(int idx);
};

// edit from here
ll res;
vector<ll> A;
vector<ll> ans;

// ここでオーバーライド
void Mo::add(int id) {}

void Mo::del(int id) {}

/*
クエリを加えるinsert(l,r)は半開区間
build()で構築
int idx=mo.process()してからresをans[idx]に代入
*/
```

実装例

https://atcoder.jp/contests/abc293/submissions/39723643

## 1次元累積和

半閉半開区間の総和を高速に求められるデータ構造

```cpp=
// 累積和
// [l,r)の範囲の総和をO(1)で求められる
// 前計算O(N),クエリの処理O(1)
class cumulative_sum {
  private:
    vector<long long> v;
    int N;

  public:
    cumulative_sum(int n) {
        v.resize(n + 1);
        for(auto &x : v)
            x = 0;
        N = n;
    }
    cumulative_sum(vector<long long> &A) {
        v.resize(A.size() + 1);
        v[0] = 0;
        N = A.size();
        for(int i = 0; i < A.size(); i++)
            v[i + 1] = A[i];
    }
    // v[idx]にxを代入
    // O(1)
    void set(long long idx, long long x) {
        if(idx > 0 && idx <= N)
            v[idx + 1] = x;
    }
    // 累積和を構築
    // O(N)
    void build() {
        v[0] = 0;
        for(int i = 1; i <= N; i++)
            v[i] += v[i - 1];
    }
    // [l,r)の総和を返す
    // O(1)
    long long prod(int l, int r) { return v[r] - v[l]; }
};
```

## 座標圧縮版1次元累積和

巨大かつ疎な配列に対する累積和を取ることができるデータ構造

普通の累積和と比べて計算量は落ちているがlogがつくだけなため、Atcoderではほぼ完全上位互換

verify:https://judge.yosupo.jp/submission/170035 https://atcoder.jp/contests/abc326/submissions/me

```cpp=
// 座標圧縮版の累積和
// [l,r)の範囲の総和をO(logN)で求められる
// 前計算O(NlogN),クエリの処理O(logN)
template<class T>
class cumulative_sum {
  private:
    map<long long,T> v;
    long long Max = numeric_limits<long long>::max();
    long long Min = numeric_limits<long long>::min();
  public:
    cumulative_sum() {
        v[Max] = 0;
        v[Min] = 0;
    }
    cumulative_sum(map<long long,T> &m) {
        for(auto x:m){
            v[x.first] = x.second;
        }
        v[Max] = 0;
        v[Min] = 0;
    }
    // v[idx]にxを代入
    // O(logN)
    void set(long long idx, T x) {
        v[idx] = x;
    }

    // v[idx]にxを加算
    // O(logN)
    void add(long long idx,T x){
        v[idx] += x;
    }
    // v[idx]を削除
    //O(logN)
    void erase(long long idx){
        v.erase(idx);
    }
    // 累積和を構築
    // O(NlogN)
    void build() {
        auto itr = v.begin();
        auto itr2 = itr;
        itr2++;
        for(; itr2 != v.end(); itr++,itr2++)
            (*itr2).second += (*itr).second;
    }
    // [l,r)の総和を返す
    // O(logN)
    long long prod(long long l, long long r) { 
        auto itr = v.lower_bound(l);
        itr--;
        auto itr2 = v.lower_bound(r);
        itr2--;
        return (*itr2).second - (*itr).second;
    }

    T &operator[](long long i){return v[i];}
};
```

## sprintfのstring版

参考 https://pyopyopyo.hatenablog.com/entry/2019/02/08/102456

```cpp=
template <typename... Args>
std::string format(const std::string &fmt, Args... args) {
    size_t len = std::snprintf(nullptr, 0, fmt.c_str(), args...);
    std::vector<char> buf(len + 1);
    std::snprintf(&buf[0], len + 1, fmt.c_str(), args...);
    return std::string(&buf[0], &buf[0] + len);
}
```

## 木上の二点間の最短距離

```cpp=
// c++テンプレ
// https://hackmd.io/J0ohc60AQaKfZJMJdT6sUg
// https://atcoder.github.io/ac-library/document_ja/index.html
#include <atcoder/all>
using namespace atcoder;
#include <bits/stdc++.h>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#define repp(i, l, r) for(int i = (int)(l); i < (int)(r); i++)
#define perp(i, r, l) for(int i = (int)(r); i > (int)(l); i--)
#define rep(i, r) for(int i = 0; (int)(i) < (int)(r); i++)
#define per(i, r) for(int i = (int)(r); (int)(i) > 0; i++)
#define repb(i, r) for(Bint i = 0; (Bint)(i) < (Bint)(r); i++)
#define perb(i, r) for(Bint i = (Bint)(r); (Bint)(i) > 0; i++)
typedef long long ll;
typedef unsigned long long ull;
using namespace std;
using namespace boost::multiprecision;
using Bint = cpp_int;
using Real32 = number<cpp_dec_float<32>>;

pair<ll, ll> op(pair<ll, ll> x, pair<ll, ll> y) {
    if(x.second <= y.second)
        return x;
    else {
        return y;
    }
}
pair<ll, ll> e() { return {1e17, 1e17}; }

void DFS(ll now, ll A, ll depth, ll &t, vector<pair<ll, ll>> &io,
         vector<pair<ll, ll>> &d, vector<ll> &d2, vector<vector<ll>> &G) {
    io[now].first = t;
    d.push_back({now, depth});
    d2[now] = depth;
    for(auto x : G[now]) {
        if(io[x].first == -1) {
            t++;
            DFS(x, now, depth + 1, t, io, d, d2, G);
        }
    }
    d.push_back({A, depth - 1});
    t++;
    io[now].second = t;
}

int main() {
    ll N;
    cin >> N;
    vector<vector<ll>> G(N);
    ll u, v;
    rep(i, N - 1) {
        cin >> u >> v, u--, v--;
        G[u].push_back(v);
        G[v].push_back(u);
    }

    vector<pair<ll, ll>> io(N, {-1, -1}), d;
    vector<ll> d2(N, -1);
    ll t = 0;
    DFS(0, -1, 0, t, io, d, d2, G); // オイラーツアーの生成
    segtree<pair<ll, ll>, op, e> seg(d);
    cin >> u >> v;
    u--, v--;
    cout << d2[u] + d2[v] -
                2 * seg.prod(min(io[u].first, io[v].first),
                             max(io[u].first, io[v].first))
                        .second
         << endl;
    return 0;
}
```

## Fastio

高速で入出力を行えるstream

```cpp=
namespace Fastio{struct Reader{template<typename T>Reader&operator>>(T&x){x=0;short f=1;char c=getchar();while(c<'0'||c>'9'){if(c=='-')f*=-1;c=getchar();}while(c>='0'&&c<='9')x=(x<<3)+(x<<1)+(c^48),c=getchar();x*=f;return*this;}Reader&operator>>(double&x){x=0;double t=0;short f=1,s=0;char c=getchar();while((c<'0'||c>'9')&&c!='.'){if(c=='-')f*=-1;c=getchar();}while(c>='0'&&c<='9'&&c!='.')x=x*10+(c^48),c=getchar();if(c=='.')c=getchar();else{x*=f;return*this;}while(c>='0'&&c<='9')t=t*10+(c^48),s++,c=getchar();while(s--)t/=10.0;x=(x+t)*f;return*this;}Reader&operator>>(long double&x){x=0;long double t=0;short f=1,s=0;char c=getchar();while((c<'0'||c>'9')&&c!='.'){if(c=='-')f*=-1;c=getchar();}while(c>='0'&&c<='9'&&c!='.')x=x*10+(c^48),c=getchar();if(c=='.')c=getchar();else{x*=f;return*this;}while(c>='0'&&c<='9')t=t*10+(c^48),s++,c=getchar();while(s--)t/=10.0;x=(x+t)*f;return*this;}Reader&operator>>(__float128&x){x=0;__float128 t=0;short f=1,s=0;char c=getchar();while((c<'0'||c>'9')&&c!='.'){if(c=='-')f*=-1;c=getchar();}while(c>='0'&&c<='9'&&c!='.')x=x*10+(c^48),c=getchar();if(c=='.')c=getchar();else{x*=f;return*this;}while(c>='0'&&c<='9')t=t*10+(c^48),s++,c=getchar();while(s--)t/=10.0;x=(x+t)*f;return*this;}Reader&operator>>(char&c){c=getchar();while(c==' '||c=='\n'||c=='\r')c=getchar();return*this;}Reader&operator>>(char*str){int len=0;char c=getchar();while(c==' '||c=='\n'||c=='\r')c=getchar();while(c!=' '&&c!='\n'&&c!='\r')str[len++]=c,c=getchar();str[len]='\0';return*this;}Reader&operator>>(string&str){str.clear();char c=getchar();while(c==' '||c=='\n'||c=='\r')c=getchar();while(c!=' '&&c!='\n'&&c!='\r')str.push_back(c),c=getchar();return*this;}Reader(){}}cin;const char endl='\n';struct Writer{const int Setprecision=6;typedef int mxdouble;template<typename T>Writer&operator<<(T x){if(x==0){putchar('0');return*this;}if(x<0)putchar('-'),x=-x;static short sta[40];short top=0;while(x>0)sta[++top]=x%10,x/=10;while(top>0)putchar(sta[top]+'0'),top--;return*this;}Writer&operator<<(double x){if(x<0)putchar('-'),x=-x;mxdouble _=x;x-=(double)_;static short sta[40];short top=0;while(_>0)sta[++top]=_%10,_/=10;if(top==0)putchar('0');while(top>0)putchar(sta[top]+'0'),top--;putchar('.');for(int i=0;i<Setprecision;i++)x*=10;_=x;while(_>0)sta[++top]=_%10,_/=10;for(int i=0;i<Setprecision-top;i++)putchar('0');while(top>0)putchar(sta[top]+'0'),top--;return*this;}Writer&operator<<(long double x){if(x<0)putchar('-'),x=-x;mxdouble _=x;x-=(long double)_;static short sta[40];short top=0;while(_>0)sta[++top]=_%10,_/=10;if(top==0)putchar('0');while(top>0)putchar(sta[top]+'0'),top--;putchar('.');for(int i=0;i<Setprecision;i++)x*=10;_=x;while(_>0)sta[++top]=_%10,_/=10;for(int i=0;i<Setprecision-top;i++)putchar('0');while(top>0)putchar(sta[top]+'0'),top--;return*this;}Writer&operator<<(__float128 x){if(x<0)putchar('-'),x=-x;mxdouble _=x;x-=(__float128)_;static short sta[40];short top=0;while(_>0)sta[++top]=_%10,_/=10;if(top==0)putchar('0');while(top>0)putchar(sta[top]+'0'),top--;putchar('.');for(int i=0;i<Setprecision;i++)x*=10;_=x;while(_>0)sta[++top]=_%10,_/=10;for(int i=0;i<Setprecision-top;i++)putchar('0');while(top>0)putchar(sta[top]+'0'),top--;return*this;}Writer&operator<<(char c){putchar(c);return*this;}Writer&operator<<(char*str){int cur=0;while(str[cur])putchar(str[cur++]);return*this;}Writer&operator<<(const char*str){int cur=0;while(str[cur])putchar(str[cur++]);return*this;}Writer&operator<<(string str){int st=0,ed=str.size();while(st<ed)putchar(str[st++]);return*this;}Writer(){}}cout;}using namespace Fastio;
#define cin Fastio::cin
#define cout Fastio::cout
#define endl Fastio::endl//;fflush(stdout)
```

## MEXライブラリ

MEXの取得を$O(\log{N})$でできるらしい?

https://rsk0315.hatenablog.com/entry/2020/10/11/125049

## コンビネーション

$nCr$の計算を$O(r)$で行える

オーバーフローに注意(__int128_tを使うとマシになるかも)

```cpp=
ll comb(ll n, ll r) {
    ll res = 1;
    if (n / 2 < r) {
        r = n - r;
    }
    for (ll i = 1; i <= r; i++) {
        res *= (n - i + 1);
        res /= i;
    }
    return res;
}
```

## 文字列のスライスっぽいやつ

文字列$S$の[l,r)での連続部分文字列を返す

```cpp=
string sub_string_slice(string &S, int l, int r) {
    return S.substr(l, r - l);
}
```

## 文字列の文字の置き換え

文字列$S$の中のbeforeをafterに置き換える

```cpp=
void string_replace(string &S, string before, string after) {
    S = regex_replace(S, regex(before), after);
}
```

## priority_deque

priority_queueの最大からでも最小からでも取り出せるようにしたクラス

multisetを継承して実装

計算量はpushが$O(\log N)$、それ以外は$O(1)$

Verify:https://judge.yosupo.jp/submission/162816

```cpp=
template <typename T, typename _Compare = std::less<T>,typename _Alloc = std::allocator<T>>
class priority_deque : multiset<T,_Compare,_Alloc>{
    using P = priority_deque;
    public:
        void push(T x){
            (*this).insert(x);
        }
        T top(){
            return (*(*this).begin());
        }
        T lowest(){
            return (*(*this).rbegin());
        }
        T pop_top(){
            T res = top();
            (*this).erase((*this).begin());
            return res;
        }
        T pop_lowest(){
            T res = lowest();
            (*this).erase(--(*this).end());
            return res;
        }
};
```