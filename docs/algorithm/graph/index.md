# C++で使えるグラフアルゴリズム詰め合わせ

## 検証済み

- dfs https://algo-method.com/submissions/904060
- bfs https://algo-method.com/submissions/904057
- bfs(パス) https://algo-method.com/submissions/904061
- ダイクストラ法 https://judge.yosupo.jp/submission/130665
- Cycle Detection https://judge.yosupo.jp/submission/130672
- 二部グラフ判定 https://algo-method.com/submissions/904062
- トポロジカルソート https://algo-method.com/submissions/904063
- ベルマンフォード法 https://algo-method.com/submissions/904069
- クラスカル法 https://algo-method.com/submissions/904074
- 重み付きdsu https://atcoder.jp/contests/abc087/submissions/39939393
- 強連結成分分解 https://judge.yosupo.jp/submission/130760
- 最長パスとそのコストを返す https://atcoder.jp/contests/dp/submissions/40288439 https://atcoder.jp/contests/abc291/submissions/40288789
- 負辺がある時の単一始点の最短経路のパス https://algo-method.com/submissions/933064

## できること

- 単一始点の最短経路
- 負辺がある時の単一始点の最短経路とそのパス
- 全対最短経路
- 二部グラフ判定
- 最小全域木
- 閉路検知
- 負閉路検知
- 有向グラフのサイクル検知(DAG判定)
- サイクルの辺を1つ返す
- 連結成分の個数のカウント
- 各連結成分の代表の頂点を返す
- 任意の2点間を行き来できるか
- トポロジカルソート
- 重み付きdsu
- 強連結成分分解
- 最長パスとそのコストを返す

## これから実装予定の機能

### 通常のグラフ関連

### 木構造

- オイラーツアー
- LCA
- HL分解

## 実装したい機能(できるかわからない)

- 重みなし無向グラフの同型判定

https://atcoder.jp/contests/abc232/submissions/35065024

これをもとに実装したい

## コード

```cpp=
// 頂点のクラス
// {to,weight,num}
struct Vertex {
    int to;
    long long weight;
    int num;
};
class Graph {
  private:
    // Node[u][i].to = 頂点uから行ける頂点
    // Node[u][i].weight = その辺の重み
    // Node[u][i].num = 辺の番号
    vector<vector<Vertex>> Node;

    // Edges[i] = i番目に追加された辺
    // Edges[i].first = 重み
    // Edges[i].second = {to,from}
    vector<pair<long long, pair<int, int>>> Edges;
    int V = 0, E = 0;

    // このクラスにおける最大値
    // 最小値は-INF
    long long INF = 1e17;

    // このクラスにおけるMOD
    // void set_modで変更可能
    long long MOD = 998244353;

  public:
    Graph(int n) {
        Node.resize(n);
        V = n;
    }
    Graph(vector<vector<int>> &g, long long weight = 1) {
        V = g.size();
        Node.resize(V);
        for(int i = 0; i < V; i++) {
            for(auto x : g[i]) {
                Node[i].push_back({x, weight, E});
                Edges.push_back({weight, {i, x}});
                E++;
            }
        }
    }
    Graph(vector<vector<pair<int, pair<long long, int>>>> &g) {
        V = g.size();
        Node.resize(V);
        for(int i = 0; i < V; i++) {
            for(auto x : g[i]) {
                Node[i].push_back({x.first, x.second.first, x.second.second});
                Edges.push_back({x.second.first, {i, x.first}});
                E++;
            }
        }
    }

    // u->vとなる重さwの重み付き有向辺を追加する
    // 重みのデフォは1
    // dsuが無効のときO(1),有効時O(α(V))
    // dsu無効化時はtrue,有効化時はdsu上に辺を新しく追加できたかどうかを返す
    // dsu上で辺が追加できないときにNode上でも辺を追加したくないときはdsu_uniteを使うこと
    bool add(int u, int v, long long w = 1) {
        Node[u].push_back({v, w, E});
        Edges.push_back({w, {u, v}});
        E++;
        if(dsu_built) {
            return dsu_unite2(u, v, w);
        } else
            return true;
    }

    // 頂点数を返す
    // O(1)
    int v_size() { return V; }

    // 辺の数の合計を返す
    // 有向グラフのときのみ使用可能
    // 無向グラフの場合は1/2をかけること
    // O(1)
    int e_size() { return E; }

    // グラフの要素を初期化する
    void clear() {
        Node.clear();
        Edges.clear();
        dsu_built = false;
        V = 0, E = 0;
    }

    // グラフの頂点数を変更
    void resize(int v) {
        Node.resize(v);
        V = v;
        if(dsu_built) {
            dsu_build();
        }
    }

    // i番目の辺を返す
    // O(1)
    pair<long long, pair<int, int>> get_edge(int i) { return Edges[i]; }

    // MODの値を変更する
    // O(1)
    void set_mod(long long m) { MOD = m; }

    ///////////////////////////////////////////////////////////////////////////

    // 基本的なグラフアルゴリズム関連のメンバ

  private:
    // void dfsの本体
    void dfs2(int now, vector<bool> &flag) {
        for(auto x : Node[now]) {
            if(!flag[x.to]) {
                flag[x.to] = true;
                dfs2(x.to, flag);
            }
        }
    }

  public:
    // startからたどり着ける頂点を返す
    // O(V+E)
    vector<bool> dfs(int start) {
        vector<bool> ans(V, false);
        dfs2(start, ans);
        return ans;
    }

    // startから各頂点への最小移動回数を返す
    // たどり着けないとき-1を返す
    // O(V+E)
    vector<long long> bfs(int start) {
        vector<long long> ans(V, -1);
        queue<int> q;
        int now = start;
        q.push(now);
        ans[now] = 0;
        while(!q.empty()) {
            now = q.front();
            q.pop();
            for(auto x : Node[now]) {
                if(ans[x.to] == -1) {
                    ans[x.to] = ans[now] + 1;
                    q.push(x.to);
                }
            }
        }
        return ans;
    }

    // startからgorlまでの最小移動パスの配列を返す
    // たどり着けなければ空の配列を返す
    // O(V+E)
    vector<int> bfs(int start, int gorl) {
        vector<int> ans;
        vector<long long> status(V, -1);
        queue<int> q;
        int now = start;
        q.push(now);
        status[now] = now;
        while(!q.empty()) {
            now = q.front();
            q.pop();
            if(now == gorl) {
                break;
            }
            for(auto x : Node[now]) {
                if(status[x.to] == -1) {
                    status[x.to] = now;
                    q.push(x.to);
                }
            }
        }
        if(status[gorl] == -1) {
            return ans;
        }
        now = gorl;
        while(status[now] != now) {
            ans.push_back(now);
            now = status[now];
        }
        ans.push_back(start);
        std::reverse(ans.begin(), ans.end());
        return ans;
    }

    // すべての連結成分について2部グラフかどうかを判定する
    // 2部グラフの場合それぞれの頂点を0か1に塗り分けた配列を返す
    // 2部グラフでない場合空の配列を返す
    // O(V+E)
    vector<int> bipartite_graph() {
        vector<int> color(V, -1);
        queue<int> q;
        int now;
        for(int i = 0; i < V; i++) {
            if(color[i] == -1) {
                q.push(i);
                color[i] = 0;
            }
            while(!q.empty()) {
                now = q.front();
                q.pop();
                for(auto x : Node[now]) {
                    if(color[x.to] == -1) {
                        color[x.to] = !color[now];
                        q.push(x.to);
                    } else {
                        if(color[x.to] == color[now]) {
                            return vector<int>();
                        }
                    }
                }
            }
        }
        return color;
    }

    // 負の辺がないときにstartから各頂点へ移動するときの最小コストを返す
    // たどり着けないときは-1を返す
    // O(ElogV)
    vector<long long> dijkstra(int start) {
        vector<long long> ans(V, -1);
        pair<int, long long> now = {0, start};
        priority_queue<pair<long long, int>> q;
        ans[start] = 0;
        q.push({0, start});
        while(!q.empty()) {
            now = q.top();
            q.pop();
            if(-now.first > ans[now.second]) {
                continue;
            }
            for(auto x : Node[now.second]) {
                if(ans[x.to] == -1 || ans[x.to] > ans[now.second] + x.weight) {
                    ans[x.to] = ans[now.second] + x.weight;
                    q.push({-ans[x.to], x.to});
                }
            }
        }
        return ans;
    }

    // 負の辺がないときにstartからgorlへ移動するときの最短パスと距離を返す
    // たどり着けないときは{空の配列,-1}を返す
    // O(ElogV)
    pair<vector<int>, long long> dijkstra(int start, int gorl) {
        vector<int> ans;
        vector<long long> status(V, -1), ans2(V, -1);
        pair<int, long long> now = {0, start};
        priority_queue<pair<long long, int>> q;
        status[start] = 0;
        ans2[start] = start;
        q.push({0, start});
        while(!q.empty()) {
            now = q.top();
            q.pop();
            if(now.second == gorl) {
                break;
            }
            if(-now.first > status[now.second]) {
                continue;
            }
            for(auto x : Node[now.second]) {
                if(status[x.to] == -1 ||
                   status[x.to] > status[now.second] + x.weight) {
                    status[x.to] = status[now.second] + x.weight;
                    ans2[x.to] = now.second;
                    q.push({-status[x.to], x.to});
                }
            }
        }
        if(ans2[gorl] == -1) {
            return {ans, -1};
        }
        int now2 = gorl;
        while(ans2[now2] != now2) {
            ans.push_back(now2);
            now2 = ans2[now2];
        }
        ans.push_back(start);
        std::reverse(ans.begin(), ans.end());
        return {ans, status[gorl]};
    }

    // 連結なグラフのときのみ使用可能
    // クラスカル法で重みの総和を返し、引数に入れられたグラフをこのグラフの最小全域木に変える
    // グラフはdsu有効化済みで返ってくる
    // O((V+E)+ElogV+Vα(V))
    long long kruskal(Graph &G) {
        G.clear();
        G.resize(V);
        G.dsu_build();
        long long sum;
        vector<pair<long long, pair<int, int>>> edges;
        for(auto x : Edges) {
            edges.push_back(x);
        }
        sort(edges.begin(), edges.end());
        sum = 0;
        for(auto e : edges) {
            if(G.dsu_unite(e.second.first, e.second.second, e.first)) {
                sum += e.first;
            }
        }
        return sum;
    }

    // 連結なグラフのときのみ使用可能
    // クラスカル法で最小全域木の重みの総和を返す
    // O((V+E)+ElogV+Vα(V))
    long long kruskal() {
        Graph G(V);
        G.dsu_build();
        long long sum;
        vector<pair<long long, pair<int, int>>> edges;
        for(auto x : Edges) {
            edges.push_back(x);
        }
        sort(edges.begin(), edges.end());
        sum = 0;
        for(auto e : edges) {
            if(G.dsu_unite(e.second.first, e.second.second)) {
                sum += e.first;
            }
        }
        return sum;
    }

    // 任意の2点間の最短距離を求めるアルゴリズム
    // 負の辺があっても機能する
    // distances[i][j]=iからjへの最短距離となるような配列と負閉路があるかどうかのboolを返す
    // warshall_floyd().second=true -> 負閉路あり
    // O(V^3)
    pair<vector<vector<long long>>, bool> warshall_floyd() {
        vector<vector<long long>> distances(V, vector<long long>(V, INF));
        for(int i = 0; i < V; i++)
            distances[i][i] = 0;
        for(int i = 0; i < V; i++) {
            for(auto x : Node[i]) {
                distances[i][x.to] = min(distances[i][x.to], x.weight);
            }
        }
        for(int i = 0; i < V; i++) {
            for(int j = 0; j < V; j++) {
                for(int k = 0; k < V; k++) {
                    if((distances[j][i] < INF) && (distances[i][k] < INF)) {
                        distances[j][k] =
                            min(distances[j][k],
                                (distances[j][i] + distances[i][k]));
                    }
                }
            }
        }
        bool ans = false;
        for(int i = 0; i < V; i++) {
            if(distances[i][i] < 0)
                ans = true;
        }
        return {distances, ans};
    }

    // 負辺を含むグラフでの単一視点最短経路を求める
    // startから各頂点への最短経路を入れた配列と負閉路が含まれているかのboolを返す
    // 負閉路がある場合はtrueを返す
    // 第一要素はたどり着けないときINF,負閉路でコストを下げられるときは-INFを返す
    // O(VE)
    pair<vector<long long>, bool> bellman_ford(int start) {
        vector<long long> ans(V, INF);
        ans[start] = 0;
        int cnt = 0;
        while(cnt < V) {
            bool end = true;
            for(auto x : Edges) {
                if(ans[x.second.first] != INF &&
                   ans[x.second.first] + x.first < ans[x.second.second]) {
                    ans[x.second.second] = ans[x.second.first] + x.first;
                    end = false;
                }
            }
            if(end)
                break;
            cnt++;
        }
        for(int i = 0; i < V; i++) {
            for(auto x : Edges) {
                if(ans[x.second.first] != INF &&
                   ans[x.second.first] + x.first < ans[x.second.second]) {
                    ans[x.second.second] = -INF;
                }
            }
        }
        return {ans, (cnt == V)};
    }

    // 負辺を含むグラフでの単一視点最短経路を求める
    // startからgorlへ移動するときの最短パスと距離を返す
    // 第一要素はたどり着けないときや不閉路でコストが下げられるときは空の配列を返し、たどり着けるときは最小パスを返す
    // 第二要素はたどり着けないときINF,負閉路でコストを下げられるときは-INFを、そうでないときは距離を返す
    // O(VE)
    pair<vector<int>, long long> bellman_ford(int start, int gorl) {
        vector<long long> ans(V, INF);
        vector<int> pre(V, -1);
        ans[start] = 0;
        pre[start] = start;
        int cnt = 0;
        while(cnt < V) {
            bool end = true;
            for(auto x : Edges) {
                if(ans[x.second.first] != INF &&
                   ans[x.second.first] + x.first < ans[x.second.second]) {
                    ans[x.second.second] = ans[x.second.first] + x.first;
                    pre[x.second.second] = x.second.first;
                    end = false;
                }
            }
            if(end)
                break;
            cnt++;
        }
        for(int i = 0; i < V; i++) {
            for(auto x : Edges) {
                if(ans[x.second.first] != INF &&
                   ans[x.second.first] + x.first < ans[x.second.second]) {
                    ans[x.second.second] = -INF;
                }
            }
        }
        if(ans[gorl] == -INF) {
            return {{}, -INF};
        } else if(ans[gorl] == INF) {
            return {{}, INF};
        } else {
            vector<int> path;
            int now = gorl;
            while(now != pre[now]) {
                path.push_back(now);
                now = pre[now];
            }
            path.push_back(now);
            reverse(path.begin(), path.end());
            return {path, ans[gorl]};
        }
    }

    // トポロジカルソートを行ったあとの配列を返す
    // サイクルを持たない有向グラフにのみ使用可能
    // トポロジカルソートできなかったときは空の配列を返す
    // O(V+E)
    vector<int> topological_sort() {
        Graph g(V);
        for(int i = 0; i < V; i++) {
            for(auto x : Node[i]) {
                g.add(x.to, i);
            }
        }
        vector<int> ans;
        vector<int> v(V, 0);
        for(int i = 0; i < V; i++) {
            v[i] = Node[i].size();
        }
        queue<int> q;
        for(int i = 0; i < V; i++) {
            if(v[i] == 0) {
                q.push(i);
            }
        }
        while(!q.empty()) {
            int now = q.front();
            q.pop();
            ans.push_back(now);
            for(auto x : g[now]) {
                v[x.to]--;
                if(v[x.to] == 0) {
                    q.push(x.to);
                }
            }
        }
        std::reverse(ans.begin(), ans.end());
        if(ans.size() != V)
            return vector<int>();
        else
            return ans;
    }

  private:
    long long longest_path_dfs(int now, vector<int> &dp, vector<int> &to) {
        if(dp[now] != -1)
            return dp[now];
        long long r = 0;
        for(auto x : Node[now]) {
            long long tmp = longest_path_dfs(x.to, dp, to) + x.weight;
            if(r < tmp) {
                r = tmp;
                to[now] = x.to;
            }
        }
        return dp[now] = r;
    }

  public:
    // グラフがDAGのとき、最長パスとコストを返す
    // DAGでなければ空の配列と-1を返す
    // DAGであれば第1引数にパスを、第2引数にコストが入ったペアを返す
    // O(V+E)
    pair<vector<int>, long long> longest_path() {
        vector<int> ans;
        auto v = topological_sort();
        if(v.empty())
            return {{}, -1};
        vector<int> dp(V, -1), to(V, -1);
        for(int i = 0; i < V; i++)
            to[i] = i;
        int Max = 0, idx = -1;
        for(int i = 0; i < V; i++) {
            int tmp = longest_path_dfs(v[i], dp, to);
            if(Max < tmp) {
                Max = tmp;
                idx = v[i];
            }
        }
        ans.push_back(idx);
        while(to[idx] != idx) {
            ans.push_back(to[idx]);
            idx = to[idx];
        }
        return {ans, Max};
    }

    // ここまで基本的なグラフアルゴリズム
    ///////////////////////////////////////////////////////////////////////////
    // Library Checkerに載っている典型アルゴリズム
    // https://judge.yosupo.jp/

  private:
    bool cycle_detection_dfs(int now, vector<int> &used,
                             vector<pair<int, int>> &pre, vector<int> &cycle) {
        used[now] = 1;
        for(auto &x : Node[now]) {
            if(used[x.to] == 0) {
                pre[x.to] = {now, x.num};
                if(cycle_detection_dfs(x.to, used, pre, cycle))
                    return true;
            } else if(used[x.to] == 1) {
                int cur = now;
                cycle.emplace_back(x.num);
                while(cur != x.to) {
                    cycle.emplace_back(pre[cur].second);
                    cur = pre[cur].first;
                }
                return true;
            }
        }
        used[now] = 2;
        return false;
    }

    // 行きがけ順を調べる
    void scc_dfs1(int now, vector<bool> &used, vector<int> &vs) {
        used[now] = true;
        for(auto x : Node[now]) {
            if(!used[x.to]) {
                scc_dfs1(x.to, used, vs);
            }
        }
        vs.push_back(now);
    }

    // 辺を逆にして強連結成分を調べる
    void scc_dfs2(int now, int k, vector<vector<Vertex>> &rG,
                  vector<bool> &used, vector<vector<int>> &ans) {
        used[now] = true;
        ans[k].push_back(now);
        for(auto x : rG[now]) {
            if(!used[x.to]) {
                scc_dfs2(x.to, k, rG, used, ans);
            }
        }
    }

  public:
    // 有向グラフにサイクルがあるかを判定する
    // サイクルがあればそのサイクルのうちの一つを、なければ空の配列を返す
    // ans[i] = i番目に通る辺の番号
    // O(V+E)
    vector<int> cycle_detection() {
        vector<int> used(V, 0), cycle;
        vector<pair<int, int>> pre(V);
        for(int i = 0; i < V; i++) {
            if(used[i] == 0 && cycle_detection_dfs(i, used, pre, cycle)) {
                std::reverse(cycle.begin(), cycle.end());
                return cycle;
            }
        }
        return {};
    }

    // 強連結成分分解を行ったあと、トポロジカルソートしたものを返す
    // O(V+E)
    vector<vector<int>> scc() {
        vector<vector<Vertex>> rG(V);
        for(int i = 0; i < V; i++) {
            for(auto x : Node[i]) {
                rG[x.to].push_back({i, x.weight, x.num});
            }
        }
        vector<bool> used(V, false);
        vector<int> vs;
        for(int i = 0; i < V; i++) {
            if(!used[i]) {
                scc_dfs1(i, used, vs);
            }
        }
        for(int i = 0; i < V; i++) {
            used[i] = false;
        }
        int k = 0;
        vector<vector<int>> ans;
        for(int i = vs.size() - 1; i >= 0; i--) {
            if(!used[vs[i]]) {
                ans.push_back({});
                scc_dfs2(vs[i], k, rG, used, ans);
                k++;
            }
        }
        return ans;
    }

    // ここまでLibrary Checker
    ///////////////////////////////////////////////////////////////////////////
    // 重み付きdsu(UnionFind)関連のメンバ
  private:
    bool dsu_built = false;
    int dsu_N;
    int dsu_num;
    vector<int> dsu_par;
    vector<int> dsu_rank;
    vector<int> dsu_depth;
    unordered_set<int> dsu_ancestor;
    vector<long long> dsu_diff_weight;

    // UnionFind木が生成されているかの判定
    bool dsu_check() {
        if(!dsu_built) {
            cout << "dsu_build()を実行してください" << endl;
            return false;
        } else
            return true;
    }

    // xの木とyの木を結合させる
    // O(α(V))
    bool dsu_unite2(int x, int y, long long w = 1) {
        if(!dsu_check())
            return false;
        w += dsu_weight(x), w -= dsu_weight(y);
        x = dsu_root(x);
        y = dsu_root(y);
        if(x == y)
            return false;
        if(dsu_rank[x] < dsu_rank[y]) {
            dsu_par[x] = y;
            dsu_depth[y] += dsu_depth[x];
            dsu_diff_weight[x] = -w;
            dsu_ancestor.erase(x);
        } else {
            if(dsu_rank[x] == dsu_rank[y]) {
                dsu_rank[x]++;
            }
            dsu_par[y] = x;
            dsu_depth[x] += dsu_depth[y];
            dsu_diff_weight[y] = w;
            dsu_ancestor.erase(y);
        }
        dsu_num--;
        return true;
    }

  public:
    // 現在のNodeからUnionFind木を作成
    // dsu系を使う場合は最初に使用
    // これを実行したあとにGraph.addを使用すると計算量にO(α(V))が足されるため、むやみに構築すべきでない
    // O(V)
    void dsu_build() {

        dsu_par.clear();
        dsu_rank.clear();
        dsu_depth.clear();
        dsu_ancestor.clear();
        dsu_diff_weight.clear();

        dsu_built = true;
        dsu_num = V;
        dsu_par.resize(V);
        dsu_rank.resize(V, 0);
        dsu_depth.resize(V, 1);
        dsu_diff_weight.resize(V, 0);
        for(int i = 0; i < V; i++) {
            dsu_par[i] = i;
            dsu_ancestor.insert(i);
        }
        for(int i = 0; i < V; i++) {
            for(auto x : Node[i]) {
                dsu_unite(i, x.to, x.weight);
            }
        }
    }

    // xの木とyの木を結合させる
    // 結合可能なときNode上に重みwの辺を追加する
    // O(α(V))
    bool dsu_unite(int x, int y, long long w = 1) {
        // flag=falseのときUnionFind木上のみの変更(ただし、変更された内容は代入不可)
        bool flag = true;
        if(flag) {
            if(!dsu_same(x, y))
                return add(x, y, w);
            else
                return false;
        } else {
            return dsu_unite2(x, y, w);
        }
    }

    // 頂点xの親ノードを返す
    // O(α(V))
    int dsu_root(int x) {
        if(!dsu_check())
            return -1;
        if(dsu_par[x] == x)
            return x;
        else {
            int r = dsu_root(dsu_par[x]);
            dsu_diff_weight[x] += dsu_diff_weight[dsu_par[x]];
            return dsu_par[x] = r;
        }
    }

    // xとyが同じ木の中にあるかをboolで返す
    // O(α(V))
    bool dsu_same(int x, int y) {
        if(!dsu_check())
            return false;
        return dsu_root(x) == dsu_root(y);
    }

    // 連結成分の数を返す
    // O(1)
    int dsu_size() {
        if(!dsu_check())
            return -1;
        return dsu_num;
    }

    // xの木の頂点数を返す
    // O(1)
    int dsu_v_size(int x) {
        if(!dsu_check())
            return -1;
        return dsu_depth[x];
    }

    // それぞれの木の親を返す
    // O(1)
    unordered_set<int> dsu_Ancestor() {
        if(!dsu_check())
            return unordered_set<int>();
        return dsu_ancestor;
    }

    // dsuが有効化されているかを返す
    // O(1)
    bool dsu_isenabled() { return dsu_built; }

    // UnionFindの更新をやめる
    // もう一度使用する場合はdsu_build()を実行しなくてはならない
    // O(1)
    void dsu_disable() { dsu_built = false; }

    // xとroot(x)との距離を返す
    // O(α(V))
    long long dsu_weight(int x) {
        dsu_root(x);
        return dsu_diff_weight[x];
    }

    // 頂点yと頂点xの距離を返す
    // 同じ木にないときはINFを返す
    // O(α(V))
    long long dsu_diff(int x, int y) {
        if(dsu_same(x, y))
            return dsu_weight(y) - dsu_weight(x);
        else {
            return INF;
        }
    }

    // dsu関連ここまで
    ///////////////////////////////////////////////////////////////////////////

    // 演算子のオーバーロード

    Graph &operator=(const Graph &x) {
        this->clear();
        this->resize(x.V);
        for(auto &e : x.Edges) {
            this->add(e.second.first, e.second.first, e.first);
        }
        if(x.dsu_built)
            this->dsu_build();
        return *this;
    }

    vector<Vertex> &operator[](int i) { return this->Node[i]; }

    // 演算子のオーバーロードここまで

    ///////////////////////////////////////////////////////////////////////////

    // 自由にオーバーライドしてください
    void solver(){};

    ///////////////////////////////////////////////////////////////////////////
};
```