# 形式的冪級数に関する関数

LibraryCheckerで使用した関数一覧です

## log

$\log(f(x))$を求めます

$O(N\log(N))$

verify:https://judge.yosupo.jp/submission/165548

```cpp=
void log(fps &f){
	ll n = f.size();
	fps fd(n,0),g(n,0);
	for(ll i=0;i<n-1;i++){
   		fd[i] = f[i+1]*(i+1);
   	}
    fd/=f;
    for(ll i=0;i<n-1;i++){
        g[i+1] = fd[i]/(i+1);
    }
    f=g;
}
```

## exp

依存関数:[log](#log)

$\exp(f(x))$を求めます

$O(N\log(N))$

verify:https://judge.yosupo.jp/submission/165551

```cpp=
void exp(fps &f){
	ll n = f.size();
	fps g(n,0);
	g[0]=1;
	for(ll i=1;i<=2*n;i<<=1){
		auto lg = g;
		log(lg);
		lg=-lg;
		lg[0]+=1;
		g=g*(f+lg);
	}
	f=g;
}
```

## pow

$(f(x))^M$を求めます

$O(N\log(N)\log(M))$

場合によっては

verify:https://judge.yosupo.jp/submission/169385

```cpp=
void pow(fps &X, ll N) {
    fps res(X.size(),0);
    res[0] = 1;
    while(N) {
        if(N & 1) {
            res *= X;
        }
        X *= X;
        N >>= 1;
    }
    X = res;
}
```

## pow2

依存関数:[log](#log) [exp](#exp)

$(f(x))^M$を求めます

$O(N\log(N))$

$[x^0] = 1$のとき限定で使用可能

verify:無し

```cpp=
void pow2(fps &X, ll N){
    log(X);
    for(auto &x:X){
        x*=N;
    }
    exp(X);
}
```

## Subset Sum

依存関数:[log](#log) [exp](#exp)

$\Pi(1+ x^{A_i})$の前から$N$項目までを求めます

$O(M+N\log(N))$

(ただし、$M$は$A$の要素の種類の数)

verify:https://judge.yosupo.jp/submission/169389

```cpp=
fps subset_sum(int N,vector<int> &A){
    vector<int> B(N,0);
    for(auto x:A){
        if(N>x)B[x]++;
    }
    fps f(N,0);
    for(int i=1;i<N;i++){
  	    if(!B[i])continue;
  	    int sign = 1;
  	    for(int j=1;i*j<N;j++){
  		    f[i*j] += B[i]*sign*mint(j).inv();
  		    sign*=-1;
  	    }
    }
    exp(f);
    return f;
}
```